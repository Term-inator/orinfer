"""Shared AOT row bindings and integer projections for independent encoders."""

from tools.build.files import write_json
from tools.model.compile import primitive


class Projections:
    norm_epsilon = 1e-5

    def emit(self, name, key, build, args, *, vision=False, divisor=1, dynamic=True):
        rows = (self.patches if vision else self.capacity) // divisor
        return self.aot.emit(
            self.kernel_prefix + "/" + name,
            self.kernel_prefix + "-" + key,
            build,
            args,
            rows,
            dynamic=dynamic,
            row_divisor=divisor,
        )

    def row_workspace(self):
        for name, shape in [
            ("RowRanges", [self.capacity, 2]),
            ("Segments", [128, 2]),
            ("SequenceCount", [1]),
        ]:
            self.w.fixed(name, "i32", shape)

    @staticmethod
    def row_spec():
        return dict(
            ranges="RowRanges",
            segments="Segments",
            count="SequenceCount",
            max_sequences=128,
            alignment=32,
        )

    def norm(self, name, source, dest, weight, hidden, *, vision=False, divisor=1):
        from kernels.model.embedding import rms_norm

        self.emit(
            name,
            f"rms-{hidden}-{self.norm_epsilon}",
            lambda: primitive(rms_norm, self.capacity, hidden, epsilon=self.norm_epsilon),
            dict(X=source, W=self.w.immutable(weight, cast="float16"), Y=dest),
            vision=vision,
            divisor=divisor,
        )

    def activation_input(self, name, source, k, *, compressed=True, vision=False, divisor=1):
        from kernels.model.embedding import quantize
        from kernels.model.rotation_a8 import rotate_activation_a8

        self.emit(
            name,
            f"a8-{k}-{compressed}-registers",
            lambda: (
                primitive(rotate_activation_a8, self.capacity, k, dynamic_rows=True, registers=True)
                if compressed
                else primitive(quantize, self.capacity, k)
            ),
            dict(X=source, Q="A8", S="AS", **(dict(Signs=self.signs[k]) if compressed else {})),
            vision=vision,
            divisor=divisor,
        )

    def attention(self, name, hidden, heads, kv_heads, bindings, *, causal=False, vision=False):
        from kernels.vision.encoder import attention

        options = dict(kv_heads=kv_heads, causal=causal, segmented=not vision)
        if not vision:
            options.update(bn=64, head_group=heads // kv_heads, stages=1, base2=True, full_row=True)
            bindings = dict(bindings, live_k=bindings["K"], live_v=bindings["V"])
        self.emit(
            name,
            f"attention-{hidden}-{heads}-{kv_heads}-{causal}-{vision}-grouped",
            lambda: primitive(attention, hidden, heads, **options),
            bindings,
            vision=vision,
        )

    def projection(
        self, name, weight, n, k, dest, *, vision=False, divisor=1, stride=None, offset=0
    ):
        compressed = (
            weight + ".packed" in self.w.index["weight_map"] or weight + ".packed" in self.w.buffers
        )
        suffix, kind = (".packed", "e8p_q4") if compressed else (".int8", "w8")
        data, scale = self.w.immutable(weight + suffix), self.w.immutable(weight + ".scale")
        capacity = (self.patches if vision else self.capacity) // divisor
        key = (kind, n, k, stride, capacity)

        if key not in self.tuned:
            import torch
            from tools.build.gpu import gpu_lock
            from tools.model.projection_tuning import ProjectionTuner

            with gpu_lock(self.lock):
                actual = dict(
                    W=torch.from_numpy(self.w.array(weight + suffix).copy()).cuda(),
                    WS=torch.from_numpy(self.w.array(weight + ".scale").copy()).cuda(),
                )
                if compressed:
                    actual["Book"] = torch.from_numpy(self.w.extra[self.book].copy()).cuda()
                tuner = ProjectionTuner(self.args.output / "tuning", self.args.cache / "tuning")
                identity = self.w.buffers[data]["data"]["sha256"]
                options = dict(
                    dynamic=True,
                    weights=actual,
                    weight_identity=identity,
                    measure_rows=sorted(
                        {min(capacity, rows) for rows in (32, 128, 512, 2048, 8192, capacity)}
                    ),
                    objective="total_latency",
                    output_stride=stride,
                    packed_k=128 if compressed else 0,
                    candidate_configs=(
                        [
                            dict(
                                algorithm="tensor_core",
                                block_m=m,
                                block_n=n,
                                block_k=128,
                                stages=stages,
                            )
                            for m, n, stages in [
                                (32, 64, 2),
                                (64, 64, 2),
                                (128, 64, 1),
                                (128, 128, 1),
                                (128, 128, 2),
                                (256, 64, 1),
                            ]
                        ]
                        + [
                            dict(
                                algorithm="tensor_core",
                                block_m=256,
                                block_n=64,
                                block_k=128,
                                stages=1,
                                group_m=1,
                            )
                        ]
                        if compressed
                        else [
                            dict(block_m=m, block_n=n, block_k=128)
                            for m, n in [(32, 64), (64, 64), (128, 64), (128, 128)]
                        ]
                        + [dict(block_m=128, block_n=128, block_k=128, group_m=4)]
                        + [
                            dict(block_m=m, block_n=n, block_k=128, group_m=1, cache_scales=True)
                            for m, n in [(32, 64), (64, 64), (128, 128)]
                        ]
                    ),
                )
                kernel, config = tuner.compile(kind, capacity, n, k, "float16", **options)
                choices = []
                if not vision:
                    choices = [
                        (suffix, choice, compiled)
                        for suffix, choice, compiled in tuner.compile_choices(
                            kind, capacity, n, k, "float16", **options
                        )
                        if choice != config
                    ]
                self.tuned[key] = (kernel, choices)
                write_json(
                    self.args.output / f"projection-{kind}-{n}-{k}-{stride}-{capacity}.json",
                    dict(
                        default=config,
                        selections=[
                            dict(range=suffix, config=choice) for suffix, choice, _ in choices
                        ],
                    ),
                )

        target = dict(kind="buffer_slice", name=dest, offset=offset * 2) if offset else dest
        bindings = (
            dict(A="A8", W=data, S=scale, AS="AS", Out=target, Book=self.book)
            if compressed
            else dict(A="A8", Weight=data, WeightScale=scale, TokenScale="AS", Output=target)
        )
        kernel, choices = self.tuned[key]
        self.emit(
            name, f"projection-{key}", lambda: kernel, bindings, vision=vision, divisor=divisor
        )
        if not vision:
            for suffix, config, compiled in choices:
                import json

                self.emit(
                    name + suffix,
                    f"projection-{key}-" + json.dumps(config, sort_keys=True),
                    lambda compiled=compiled: compiled,
                    bindings,
                )
