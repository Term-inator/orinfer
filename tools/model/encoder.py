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
            f"a8-{k}-{compressed}",
            lambda: (
                primitive(rotate_activation_a8, self.capacity, k, dynamic_rows=True)
                if compressed
                else primitive(quantize, self.capacity, k)
            ),
            dict(X=source, Q="A8", S="AS", **(dict(Signs=self.signs[k]) if compressed else {})),
            vision=vision,
            divisor=divisor,
        )

    def projection(
        self, name, weight, n, k, dest, *, vision=False, divisor=1, stride=None, offset=0
    ):
        compressed = (
            weight + ".packed" in self.w.index["weight_map"] or weight + ".packed" in self.w.buffers
        )
        suffix, kind = (".packed", "e8p_q4") if compressed else (".int8", "w8")
        data, scale = self.w.immutable(weight + suffix), self.w.immutable(weight + ".scale")
        key = (kind, n, k, stride)

        def build():
            if key not in self.tuned:
                import torch
                from tools.model.projection_tuning import ProjectionTuner

                actual = dict(
                    W=torch.from_numpy(self.w.array(weight + suffix).copy()).cuda(),
                    WS=torch.from_numpy(self.w.array(weight + ".scale").copy()).cuda(),
                )
                if compressed:
                    actual["Book"] = torch.from_numpy(self.w.extra[self.book].copy()).cuda()
                tuner = ProjectionTuner(self.args.output / "tuning", self.args.cache / "tuning")
                identity = self.w.buffers[data]["data"]["sha256"]
                kernel, config = tuner.compile(
                    kind,
                    self.capacity,
                    n,
                    k,
                    "float16",
                    dynamic=True,
                    weights=actual,
                    weight_identity=identity,
                    measure_rows=sorted(
                        {min(self.capacity, rows) for rows in (32, 128, 512, 2048)}
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
                                (128, 128, 1),
                                (128, 128, 2),
                                (256, 64, 1),
                            ]
                        ]
                        if compressed
                        else [
                            dict(block_m=m, block_n=n, block_k=128)
                            for m, n in [(32, 64), (64, 64), (128, 64), (128, 128)]
                        ]
                    ),
                )
                self.tuned[key] = kernel
                write_json(self.args.output / f"projection-{kind}-{n}-{k}-{stride}.json", config)
            return self.tuned[key]

        target = dict(kind="buffer_slice", name=dest, offset=offset * 2) if offset else dest
        bindings = (
            dict(A="A8", W=data, S=scale, AS="AS", Out=target, Book=self.book)
            if compressed
            else dict(A="A8", Weight=data, WeightScale=scale, TokenScale="AS", Output=target)
        )
        self.emit(name, f"projection-{key}", build, bindings, vision=vision, divisor=divisor)
