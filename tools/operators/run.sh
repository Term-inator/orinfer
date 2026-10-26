#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
resume=false
if [[ "${1:-}" == --resume ]]; then resume=true; shift; fi
if [[ $# -lt 2 ]]; then
    echo "usage: run.sh [--resume] runner.py output-directory [runner args]" >&2
    exit 2
fi
runner="$1"
output_arg="$2"
shift 2
cd "$repo_dir"
if [[ "$output_arg" = /* ]]; then output_dir="$output_arg"; else output_dir="$repo_dir/$output_arg"; fi
if [[ -e "$output_dir" && "$resume" != true ]]; then echo "Refusing existing output: $output_dir (use --resume to reuse an offline build)" >&2; exit 2; fi
mkdir -p "$output_dir"
mkdir -p "$output_dir/measurement-source"
cp --parents "$runner" tools/operators/common.py tools/operators/run.sh "$output_dir/measurement-source/"
kernel_file="kernels/operators/$(basename -- "$runner")"
if [[ -f "$kernel_file" ]]; then cp --parents "$kernel_file" "$output_dir/measurement-source/"; fi
exec 9>"$repo_dir/artifacts/gpu-experiment.lock"
flock 9
"${PYTHON:-$repo_dir/.venv/bin/python}" tools/bench/sample_machine.py --output "$output_dir/machine-before.json"
printf '%s\n' "compile-and-test" > "$output_dir/phase.txt"
sampler_pid=""
container_name="orinfer-operator-$$"
extra_mounts=()
group_args=()
for group_id in $(id -G); do group_args+=(--group-add "$group_id"); done
reference_env=()
network=none
resource_limits=()
# Initial Flash construction reads bounded BF16 ranges from its pinned source.
# Its downloader/compiler and charged file cache share a bounded OOM domain.
if [[ "$runner" == tools/model/flash_next/build.py ]]; then
    for argument in "$@"; do
        if [[ "$argument" == --checkpoint ]]; then
            network=host
            resource_limits=(--memory 6g --memory-swap 6g)
            reference_env+=(-e CARGO_BUILD_JOBS=2)
            break
        fi
    done
fi
# Native binding finalization uses the same pinned host Rust toolchain and
# cached dependencies. Only compiler/registry assets are mounted, no credentials.
native_cargo_binary="$(rustup which cargo)"
native_toolchain="$(dirname -- "$(dirname -- "$native_cargo_binary")")"
native_registry="${CARGO_HOME:-$HOME/.cargo}/registry"
native_cargo_cache="$repo_dir/artifacts/compiler-cargo"
mkdir -p "$native_cargo_cache/registry"
native_cargo_config="${CARGO_HOME:-$HOME/.cargo}/config.toml"
if [[ -f "$native_cargo_config" ]]; then
    # Preserve registry replacement identities for offline lookup, excluding
    # unrelated HTTP, credential-provider and user configuration sections.
    awk '/^\[/{source_section=($0 ~ /^\[source\./)} source_section {print}' \
        "$native_cargo_config" > "$native_cargo_cache/config.toml"
fi
extra_mounts+=(-v "$native_toolchain:$native_toolchain:ro"
    -v "$native_registry:$native_cargo_cache/registry:ro")
reference_env+=(-e "CARGO_HOME=$native_cargo_cache"
    -e "RUSTC=$native_toolchain/bin/rustc"
    -e "RUSTDOC=$native_toolchain/bin/rustdoc"
    -e "PATH=$native_toolchain/bin:/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
declare -A mounted_inputs=()
for variable in ORINFER_CHECKPOINT_DIR ORINFER_REFERENCE_CHECKPOINT ORINFER_REFERENCE_SOURCE ORINFER_REFERENCE_ACTIVATIONS ORINFER_EXECUTION_CACHE ORINFER_ENGINE; do
    if [[ -n "${!variable:-}" ]]; then
        input_path="$(realpath -- "${!variable}")"
        if [[ -z "${mounted_inputs[$input_path]+present}" ]]; then
            extra_mounts+=(-v "$input_path:$input_path:ro")
            mounted_inputs["$input_path"]=1
        fi
        reference_env+=(-e "$variable=$input_path")
    fi
done
entrypoint="${ORINFER_COMPILER_PYTHON:-/opt/venv/bin/python}"
command_args=("$runner")
if [[ -n "${ORINFER_OPERATOR_SANITIZER:-}" ]]; then
    entrypoint="/usr/local/cuda/bin/compute-sanitizer"
    command_args=(--tool "$ORINFER_OPERATOR_SANITIZER" --error-exitcode 99)
    if [[ -n "${ORINFER_OPERATOR_SANITIZER_KERNEL_NAME:-}" ]]; then
        # Compute Sanitizer accepts an explicit kernel filter as one argument;
        # never interpret it as shell flags. Preserve the inspection scope.
        command_args+=(--kernel-name "$ORINFER_OPERATOR_SANITIZER_KERNEL_NAME")
    fi
    command_args+=(/opt/venv/bin/python "$runner")
    printf '%q ' "${command_args[@]}" > "$output_dir/sanitizer-command.txt"
    printf '\n' >> "$output_dir/sanitizer-command.txt"
fi
if [[ -n "${ORINFER_OPERATOR_NCU:-}" ]]; then
    if [[ -n "${ORINFER_OPERATOR_SANITIZER:-}" ]]; then
        echo 'Cannot combine Nsight Compute and Compute Sanitizer' >&2
        exit 2
    fi
    entrypoint="/usr/local/cuda/bin/ncu"
    # Keep clocks and caches untouched. Explicit candidate symbols prevent
    # accidentally profiling a same-run reference instead of the candidate.
    command_args=(--clock-control none --cache-control none --set detailed
        --kernel-name-base function --kernel-name "${ORINFER_OPERATOR_NCU_KERNEL_NAME:-regex:^kernel_kernel$}"
        --launch-count 1 --export "$output_dir/ncu" /opt/venv/bin/python "$runner")
    printf '%q ' "${command_args[@]}" > "$output_dir/ncu-command.txt"
    printf '\n' >> "$output_dir/ncu-command.txt"
fi
if [[ -n "${ORINFER_OPERATOR_NSYS:-}" ]]; then
    if [[ -n "${ORINFER_OPERATOR_NCU:-}${ORINFER_OPERATOR_SANITIZER:-}" ]]; then
        echo 'Cannot combine Nsight Systems with another GPU profiler' >&2
        exit 2
    fi
    entrypoint="$(realpath -- "$(command -v nsys)")"
    nsys_directory="$(dirname -- "$(dirname -- "$entrypoint")")"
    if [[ ! -x "$nsys_directory/host-linux-x64/QdstrmImporter" ]]; then
        echo 'x86_64 Nsight Systems installation must include QdstrmImporter' >&2
        exit 2
    fi
    extra_mounts+=(-v "$nsys_directory:$nsys_directory:ro")
    command_args=(profile --trace cuda,nvtx --sample none --cpuctxsw none
        --capture-range cudaProfilerApi --capture-range-end stop
        --cuda-graph-trace node --output "$output_dir/nsys" /opt/venv/bin/python "$runner")
    printf '%q ' "${command_args[@]}" > "$output_dir/nsys-command.txt"
    printf '\n' >> "$output_dir/nsys-command.txt"
fi
cleanup() {
    docker rm -f "$container_name" >/dev/null 2>&1 || true
    touch "$output_dir/sampler.stop"
    if [[ -n "$sampler_pid" ]]; then wait "$sampler_pid" || true; fi
    "${PYTHON:-$repo_dir/.venv/bin/python}" tools/bench/sample_machine.py --output "$output_dir/machine-after.json" || true
}
trap cleanup EXIT
rm -f "$output_dir/sampler.stop"
"${PYTHON:-$repo_dir/.venv/bin/python}" tools/bench/sample_continuous.py --output "$output_dir/machine.jsonl" \
    --phase "$output_dir/phase.txt" --stop "$output_dir/sampler.stop" &
sampler_pid=$!
docker run --rm --name "$container_name" --gpus all --network "$network" \
    "${resource_limits[@]}" \
    --user "$(id -u):$(id -g)" "${group_args[@]}" \
    --entrypoint "$entrypoint" --shm-size 2g -v "$repo_dir:$repo_dir" -v "$output_dir:$output_dir" \
    "${extra_mounts[@]}" "${reference_env[@]}" -w "$repo_dir" -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
    -e PYTHONPATH="$repo_dir:$repo_dir/tools/operators" \
    -e TORCH_EXTENSIONS_DIR="$output_dir/cache/torch" -e XDG_CACHE_HOME="$output_dir/cache" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e ORINFER_OPERATOR_OUTPUT="$output_dir" -e TILELANG_CACHE_DIR="${ORINFER_TILELANG_CACHE_DIR:-$output_dir/cache}" \
    "${ORINFER_OPERATOR_IMAGE:-orinfer-compiler:0.2.0-sm120}" "${command_args[@]}" \
    --output "$output_dir" "$@" 2>&1 | tee "$output_dir/run.log"
if [[ -n "${ORINFER_OPERATOR_NSYS:-}" && ! -f "$output_dir/nsys.nsys-rep" ]]; then
    echo 'Nsight Systems did not export a report; any raw stream remains in the output directory' >&2
    exit 1
fi
