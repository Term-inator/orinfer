#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
module="${1:?usage: run.sh flash_next|qwen35|qwen_e8p [converter arguments]}"
shift
case "$module" in
    flash_next|qwen35|qwen_e8p) ;;
    *) echo 'Unknown quantization entrypoint' >&2; exit 2 ;;
esac
cd "$repo_dir"
mkdir -p artifacts/quantization/compiler-cache/sm120
resource_limits=()
if [[ "$module" == flash_next ]]; then
    # Bound host allocations and charged file cache; keep desktop applications
    # outside the converter's OOM domain and avoid consuming host swap.
    resource_limits=(--memory 6g --memory-swap 6g)
    exec 9>artifacts/gpu-experiment.lock
    flock 9
fi
# qwen35 acquires the same GPU lock per bounded chunk, including synchronization
# and allocator release, so CPU downloads can overlap unrelated GPU validation.
source_mounts=()
if [[ -n "${ORINFER_CHECKPOINT_DIR:-}" ]]; then
    checkpoint="$(realpath -- "$ORINFER_CHECKPOINT_DIR")"
    source_mounts+=(-v "$checkpoint:$checkpoint:ro")
fi
exec docker run --rm --name "orinfer-quantizer-$module-$$" --gpus all --network host \
    "${resource_limits[@]}" \
    --user "$(id -u):$(id -g)" --entrypoint /opt/venv/bin/python \
    -v "$repo_dir:$repo_dir" -w "$repo_dir" \
    "${source_mounts[@]}" \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
    -e HTTPS_PROXY -e HTTP_PROXY -e https_proxy -e http_proxy -e NO_PROXY -e no_proxy \
    -e PYTHONPATH="$repo_dir" \
    -e TILELANG_CACHE_DIR="$repo_dir/artifacts/quantization/compiler-cache/sm120" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    orinfer-compiler:0.2.0-sm120 -m "tools.quantization.$module" "$@"
