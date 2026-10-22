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
mkdir -p artifacts/quantization/compiler-cache
if [[ "$module" == flash_next ]]; then
    exec 9>artifacts/gpu-experiment.lock
    flock 9
fi
# qwen35 acquires the same GPU lock per bounded chunk, including synchronization
# and allocator release, so CPU downloads can overlap unrelated GPU validation.
exec docker run --rm --name "orinfer-quantizer-$module-$$" --runtime nvidia --network host \
    --user "$(id -u):$(id -g)" --entrypoint /opt/venv/bin/python \
    -v "$repo_dir:$repo_dir" -w "$repo_dir" \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
    -e HTTPS_PROXY -e HTTP_PROXY -e https_proxy -e http_proxy -e NO_PROXY -e no_proxy \
    -e PYTHONPATH="$repo_dir" \
    -e TILELANG_CACHE_DIR="$repo_dir/artifacts/quantization/compiler-cache" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    "${ORINFER_OPERATOR_IMAGE:-orinfer-compiler:0.1.3}" -m "tools.quantization.$module" "$@"
