#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
mkdir -p artifacts
mkdir -p artifacts/quantization/flash-next/teacher-cache
exec 9>artifacts/gpu-experiment.lock
flock 9
exec docker run --rm --name "orinfer-flash-teacher-$$" --gpus all --network host \
    --user "$(id -u):$(id -g)" --entrypoint /opt/venv/bin/python \
    -v "$repo_dir:$repo_dir" -w "$repo_dir" \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
    -e HTTPS_PROXY -e HTTP_PROXY -e https_proxy -e http_proxy -e NO_PROXY -e no_proxy \
    -e PYTHONPATH="$repo_dir" \
    -e XDG_CACHE_HOME="$repo_dir/artifacts/quantization/flash-next/teacher-cache" \
    -e TILELANG_CACHE_DIR="$repo_dir/artifacts/quantization/flash-next/teacher-cache/tilelang" \
    -e PYTHONDONTWRITEBYTECODE=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    orinfer-compiler:0.2.0-sm120 -m tools.model.flash_next.reference.teacher "$@"
