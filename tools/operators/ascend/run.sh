#!/usr/bin/env bash
# Offline compiler/validation runner. Production execution uses the Rust loader.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
image=${ORINFER_ASCEND_IMAGE:-orinfer-a2-tilelang:0.1.4-d182ae8-cann9.1-arm64}
driver_root=${ORINFER_ASCEND_DRIVER_ROOT:-/usr/local/Ascend/driver}
model_root=${ORINFER_ASCEND_MODEL_ROOT:-/data/models}
wheel_name=tilelang-0.1.4+ubuntu.24.3.cann910-cp312-cp312-linux_aarch64.whl
wheel_dir="$repo_dir/artifacts/ascend-runtime/wheels"
wheel_url='https://github.com/tile-ai/tilelang-ascend/releases/download/TileLang-ascend-v0.1.2.000-release/tilelang-0.1.4%2Bubuntu.24.3.cann910-cp312-cp312-linux_aarch64.whl'

if [[ ${1:-} == build ]]; then
    base_image=${ORINFER_ASCEND_BASE_IMAGE:?Set ORINFER_ASCEND_BASE_IMAGE to a CANN 9.1 / Python 3.12 runtime image}
    base_id=$(docker image inspect --format '{{.Id}}' "$base_image")
    base_tag="orinfer-ascend-base:build-$$"
    docker tag "$base_id" "$base_tag"
    trap 'docker image rm "$base_tag" >/dev/null 2>&1 || true' EXIT
    mkdir -p "$wheel_dir"
    if [[ ! -f "$wheel_dir/$wheel_name" ]]; then
        curl --fail --location --retry 2 "$wheel_url" -o "$wheel_dir/$wheel_name"
    fi
    docker build --pull=false --build-arg "BASE_IMAGE=$base_tag" -t "$image" \
        -f "$repo_dir/tools/operators/ascend/Dockerfile" "$wheel_dir"
    exit
fi

if [[ $# == 0 ]]; then
    echo 'Usage: run.sh build | run.sh python3 -m tools.operators.ascend.validate --output artifacts/ascend-runtime/aot' >&2
    exit 2
fi
without_devices=false
if [[ ${1:-} == --without-devices ]]; then
    without_devices=true
    shift
    if [[ $# == 0 ]]; then
        echo 'Pass the compiler command after --without-devices.' >&2
        exit 2
    fi
fi
docker image inspect "$image" >/dev/null
devices=()
lock=()
if [[ $without_devices == false ]]; then
    for device in /dev/davinci{0..7} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
        devices+=(--device "$device")
    done
    lock=(flock -w 5 "$repo_dir/artifacts/gpu-experiment.lock")
fi
mounts=()
if [[ -d $model_root ]]; then
    mounts+=(--mount "type=bind,src=$model_root,dst=$model_root,readonly")
fi
mkdir -p "$repo_dir/artifacts"
"${lock[@]}" docker run --rm --init --runtime runc --shm-size=8g \
    "${devices[@]}" \
    "${mounts[@]}" \
    --mount "type=bind,src=$driver_root,dst=/usr/local/Ascend/driver,readonly" \
    --mount "type=bind,src=$repo_dir,dst=/workspace/orinfer" \
    --workdir /workspace/orinfer --entrypoint "$1" "$image" "${@:2}"
