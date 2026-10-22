#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
toolchain="$(dirname -- "$(dirname -- "$(rustup which cargo)")")"
registry="${CARGO_HOME:-$HOME/.cargo}/registry"
cache="$repo_dir/artifacts/compiler-cargo"
build_dir="$repo_dir/artifacts/native-build-sm120"
mkdir -p "$cache/registry" "$build_dir"
config="${CARGO_HOME:-$HOME/.cargo}/config.toml"
if [[ -f "$config" ]]; then
    awk '/^\[/{source_section=($0 ~ /^\[source\./)} source_section {print}' "$config" > "$cache/config.toml"
fi
docker run --rm --runtime runc --network none --user "$(id -u):$(id -g)" \
    -v "$repo_dir:$repo_dir" -w "$repo_dir" \
    -v "$toolchain:$toolchain:ro" -v "$registry:$cache/registry:ro" \
    -e "CARGO_HOME=$cache" -e "CARGO_TARGET_DIR=$build_dir" \
    -e "RUSTC=$toolchain/bin/rustc" -e "RUSTDOC=$toolchain/bin/rustdoc" \
    -e "PATH=$toolchain/bin:/opt/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    --entrypoint "$toolchain/bin/cargo" orinfer-compiler:0.2.0-sm120 \
    build --release --locked --offline -p orinfer-cli -p orinfer-models
mkdir -p target/release
for binary in orinfer liborinfer_models.so; do
    install -m 755 "$build_dir/release/$binary" "target/release/.$binary.new"
    mv -f "target/release/.$binary.new" "target/release/$binary"
done
