use std::{env, fs, path::PathBuf};

fn main() {
    let path = "../../configs/deployment-target.json";
    println!("cargo:rerun-if-changed={path}");
    let target: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let source = format!(
        "pub const DEVICE: &str = {device:?};\npub const HOST_ARCH: &str = {host:?};\npub const CUDA_ARCH: &str = {cuda:?};\npub const COMPUTE_CAPABILITY: [i32; 2] = [{major}, {minor}];\npub const ELF_MACHINE: u16 = {elf};\n",
        device = target["device"].as_str().unwrap(),
        host = target["host_arch"].as_str().unwrap(),
        cuda = target["cuda_arch"].as_str().unwrap(),
        major = target["compute_capability"][0].as_u64().unwrap(),
        minor = target["compute_capability"][1].as_u64().unwrap(),
        elf = target["elf_machine"].as_u64().unwrap(),
    );
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("target.rs"),
        source,
    )
    .unwrap();
}
