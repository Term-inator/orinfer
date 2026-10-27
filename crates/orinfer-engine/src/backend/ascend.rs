//! Read-only CANN discovery. Model programs must register a separate executor.
use super::Inventory;
use crate::artifact::Result;
use libloading::Library;
use std::{
    ffi::{CStr, c_char},
    path::Path,
    ptr,
};

type Init = unsafe extern "C" fn(*const c_char) -> i32;
type Finalize = unsafe extern "C" fn() -> i32;
type Count = unsafe extern "C" fn(*mut u32) -> i32;
type Soc = unsafe extern "C" fn() -> *const c_char;

struct Runtime {
    _library: Library,
    finalize: Finalize,
    initialized: bool,
}
impl Drop for Runtime {
    fn drop(&mut self) {
        if self.initialized {
            // SAFETY: This owner successfully initialized ACL, owns no device
            // resources, and keeps the library loaded through finalization.
            unsafe {
                (self.finalize)();
            }
        }
    }
}
fn check(status: i32, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("{operation}: ACL error {status}"))
    }
}

fn inspect_library(path: &Path) -> Result<Inventory> {
    // SAFETY: CANN is an explicitly selected native dependency. The Library
    // owner stays live for every function pointer and ACL lifetime below.
    let library = unsafe { Library::new(path) }.map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: These four symbols and signatures are declared by CANN's public
    // acl_rt.h and acl_base_rt.h. No device allocation or kernel is launched.
    let (init, finalize, count, soc) = unsafe {
        (
            *library
                .get::<Init>(b"aclInit\0")
                .map_err(|e| e.to_string())?,
            *library
                .get::<Finalize>(b"aclFinalize\0")
                .map_err(|e| e.to_string())?,
            *library
                .get::<Count>(b"aclrtGetDeviceCount\0")
                .map_err(|e| e.to_string())?,
            *library
                .get::<Soc>(b"aclrtGetSocName\0")
                .map_err(|e| e.to_string())?,
        )
    };
    let mut runtime = Runtime {
        _library: library,
        finalize,
        initialized: false,
    };
    // SAFETY: Null selects default ACL configuration. Only a successful init
    // transfers finalization ownership; an existing caller's ACL is not closed.
    check(unsafe { init(ptr::null()) }, "aclInit")?;
    runtime.initialized = true;
    let mut devices = 0;
    // SAFETY: ACL initialized successfully and devices is a live writable u32.
    check(unsafe { count(&mut devices) }, "aclrtGetDeviceCount")?;
    let soc_name = if devices == 0 {
        None
    } else {
        // SAFETY: ACL owns the returned static NUL-terminated SoC name while
        // the initialized runtime is retained; copy it before finalization.
        let name = unsafe { soc() };
        if name.is_null() {
            return Err("aclrtGetSocName returned null".into());
        }
        // SAFETY: Non-null CANN string with the lifetime described above.
        Some(
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned(),
        )
    };
    let mut api_symbols = Vec::new();
    for name in [
        "aclrtReserveMemAddress",
        "aclrtMapMem",
        "aclmdlRICaptureBegin",
        "aclrtLaunchKernel",
    ] {
        let symbol = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
        // SAFETY: Only resolve an opaque symbol address; it is never called or
        // interpreted as proof that this chip supports the operation.
        if unsafe {
            runtime
                ._library
                .get::<*const std::ffi::c_void>(symbol.as_bytes_with_nul())
        }
        .is_ok()
        {
            api_symbols.push(name.into());
        }
    }
    Ok(Inventory {
        backend: orinfer_device::Backend::Ascend,
        host_arch: std::env::consts::ARCH.into(),
        device_count: devices,
        soc: soc_name,
        runtime_library: path.display().to_string(),
        api_symbols,
        model_execution_available: false,
    })
}

pub(super) fn inspect() -> Result<Inventory> {
    if let Some(path) = std::env::var_os("ORINFER_ASCEND_LIBRARY") {
        return inspect_library(Path::new(&path));
    }
    let mut errors = Vec::new();
    for path in [
        "libascendcl.so",
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
    ] {
        match inspect_library(Path::new(path)) {
            Ok(value) => return Ok(value),
            Err(error) => errors.push(error),
        }
    }
    Err(format!(
        "CANN discovery failed; initialize the CANN runtime library environment: {}",
        errors.join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(init_status: i32, count_status: i32) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("acl.c");
        let library = dir.path().join("libacl-test.so");
        std::fs::write(
            &source,
            format!(
                r#"
            #include <stdint.h>
            static int initialized=0,finalized=0;
            int aclInit(const char *p) {{ (void)p;initialized++;return {init_status}; }}
            int aclFinalize(void) {{ finalized++;return 0; }}
            int aclrtGetDeviceCount(uint32_t *n) {{ *n=8;return {count_status}; }}
            const char *aclrtGetSocName(void) {{ return "Ascend910B3"; }}
            int test_finalized(void) {{ return finalized; }}
            int test_initialized(void) {{ return initialized; }}
        "#
            ),
        )
        .unwrap();
        assert!(
            std::process::Command::new("cc")
                .args(["-shared", "-fPIC"])
                .arg(&source)
                .arg("-o")
                .arg(&library)
                .status()
                .unwrap()
                .success()
        );
        (dir, library)
    }
    #[test]
    fn successful_and_failed_queries_finalize_owned_acl() {
        for (init_status, count_status) in [(0, 0), (0, 7), (7, 0)] {
            let (_dir, path) = fixture(init_status, count_status);
            // SAFETY: The test builds this exact library and retains a second
            // reference so its counters remain accessible after probe cleanup.
            let library = unsafe { Library::new(&path) }.unwrap();
            let result = inspect_library(&path);
            if init_status == 0 && count_status == 0 {
                let info = result.unwrap();
                assert_eq!(info.device_count, 8);
                assert_eq!(info.soc.as_deref(), Some("Ascend910B3"));
                assert!(!info.model_execution_available);
            } else {
                assert!(result.unwrap_err().contains("ACL error 7"));
            }
            // SAFETY: The fixture declares both functions with this signature.
            unsafe {
                assert_eq!(
                    library
                        .get::<unsafe extern "C" fn() -> i32>(b"test_finalized\0")
                        .unwrap()(),
                    i32::from(init_status == 0)
                );
                assert_eq!(
                    library
                        .get::<unsafe extern "C" fn() -> i32>(b"test_initialized\0")
                        .unwrap()(),
                    1
                );
            }
        }
    }
}
