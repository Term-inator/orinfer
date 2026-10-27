use super::{
    DriverLease, open_library,
    runtime::{Limits, Session},
};
use libloading::Library;
use std::{path::Path, sync::Arc};

struct Fixture {
    _dir: tempfile::TempDir,
    library: Library,
    driver: Option<Arc<DriverLease>>,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("libacl-fixture.so");
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ascend_runtime.c");
        assert!(
            std::process::Command::new("cc")
                .args(["-shared", "-fPIC", "-Wall", "-Werror"])
                .arg(source)
                .arg("-o")
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        // SAFETY: The test compiles this exact fixture and retains its code.
        let library = unsafe { Library::new(&path) }.unwrap();
        let driver = Arc::new(DriverLease {
            driver: Some(Arc::new(open_library(&path).unwrap())),
            registered: false,
        });
        Self {
            _dir: dir,
            library,
            driver: Some(driver),
        }
    }
    fn session(&self) -> Session {
        Session::with_driver(
            Arc::clone(self.driver.as_ref().unwrap()),
            0,
            Limits::default(),
        )
        .unwrap()
    }
    fn aot(&self) -> (std::path::PathBuf, super::kernels::Spec) {
        use super::kernels::{Access, Operand, Spec};
        use crate::artifact::{Dtype, FileIdentity};
        let library = self._dir.path().join("libacl-fixture.so");
        let spec = Spec {
            schema_version: 1,
            target: orinfer_device::Target {
                host_arch: std::env::consts::ARCH.into(),
                backend: orinfer_device::Backend::Ascend,
                device_arch: "ascend910b".into(),
            },
            op: "copy_fixture".into(),
            operands: vec![
                Operand {
                    name: "input".into(),
                    dtype: Dtype::I8,
                    shape: vec![4],
                    access: Access::Read,
                },
                Operand {
                    name: "output".into(),
                    dtype: Dtype::I8,
                    shape: vec![4],
                    access: Access::Write,
                },
            ],
            parameters: Default::default(),
            compiler: [("fixture".into(), "cc".into())].into(),
            source_sha256: "1".repeat(64),
            library: FileIdentity {
                file: "libacl-fixture.so".into(),
                sha256: crate::artifact::sha256(&std::fs::read(library).unwrap()),
                tensor: None,
            },
        };
        let text =
            std::ffi::CString::new(serde_json::to_string(&spec.identity()).unwrap()).unwrap();
        // SAFETY: The CPU fixture copies this string into its bounded static
        // descriptor; no borrowed host pointer survives this setter.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn(*const std::ffi::c_char)>(b"test_kernel_contract\0")
                .unwrap()(text.as_ptr());
        }
        let path = self._dir.path().join("kernel.json");
        std::fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
        (path, spec)
    }
    fn counter(&self, index: i32) -> i32 {
        // SAFETY: Fixture declares this function with the exact C signature.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn(i32) -> i32>(b"test_counter\0")
                .unwrap()(index)
        }
    }
    fn failure(&self, index: i32) {
        // SAFETY: Fixture declares this function with the exact C signature.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn(i32)>(b"test_failure\0")
                .unwrap()(index);
        }
    }
    fn assert_clean(&mut self) {
        self.driver.take();
        for index in 0..5 {
            assert_eq!(self.counter(index), 0, "resource counter {index}");
        }
        assert_eq!(self.counter(5), 1);
        assert_eq!(self.counter(6), 1);
        assert_eq!(self.counter(7), 0);
        assert_eq!(self.counter(8), 0);
    }
}

#[test]
fn aot_binding_rejects_foreign_buffers_aliases_and_tampered_contract() {
    use super::kernels::Kernel;
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let other = fixture.session();
    let input = session.allocate(4).unwrap();
    let output = session.allocate(4).unwrap();
    let foreign = other.allocate(4).unwrap();
    let short = session.allocate(2).unwrap();
    let (path, mut spec) = fixture.aot();
    assert!(Kernel::load(&session, &path, &[input.clone(), foreign.clone()]).is_err());
    assert!(Kernel::load(&session, &path, &[input.clone(), input.clone()]).is_err());
    assert!(Kernel::load(&session, &path, &[input.clone(), short.clone()]).is_err());
    assert!(Kernel::load(&session, &path, std::slice::from_ref(&input)).is_err());
    session
        .upload(&input, 0, &[1, 2, 3, 4])
        .unwrap()
        .wait()
        .unwrap();
    let kernel = Kernel::load(&session, &path, &[input.clone(), output.clone()]).unwrap();
    assert!(kernel.enqueue(&other).is_err());
    kernel.enqueue(&session).unwrap().wait().unwrap();
    assert_eq!(session.download(&output, 0, 4).unwrap(), [1, 2, 3, 4]);
    spec.op = "tampered".into();
    std::fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
    assert!(Kernel::load(&session, &path, &[input.clone(), output.clone()]).is_err());
    drop((kernel, input, output, foreign, short, other, session));
    fixture.assert_clean();
}

#[test]
fn aot_graph_retains_operands_code_and_context_until_last_fence() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let (path, _) = fixture.aot();
    let input = session.allocate(4).unwrap();
    let output = session.allocate(4).unwrap();
    session
        .upload(&input, 0, &[1, 2, 3, 4])
        .unwrap()
        .wait()
        .unwrap();
    let kernel =
        super::kernels::Kernel::load(&session, &path, &[input.clone(), output.clone()]).unwrap();
    let graph = kernel.capture(&session).unwrap();
    drop((kernel, input, output));
    assert_eq!(session.allocated_bytes(), 8);
    let ticket = graph.enqueue(&session).unwrap();
    drop(graph);
    assert_eq!(session.allocated_bytes(), 8);
    ticket.wait().unwrap();
    assert_eq!(session.allocated_bytes(), 0);
    drop((ticket, session));
    fixture.assert_clean();
}

#[test]
fn aot_program_orders_shared_operands_and_retains_the_whole_fragment() {
    use super::kernels::{Kernel, Program};
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let other = fixture.session();
    let (path, _) = fixture.aot();
    let input = session.allocate(4).unwrap();
    let middle = session.allocate(4).unwrap();
    let output = session.allocate(4).unwrap();
    session
        .upload(&input, 0, &[9, 8, 7, 6])
        .unwrap()
        .wait()
        .unwrap();
    let first = Kernel::load(&session, &path, &[input.clone(), middle.clone()]).unwrap();
    let second = Kernel::load(&session, &path, &[middle.clone(), output.clone()]).unwrap();
    assert!(Program::new(&session, &[]).is_err());
    assert!(Program::new(&other, &[&first]).is_err());
    let program = Program::new(&session, &[&first, &second]).unwrap();
    assert!(program.enqueue(&other).is_err());
    program.enqueue(&session).unwrap().wait().unwrap();
    assert_eq!(session.download(&output, 0, 4).unwrap(), [9, 8, 7, 6]);
    let graph = program.capture(&session).unwrap();
    drop((program, first, second, input, middle, output));
    assert_eq!(session.allocated_bytes(), 12);
    let ticket = graph.enqueue(&session).unwrap();
    drop(graph);
    ticket.wait().unwrap();
    assert_eq!(session.allocated_bytes(), 0);
    drop((ticket, other, session));
    fixture.assert_clean();
}

#[test]
fn async_upload_retains_copied_host_data_and_dropped_device_handles() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(4).unwrap();
    let mut source = vec![1, 2, 3, 4];
    let ticket = session.upload(&buffer, 0, &source).unwrap();
    source.fill(0);
    assert!(!ticket.is_ready().unwrap());
    assert_eq!(session.pinned_bytes(), 4);
    drop(buffer);
    assert_eq!(fixture.counter(0), 1);
    ticket.wait().unwrap();
    assert_eq!(session.pinned_bytes(), 0);
    assert_eq!(fixture.counter(0), 0);
    drop(ticket);
    drop(session);
    fixture.assert_clean();
}

#[test]
fn stream_dependency_roundtrip_and_session_ownership() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let other = fixture.session();
    let buffer = session.allocate(8).unwrap();
    let mut source = vec![3, 4, 5, 6];
    let ticket = session.upload(&buffer, 2, &source).unwrap();
    source.fill(0);
    session.wait_on_compute(&ticket).unwrap();
    assert!(other.upload(&buffer, 0, &[1]).is_err());
    assert!(other.wait_on_compute(&ticket).is_err());
    assert!(session.upload(&buffer, usize::MAX, &[1]).is_err());
    assert!(session.upload(&buffer, 7, &[1, 2]).is_err());
    let downloaded = session.download(&buffer, 2, 4).unwrap();
    assert_eq!(downloaded, [3, 4, 5, 6]);
    session.fill(&buffer, 9).unwrap().wait().unwrap();
    assert_eq!(session.download(&buffer, 0, 8).unwrap(), [9; 8]);
    drop((ticket, buffer, session, other));
    fixture.assert_clean();
}

#[test]
fn buffer_can_outlive_session_without_finalizing_its_context() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(8).unwrap();
    session.upload(&buffer, 0, &[7; 8]).unwrap();
    drop(session);
    assert_eq!(fixture.counter(1), 0);
    assert_eq!(fixture.counter(4), 1);
    drop(buffer);
    fixture.assert_clean();
}

#[test]
fn failed_record_fences_queued_dma_and_failed_creation_rolls_back() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(4).unwrap();
    fixture.failure(4);
    assert!(session.upload(&buffer, 0, &[1, 2, 3, 4]).is_err());
    assert_eq!(fixture.counter(1), 0);
    assert_eq!(fixture.counter(2), 0);
    fixture.failure(0);
    assert_eq!(session.download(&buffer, 0, 4).unwrap(), [1, 2, 3, 4]);
    drop((buffer, session));
    fixture.failure(5);
    assert!(
        Session::with_driver(
            Arc::clone(fixture.driver.as_ref().unwrap()),
            0,
            Limits::default()
        )
        .is_err()
    );
    fixture.assert_clean();
}

#[test]
fn tensor_views_retain_backing_and_reject_overlapping_device_copies() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let storage = session.allocate(32).unwrap();
    let left = storage.slice(4, 8).unwrap();
    let overlapping = storage.slice(8, 8).unwrap();
    let right = storage.slice(20, 8).unwrap();
    assert!(storage.slice(30, 4).is_err());
    assert!(storage.slice(0, 0).is_err());
    assert!(session.copy(&left, 0, &overlapping, 0, 8).is_err());
    session
        .upload(&left, 0, &[1, 2, 3, 4, 5, 6, 7, 8])
        .unwrap()
        .wait()
        .unwrap();
    session
        .copy(&left, 0, &right, 0, 8)
        .unwrap()
        .wait()
        .unwrap();
    drop(storage);
    assert_eq!(session.allocated_bytes(), 32);
    assert_eq!(
        session.download(&right, 0, 8).unwrap(),
        [1, 2, 3, 4, 5, 6, 7, 8]
    );
    drop((left, overlapping, right));
    assert_eq!(session.allocated_bytes(), 0);
    drop(session);
    fixture.assert_clean();
}

#[test]
fn budgets_are_enforced_before_native_allocation() {
    let mut fixture = Fixture::new();
    let session = Session::with_driver(
        Arc::clone(fixture.driver.as_ref().unwrap()),
        0,
        Limits {
            device_bytes: 8,
            pinned_bytes: 4,
            timeout_ms: 100,
        },
    )
    .unwrap();
    let buffer = session.allocate(8).unwrap();
    assert!(session.allocate(1).is_err());
    assert!(session.allocate(0).is_err());
    assert!(session.upload(&buffer, 0, &[1; 5]).is_err());
    let ticket = session.upload(&buffer, 0, &[1; 4]).unwrap();
    assert!(session.upload(&buffer, 4, &[2; 4]).is_err());
    ticket.wait().unwrap();
    session.upload(&buffer, 4, &[2; 4]).unwrap().wait().unwrap();
    drop((buffer, ticket, session));
    fixture.assert_clean();
}

#[test]
fn device_copy_preserves_extents_and_retains_waited_events() {
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let source = session.allocate(8).unwrap();
    let destination = session.allocate(8).unwrap();
    let uploaded = session
        .upload(&source, 0, &[1, 2, 3, 4, 5, 6, 7, 8])
        .unwrap();
    uploaded.wait().unwrap();
    let waited = session.wait_on_compute(&uploaded).unwrap();
    drop(uploaded);
    session.reap().unwrap();
    assert_eq!(
        fixture.counter(2),
        2,
        "queued compute wait retains its source event"
    );
    let copied = session.copy(&source, 2, &destination, 4, 4).unwrap();
    assert!(session.copy(&source, 1, &source, 2, 4).is_err());
    assert_eq!(session.download(&destination, 4, 4).unwrap(), [3, 4, 5, 6]);
    drop((waited, copied, source, destination, session));
    fixture.assert_clean();
}

#[test]
fn failed_completion_retires_and_quarantines_dma_backing() {
    let fixture = Fixture::new();
    let session = fixture.session();
    let buffer = session.allocate(8).unwrap();
    let ticket = session.upload(&buffer, 0, &[1; 8]).unwrap();
    fixture.failure(3);
    assert!(ticket.wait().is_err());
    assert!(session.allocate(1).is_err());
    drop((ticket, buffer, session));
    assert_eq!(fixture.counter(0), 1);
    assert_eq!(fixture.counter(1), 1);
    assert_eq!(fixture.counter(4), 1);
    assert_eq!(
        fixture.counter(6),
        0,
        "unfenced context must not finalize ACL"
    );
}

#[test]
fn graph_retains_its_resources_through_replay_and_rolls_back_capture_errors() {
    struct Guard(std::rc::Rc<std::cell::Cell<usize>>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let dropped = std::rc::Rc::new(std::cell::Cell::new(0));
    // SAFETY: Empty capture references only the retained Guard, with no raw data.
    let graph = unsafe {
        super::graph::Graph::capture(&session, Box::new(Guard(dropped.clone())), |_| Ok(()))
    }
    .unwrap();
    let ticket = graph.enqueue(&session).unwrap();
    drop(graph);
    assert_eq!(dropped.get(), 0);
    assert_eq!(fixture.counter(7), 1);
    ticket.wait().unwrap();
    assert_eq!(dropped.get(), 1);
    assert_eq!(fixture.counter(7), 0);
    // SAFETY: No pointer is submitted; failure exercises the capture rollback.
    let result = unsafe {
        super::graph::Graph::capture(&session, Box::new(()), |_| {
            Err("injected capture failure".into())
        })
    };
    assert!(result.is_err());
    assert_eq!(fixture.counter(7), 0);
    drop((ticket, session));
    fixture.assert_clean();
}

#[test]
fn collective_completion_retains_communicator_and_validates_byte_units() {
    use super::hccl::{Communicator, Counts, Element, RootInfo};
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let other = Session::with_driver(
        Arc::clone(fixture.driver.as_ref().unwrap()),
        1,
        Limits::default(),
    )
    .unwrap();
    let root = {
        let _current = other.context.enter().unwrap();
        let root =
            RootInfo::fixture(&session, &fixture._dir.path().join("libacl-fixture.so")).unwrap();
        assert_eq!(
            root.as_bytes()[0],
            0,
            "bootstrap selects its own session context"
        );
        root
    };
    drop(other);
    let comm = Communicator::join(&session, &root, 0).unwrap();
    let a = session.allocate(16).unwrap();
    let b = session.allocate(16).unwrap();
    session.upload(&a, 0, &[7; 16]).unwrap().wait().unwrap();
    assert!(
        comm.all_reduce_sum(&session, &a, &b, 5, Element::Float32)
            .is_err()
    );
    let ticket = comm
        .all_reduce_sum(&session, &a, &b, 4, Element::Float32)
        .unwrap();
    drop(comm);
    assert_eq!(fixture.counter(8), 1);
    ticket.wait().unwrap();
    assert_eq!(fixture.counter(8), 0);
    assert_eq!(session.download(&b, 0, 16).unwrap(), [7; 16]);
    let comm = Communicator::join(&session, &root, 0).unwrap();
    comm.all_to_all_v(
        &session,
        &a,
        &b,
        Counts::new(vec![4]).unwrap(),
        Counts::new(vec![4]).unwrap(),
        Element::Float32,
    )
    .unwrap()
    .wait()
    .unwrap();
    assert_eq!(session.download(&b, 0, 16).unwrap(), [7; 16]);
    drop((comm, ticket, a, b, root, session));
    fixture.assert_clean();
}

#[test]
fn failed_collective_creation_never_destroys_an_unowned_output_handle() {
    use super::hccl::{Communicator, RootInfo};
    let mut fixture = Fixture::new();
    let session = fixture.session();
    let root = RootInfo::fixture(&session, &fixture._dir.path().join("libacl-fixture.so")).unwrap();
    fixture.failure(10);
    assert!(Communicator::join(&session, &root, 0).is_err());
    assert!(session.allocate(1).is_err());
    drop((root, session));
    fixture.assert_clean();
}
