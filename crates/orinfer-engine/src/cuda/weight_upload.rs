//! Bounded disk/DMA pipeline with session-owned reusable registered staging.
//! Reader threads never own or call CUDA handles.
use super::{Session, check, host_weights, sys};
use crate::{artifact::Result, weights::TensorSource};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::{File, OpenOptions},
    io::ErrorKind,
    os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    ptr,
    sync::{Arc, Mutex, mpsc},
    time::Instant,
};

const ALIGNMENT: usize = 4096;
const SLOT_BYTES: usize = 8 << 20;
pub(super) const PAYLOAD_BYTES: usize = SLOT_BYTES - ALIGNMENT;

pub(super) struct Upload {
    pub source: TensorSource,
    pub destination: u64,
}

#[derive(Default)]
pub(super) struct Statistics {
    pub read_wait_s: f64,
    pub dma_wait_s: f64,
    pub staging_s: f64,
    pub idle_work_s: f64,
    pub chunks: usize,
    pub reader_service_s: f64,
    pub file_read_s: f64,
    cache_copy_s: f64,
}

/// Ready reads have priority. While disk readers are busy, the CUDA owner may
/// finish bounded initialization work; no CUDA handle crosses into a reader.
fn receive<T>(
    completed: &mpsc::Receiver<T>,
    idle: &mut dyn FnMut() -> Result<bool>,
) -> Result<(T, f64, f64)> {
    let mut idle_s = 0.;
    loop {
        match completed.try_recv() {
            Ok(value) => return Ok((value, 0., idle_s)),
            Err(mpsc::TryRecvError::Disconnected) => return Err("Weight readers stopped".into()),
            Err(mpsc::TryRecvError::Empty) => {}
        }
        let at = Instant::now();
        let worked = idle()?;
        idle_s += at.elapsed().as_secs_f64();
        if !worked {
            let at = Instant::now();
            let value = completed.recv().map_err(|_| "Weight readers stopped")?;
            return Ok((value, at.elapsed().as_secs_f64(), idle_s));
        }
    }
}

struct ReadFile {
    buffered: File,
    direct: Option<File>,
    size: usize,
    identity: (u64, u64),
}
#[derive(Default)]
struct Reader {
    files: BTreeMap<std::path::PathBuf, ReadFile>,
    file_read_s: f64,
    cache_copy_s: f64,
}

struct ReadReport {
    result: Result<usize>,
    service_s: f64,
    file_read_s: f64,
    cache_copy_s: f64,
}

struct Slot {
    memory: Option<memmap2::MmapMut>,
    pointer: *mut std::ffi::c_void,
    event: sys::CUevent,
    registered: bool,
}
pub(super) struct Staging {
    stream: sys::CUstream,
    pending: bool,
    slots: Vec<Slot>,
    readers: Option<Readers>,
}
impl Drop for Staging {
    fn drop(&mut self) {
        // Stop dispatch and join CPU readers before releasing their host ranges.
        drop(self.readers.take());
        // Finish DMA before freeing memory, including after an upload failure.
        // SAFETY: The session and its explicit stream outlive this pool.
        unsafe {
            let code = sys::cuStreamSynchronize(self.stream);
            if code != sys::CUresult::CUDA_SUCCESS {
                // Never release a host range which may still be in use by DMA.
                eprintln!(
                    "CUDA weight staging cleanup: synchronize error {}",
                    code as i32
                );
                for slot in &mut self.slots {
                    if let Some(memory) = slot.memory.take() {
                        std::mem::forget(memory);
                    }
                }
                return;
            }
            for slot in &mut self.slots {
                if !slot.event.is_null()
                    && let Err(error) = check(
                        sys::cuEventDestroy_v2(slot.event),
                        "weight staging event cleanup",
                    )
                {
                    eprintln!("CUDA weight staging cleanup: {error}");
                }
                if slot.registered
                    && let Err(error) = check(
                        sys::cuMemHostUnregister(slot.pointer),
                        "weight staging unregister",
                    )
                {
                    eprintln!("CUDA weight staging cleanup: {error}");
                    if let Some(memory) = slot.memory.take() {
                        std::mem::forget(memory);
                    }
                }
            }
        }
    }
}
impl Staging {
    fn new(session: &Session, count: usize) -> Result<Self> {
        let mut pool = Self {
            stream: session.stream,
            pending: false,
            slots: Vec::with_capacity(count),
            readers: None,
        };
        for _ in 0..count {
            // Ordinary page-aligned CPU mappings keep direct file I/O on the
            // normal page path. Register once for the existing asynchronous DMA.
            let mut memory = memmap2::MmapMut::map_anon(SLOT_BYTES).map_err(|e| e.to_string())?;
            let pointer = memory.as_mut_ptr().cast();
            pool.slots.push(Slot {
                memory: Some(memory),
                pointer,
                event: ptr::null_mut(),
                registered: false,
            });
            // SAFETY: The pool owns this complete page-aligned mapping and
            // retains it until readers join, DMA completes and registration ends.
            unsafe {
                check(
                    sys::cuMemHostRegister_v2(pointer, SLOT_BYTES, 0),
                    "weight staging register",
                )?;
            }
            pool.slots.last_mut().unwrap().registered = true;
            // SAFETY: Event immediately belongs to the pool; timing is disabled.
            unsafe {
                check(
                    sys::cuEventCreate(&mut pool.slots.last_mut().unwrap().event, 2),
                    "weight staging event",
                )?;
            }
        }
        Ok(pool)
    }
}

#[derive(Clone)]
struct Job {
    slot: usize,
    source: Arc<TensorSource>,
    offset: usize,
    bytes: usize,
    destination: u64,
    // A slot is exclusively borrowed by exactly one job. The coordinator only
    // dispatches it after its previous DMA event completes; all readers join
    // before the CUDA owner frees this allocation. Integers carry no CUDA API.
    pointer: usize,
    cache: Option<Arc<Mutex<host_weights::Cache>>>,
}

/// Only slot-sized jobs are queued, so both pending work and completions are
/// bounded by the staging slot count. Threads keep immutable shard files open.
struct Readers {
    jobs: Option<mpsc::SyncSender<Job>>,
    completed: mpsc::Receiver<(Job, ReadReport)>,
    handles: Vec<std::thread::JoinHandle<()>>,
}
impl Readers {
    fn new(workers: usize, slots: usize) -> Result<Self> {
        let (jobs_tx, jobs_rx) = mpsc::sync_channel::<Job>(slots);
        let jobs_rx = Arc::new(Mutex::new(jobs_rx));
        let (done_tx, completed) = mpsc::channel();
        let mut pool = Self {
            jobs: Some(jobs_tx),
            completed,
            handles: vec![],
        };
        for _ in 0..workers {
            let jobs = Arc::clone(&jobs_rx);
            let done = done_tx.clone();
            pool.handles.push(
                std::thread::Builder::new()
                    .name("orin-weight-io".into())
                    .spawn(move || {
                        let mut reader = Reader::default();
                        loop {
                            let Ok(job) = jobs.lock().unwrap().recv() else {
                                break;
                            };
                            // SAFETY: Dispatch exclusively lends this slot until the
                            // completion is received. Staging joins us before freeing it.
                            let target = unsafe {
                                std::slice::from_raw_parts_mut(job.pointer as *mut u8, SLOT_BYTES)
                            };
                            let at = Instant::now();
                            let file_before = reader.file_read_s;
                            let copy_before = reader.cache_copy_s;
                            let result = reader.read(
                                &job.source,
                                job.offset,
                                job.bytes,
                                target,
                                job.cache.as_deref(),
                            );
                            let report = ReadReport {
                                result,
                                service_s: at.elapsed().as_secs_f64(),
                                file_read_s: reader.file_read_s - file_before,
                                cache_copy_s: reader.cache_copy_s - copy_before,
                            };
                            if done.send((job, report)).is_err() {
                                break;
                            }
                        }
                    })
                    .map_err(|e| format!("Start weight reader: {e}"))?,
            );
        }
        Ok(pool)
    }
}
impl Drop for Readers {
    fn drop(&mut self) {
        // Closing dispatch lets every queued read finish. Completions do not
        // block on the CUDA coordinator, including during error unwinding.
        self.jobs.take();
        for handle in self.handles.drain(..) {
            if handle.join().is_err() {
                eprintln!("Weight reader panicked during cleanup");
            }
        }
    }
}

fn unsupported_direct(error: &std::io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EINVAL | libc::EOPNOTSUPP | libc::ENOSYS)
    )
}

fn cache_lock(
    cache: &Mutex<host_weights::Cache>,
) -> std::sync::MutexGuard<'_, host_weights::Cache> {
    let at = Instant::now();
    let mut guard = cache.lock().unwrap();
    guard.statistics.lock_wait_s += at.elapsed().as_secs_f64();
    guard
}

impl Reader {
    /// Reuse shard descriptors and accept EOF tails on sector-aligned direct reads.
    fn read(
        &mut self,
        source: &TensorSource,
        offset: usize,
        bytes: usize,
        target: &mut [u8],
        cache: Option<&Mutex<host_weights::Cache>>,
    ) -> Result<usize> {
        let absolute = source
            .offset
            .checked_add(offset)
            .ok_or("Weight offset overflow")?;
        if bytes == 0
            || bytes > PAYLOAD_BYTES
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > source.bytes)
        {
            return Err("Weight chunk exceeds tensor".into());
        }
        // A retained descriptor must still refer to the named shard. Check the
        // current length too, so truncation/removal retain their error semantics.
        let metadata_at = Instant::now();
        let metadata = std::fs::metadata(&source.path)
            .map_err(|e| format!("{}: {e}", source.path.display()))?;
        let metadata_s = metadata_at.elapsed().as_secs_f64();
        let identity = (metadata.dev(), metadata.ino());
        let size = usize::try_from(metadata.len()).map_err(|_| "Weight shard length overflow")?;
        if self
            .files
            .get(&source.path)
            .is_some_and(|file| file.identity != identity)
        {
            self.files.remove(&source.path);
        }
        if absolute.checked_add(bytes).is_none_or(|end| end > size) {
            return Err(format!(
                "{}: truncated weight payload",
                source.path.display()
            ));
        }
        let mut key =
            cache.map(|_| host_weights::Key::new(&source.path, &metadata, absolute, bytes));
        let mut read_absolute = absolute;
        let mut read_bytes = bytes;
        if let Some(cache) = cache {
            let (resolved, payload) = {
                let mut guard = cache_lock(cache);
                guard.statistics.metadata_s += metadata_s;
                let resolved = guard.resolve(key.take().unwrap());
                let payload = guard.acquire(&resolved, bytes);
                (resolved, payload)
            };
            (read_absolute, read_bytes) = resolved.extent();
            key = Some(resolved);
            if let Some(payload) = payload {
                if target.len() < bytes {
                    return Err("Weight chunk exceeds staging slot".into());
                }
                let at = Instant::now();
                let first = absolute - read_absolute;
                target[..bytes].copy_from_slice(&payload[first..first + bytes]);
                self.cache_copy_s += at.elapsed().as_secs_f64();
                return Ok(0);
            }
        }
        let aligned = read_absolute / ALIGNMENT * ALIGNMENT;
        let skip = read_absolute - aligned;
        let extent = (skip + read_bytes).next_multiple_of(ALIGNMENT);
        if extent > target.len()
            || read_absolute
                .checked_add(read_bytes)
                .is_none_or(|end| end > size)
        {
            return Err("Weight bank exceeds its file or staging extent".into());
        }
        let store = |skip: usize, target: &[u8]| -> Result<usize> {
            if let Some((cache, key)) = cache.zip(key.as_ref())
                && cache_lock(cache).reserve(key)?
            {
                // Reserve first, then fault/copy pages without holding the
                // shared cache lock. All workers' pending bytes are billed.
                let fill_at = Instant::now();
                let mut data = match memmap2::MmapMut::map_anon(read_bytes) {
                    Ok(data) => data,
                    Err(error) => {
                        cache_lock(cache).cancel(key);
                        return Err(format!("Host weight cache allocation: {error}"));
                    }
                };
                data.copy_from_slice(&target[skip..skip + read_bytes]);
                let fill_s = fill_at.elapsed().as_secs_f64();
                let mut guard = cache_lock(cache);
                guard.statistics.fill_s += fill_s;
                guard.commit(key.clone(), data)?;
            }
            Ok(skip + absolute - read_absolute)
        };
        let file = match self.files.entry(source.path.clone()) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let buffered = File::open(&source.path)
                    .map_err(|e| format!("{}: {e}", source.path.display()))?;
                let direct = match OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&source.path)
                {
                    Ok(file) => Some(file),
                    Err(error) if unsupported_direct(&error) => None,
                    Err(error) => return Err(format!("{}: {error}", source.path.display())),
                };
                entry.insert(ReadFile {
                    buffered,
                    direct,
                    size,
                    identity,
                })
            }
        };
        file.size = size;
        if absolute
            .checked_add(bytes)
            .is_none_or(|end| end > file.size)
        {
            return Err(format!(
                "{}: truncated weight payload",
                source.path.display()
            ));
        }
        if (target.as_ptr() as usize).is_multiple_of(ALIGNMENT)
            && let Some(direct) = &file.direct
        {
            let at = Instant::now();
            let read = loop {
                match direct.read_at(&mut target[..extent], aligned as u64) {
                    Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                    result => break result,
                }
            };
            self.file_read_s += at.elapsed().as_secs_f64();
            match read {
                Ok(n) if n == extent.min(file.size - aligned) => return store(skip, target),
                Ok(_) => {
                    return Err(format!(
                        "{}: short direct weight read",
                        source.path.display()
                    ));
                }
                Err(error) if unsupported_direct(&error) => {
                    file.direct = None;
                }
                Err(error) => return Err(format!("{}: {error}", source.path.display())),
            }
        }
        let at = Instant::now();
        let result = file
            .buffered
            .read_exact_at(&mut target[..read_bytes], read_absolute as u64)
            .map_err(|e| format!("{}: {e}", source.path.display()));
        self.file_read_s += at.elapsed().as_secs_f64();
        result?;
        store(0, target)
    }
}

pub(super) fn upload(
    session: &Session,
    uploads: Vec<Upload>,
    workers: usize,
) -> Result<Statistics> {
    upload_with_idle(session, uploads, workers, &mut || Ok(false))
}

pub(super) fn upload_with_idle(
    session: &Session,
    uploads: Vec<Upload>,
    workers: usize,
    idle: &mut dyn FnMut() -> Result<bool>,
) -> Result<Statistics> {
    upload_with_cache(session, uploads, workers, idle, None)
}

pub(super) fn upload_experts(
    session: &Session,
    uploads: Vec<Upload>,
    workers: usize,
    cache: &Arc<Mutex<host_weights::Cache>>,
) -> Result<Statistics> {
    if cache.lock().unwrap().statistics.capacity_bytes == 0 {
        return upload(session, uploads, workers);
    }
    upload_with_cache(session, uploads, workers, &mut || Ok(false), Some(cache))
}

fn upload_with_cache(
    session: &Session,
    mut uploads: Vec<Upload>,
    workers: usize,
    idle: &mut dyn FnMut() -> Result<bool>,
    cache: Option<&Arc<Mutex<host_weights::Cache>>>,
) -> Result<Statistics> {
    if !(1..=32).contains(&workers) {
        return Err("Weight reader count must be 1..32".into());
    }
    if uploads.is_empty() {
        return Ok(Statistics::default());
    }
    for upload in &uploads {
        upload
            .destination
            .checked_add(upload.source.bytes as u64)
            .ok_or("Weight upload address overflow")?;
    }
    uploads
        .sort_by(|a, b| (&a.source.path, a.source.offset).cmp(&(&b.source.path, b.source.offset)));
    let staging_at = Instant::now();
    let mut staging = session.weight_staging.borrow_mut();
    let count = (workers * 2).min(32);
    if staging
        .as_ref()
        .is_none_or(|pool| pool.slots.len() != count)
    {
        *staging = Some(Staging::new(session, count)?);
    }
    let pool = staging.as_mut().unwrap();
    if pool.pending {
        // A previous failed upload may have queued DMA without an event.
        // Reader threads have joined; finish GPU access before any slot reuse.
        // SAFETY: The session retains this pool's stream and CUDA context.
        unsafe {
            check(
                sys::cuStreamSynchronize(session.stream),
                "recover weight staging",
            )?;
        }
        pool.pending = false;
    }
    let staging_s = staging_at.elapsed().as_secs_f64();
    let mut pending = uploads.into_iter().flat_map(|upload| {
        let source = Arc::new(upload.source);
        let destination = upload.destination;
        (0..source.bytes).step_by(PAYLOAD_BYTES).map(move |offset| {
            let bytes = (source.bytes - offset).min(PAYLOAD_BYTES);
            (
                Arc::clone(&source),
                offset,
                bytes,
                destination + offset as u64,
            )
        })
    });
    if pool
        .readers
        .as_ref()
        .is_some_and(|readers| readers.handles.len() != workers)
    {
        drop(pool.readers.take());
    }
    if pool.readers.is_none() {
        pool.readers = Some(Readers::new(workers, pool.slots.len())?);
    }
    pool.pending = true;
    let result = {
        let readers = pool.readers.as_ref().unwrap();
        let jobs_tx = readers.jobs.as_ref().unwrap();
        let done_rx = &readers.completed;
        let enqueue =
            |slot: usize,
             pending: &mut dyn Iterator<Item = (Arc<TensorSource>, usize, usize, u64)>|
             -> Result<bool> {
                let Some((source, offset, bytes, destination)) = pending.next() else {
                    return Ok(false);
                };
                jobs_tx
                    .send(Job {
                        slot,
                        source,
                        offset,
                        bytes,
                        destination,
                        pointer: pool.slots[slot].pointer as usize,
                        cache: cache.cloned(),
                    })
                    .map_err(|_| "Weight reader queue closed")?;
                Ok(true)
            };
        (|| {
            let mut active = 0;
            let mut reading = 0;
            let mut in_flight = VecDeque::new();
            let mut statistics = Statistics {
                staging_s,
                ..Default::default()
            };
            for slot in 0..pool.slots.len() {
                reading += usize::from(enqueue(slot, &mut pending)?);
            }
            active += reading;
            while active != 0 {
                while let Some(&index) = in_flight.front() {
                    let slot: &Slot = &pool.slots[index];
                    // SAFETY: Only the CUDA owner queries events. A completed
                    // event releases this slot for a new CPU read, never sooner.
                    unsafe {
                        let status = sys::cuEventQuery(slot.event);
                        if status == sys::CUresult::CUDA_ERROR_NOT_READY {
                            if reading != 0 {
                                break;
                            }
                            let at = Instant::now();
                            check(
                                sys::cuEventSynchronize(slot.event),
                                "weight upload completion",
                            )?;
                            statistics.dma_wait_s += at.elapsed().as_secs_f64();
                        } else {
                            check(status, "weight upload completion query")?;
                        }
                    }
                    in_flight.pop_front();
                    if enqueue(index, &mut pending)? {
                        reading += 1;
                    } else {
                        active -= 1;
                    }
                }
                if active == 0 {
                    break;
                }
                debug_assert_ne!(reading, 0);
                let ((job, report), wait_s, idle_s) = receive(done_rx, idle)?;
                statistics.reader_service_s += report.service_s;
                statistics.file_read_s += report.file_read_s;
                statistics.cache_copy_s += report.cache_copy_s;
                statistics.read_wait_s += wait_s;
                statistics.idle_work_s += idle_s;
                reading -= 1;
                let skip = report.result?;
                let slot = &pool.slots[job.slot];
                // SAFETY: CPU read completed; GPU allocation covers this chunk.
                // The event completes before the slot is returned to a reader.
                unsafe {
                    check(
                        sys::cuMemcpyHtoDAsync_v2(
                            job.destination,
                            slot.pointer.add(skip),
                            job.bytes,
                            session.stream,
                        ),
                        "stream weight chunk",
                    )?;
                    check(
                        sys::cuEventRecord(slot.event, session.stream),
                        "weight upload event",
                    )?;
                }
                in_flight.push_back(job.slot);
                statistics.chunks += 1;
            }
            Ok(statistics)
        })()
    };
    if result.is_err() {
        // Join every outstanding read before a failed call returns. A later
        // upload can recreate readers only after its pending DMA is recovered.
        drop(pool.readers.take());
    }
    if result.is_ok() {
        // Every slot's DMA event completed before active reached zero.
        pool.pending = false;
    }
    if let (Some(cache), Ok(statistics)) = (cache, &result) {
        cache.lock().unwrap().statistics.copy_s += statistics.cache_copy_s;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn cached_shards_detect_replacement_truncation_and_removal() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("weights");
        fs::write(&path, [7u8; 1024]).unwrap();
        let source = TensorSource {
            path: path.clone(),
            offset: 0,
            bytes: 1024,
        };
        let mut target = memmap2::MmapMut::map_anon(SLOT_BYTES).unwrap();
        let mut reader = Reader::default();
        let skip = reader.read(&source, 0, 1024, &mut target, None).unwrap();
        assert!(target[skip..skip + 1024].iter().all(|&b| b == 7));
        let replacement = directory.path().join("replacement");
        fs::write(&replacement, [19u8; 1024]).unwrap();
        fs::rename(replacement, &path).unwrap();
        let skip = reader.read(&source, 0, 1024, &mut target, None).unwrap();
        assert!(target[skip..skip + 1024].iter().all(|&b| b == 19));
        fs::write(&path, [3u8; 512]).unwrap();
        assert!(reader.read(&source, 0, 1024, &mut target, None).is_err());
        fs::remove_file(path).unwrap();
        assert!(reader.read(&source, 0, 1024, &mut target, None).is_err());
    }

    #[test]
    fn cached_banks_select_exact_rows_and_revalidate_file_identity() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.path();
        let data: Vec<_> = (0..12345).map(|i| (i * 37) as u8).collect();
        fs::write(path, &data).unwrap();
        let bank = TensorSource {
            path: path.into(),
            offset: 173,
            bytes: 8192 + 537,
        };
        let mut cache = host_weights::Cache::new(1 << 20);
        cache.register_bank(&bank);
        let cache = Mutex::new(cache);
        let mut reader = Reader::default();
        let mut target = memmap2::MmapMut::map_anon(SLOT_BYTES).unwrap();
        for (row, bytes) in [(37, 131), (1024, 4097), (8192, 537)] {
            let source = TensorSource {
                path: path.into(),
                offset: bank.offset + row,
                bytes,
            };
            let skip = reader
                .read(&source, 0, bytes, &mut target, Some(&cache))
                .unwrap();
            assert_eq!(
                &target[skip..skip + bytes],
                &data[source.offset..source.offset + bytes]
            );
        }
        assert_eq!(cache.lock().unwrap().statistics.hits, 2);
        assert_eq!(
            cache.lock().unwrap().statistics.requested_bytes,
            131 + 4097 + 537
        );
        let replacement = path.with_extension("replacement");
        let changed: Vec<_> = data.iter().map(|b| b.wrapping_add(19)).collect();
        fs::write(&replacement, &changed).unwrap();
        fs::rename(replacement, path).unwrap();
        let source = TensorSource {
            path: path.into(),
            offset: bank.offset + 37,
            bytes: 131,
        };
        let skip = reader
            .read(&source, 0, source.bytes, &mut target, Some(&cache))
            .unwrap();
        assert_eq!(
            &target[skip..skip + source.bytes],
            &changed[source.offset..source.offset + source.bytes]
        );
        fs::write(path, &changed[..512]).unwrap();
        assert!(
            reader
                .read(&source, 0, source.bytes, &mut target, Some(&cache))
                .is_err()
        );
        fs::remove_file(path).unwrap();
        assert!(
            reader
                .read(&source, 0, source.bytes, &mut target, Some(&cache))
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires the serialized SM120 GPU"]
    fn staging_reuse_preserves_changed_inputs_and_recovers_read_errors() {
        let mut session = Session::new(super::super::Driver::load().unwrap());
        // SAFETY: This test owns the context, stream and recorded allocations.
        unsafe {
            check(sys::cuInit(0), "test init").unwrap();
            check(sys::cuDeviceGet(&mut session.device, 0), "test device").unwrap();
            check(sys::cuCtxGetCurrent(&mut session.previous), "test previous").unwrap();
            let mut context = ptr::null_mut();
            check(
                sys::cuDevicePrimaryCtxRetain(&mut context, session.device),
                "test context",
            )
            .unwrap();
            session.retained = true;
            check(sys::cuCtxSetCurrent(context), "test current").unwrap();
            check(sys::cuStreamCreate(&mut session.stream, 1), "test stream").unwrap();
        }
        let bytes = SLOT_BYTES * 2 + 731;
        let file = tempfile::NamedTempFile::new().unwrap();
        let mut destination = 0;
        // SAFETY: Session owns this complete destination until synchronized cleanup.
        unsafe {
            check(
                sys::cuMemAlloc_v2(&mut destination, bytes),
                "test allocation",
            )
            .unwrap();
        }
        session.buffers.push(destination);
        let source = TensorSource {
            path: file.path().into(),
            offset: 173,
            bytes: bytes - 173,
        };
        let mut slots = None;
        for round in 0..3 {
            let data: Vec<_> = (0..bytes).map(|i| (i * 37 + round * 19) as u8).collect();
            fs::write(file.path(), &data).unwrap();
            let statistics = upload(
                &session,
                vec![Upload {
                    source: source.clone(),
                    destination,
                }],
                1,
            )
            .unwrap();
            assert_eq!(statistics.chunks, 3);
            let pointers: Vec<_> = session
                .weight_staging
                .borrow()
                .as_ref()
                .unwrap()
                .slots
                .iter()
                .map(|slot| slot.pointer)
                .collect();
            if let Some(expected) = &slots {
                assert_eq!(&pointers, expected);
            } else {
                slots = Some(pointers);
            }
            let mut result = vec![0; source.bytes];
            // SAFETY: Upload completed all DMA; both ranges cover source.bytes.
            unsafe {
                check(
                    sys::cuMemcpyDtoH_v2(result.as_mut_ptr().cast(), destination, result.len()),
                    "test readback",
                )
                .unwrap();
            }
            assert_eq!(result, data[173..]);
            let reader_ids = || {
                session
                    .weight_staging
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .readers
                    .as_ref()
                    .unwrap()
                    .handles
                    .iter()
                    .map(|handle| handle.thread().id())
                    .collect::<Vec<_>>()
            };
            let ids = reader_ids();
            upload(
                &session,
                vec![Upload {
                    source: source.clone(),
                    destination,
                }],
                1,
            )
            .unwrap();
            assert_eq!(reader_ids(), ids, "Successful uploads reuse their readers");
            fs::write(file.path(), &data[..SLOT_BYTES]).unwrap();
            assert!(
                upload(
                    &session,
                    vec![Upload {
                        source: source.clone(),
                        destination
                    }],
                    1
                )
                .is_err()
            );
            assert!(
                session
                    .weight_staging
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .readers
                    .is_none(),
                "Failed uploads join all readers before returning"
            );
        }
        session.cleanup().unwrap();
        assert!(session.weight_staging.borrow().is_none());
        session.cleanup().unwrap();
    }

    #[test]
    fn ready_reads_take_priority_and_initialization_errors_propagate() {
        let (tx, rx) = mpsc::channel();
        tx.send(7).unwrap();
        assert_eq!(
            receive(&rx, &mut || panic!("ready read must not wait"))
                .unwrap()
                .0,
            7
        );
        let mut calls = 0;
        let (value, _, idle_s) = receive(&rx, &mut || {
            calls += 1;
            if calls == 3 {
                tx.send(9).unwrap();
            }
            Ok(true)
        })
        .unwrap();
        assert_eq!((value, calls), (9, 3));
        assert!(idle_s >= 0.);
        assert_eq!(
            receive(&rx, &mut || Err("module initialization failed".into())).unwrap_err(),
            "module initialization failed"
        );
        drop(tx);
        assert_eq!(
            receive(&rx, &mut || panic!("closed readers must stop")).unwrap_err(),
            "Weight readers stopped"
        );
    }

    #[test]
    fn direct_reads_preserve_header_offsets_chunks_and_eof_tails() {
        let path = std::env::temp_dir().join(format!(
            "orin-stream-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let data: Vec<_> = (0..SLOT_BYTES * 2 + 731).map(|i| (i * 37) as u8).collect();
        fs::write(&path, &data).unwrap();
        let source = TensorSource {
            path: path.clone(),
            offset: 173,
            bytes: data.len() - 173,
        };
        let mut staging = memmap2::MmapMut::map_anon(SLOT_BYTES).unwrap();
        let mut reader = Reader::default();
        for offset in (0..source.bytes).step_by(PAYLOAD_BYTES) {
            let bytes = (source.bytes - offset).min(PAYLOAD_BYTES);
            let skip = reader
                .read(&source, offset, bytes, &mut staging, None)
                .unwrap();
            assert_eq!(
                &staging[skip..skip + bytes],
                &data[source.offset + offset..source.offset + offset + bytes]
            );
        }
        assert!(
            reader
                .read(&source, source.bytes - 1, 2, &mut staging, None)
                .is_err()
        );
        fs::write(&path, &data[..100]).unwrap();
        assert!(reader.read(&source, 0, 100, &mut staging, None).is_err());
        fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    #[test]
    #[ignore = "requires the serialized SM120 GPU"]
    fn streamed_bytes_and_reader_error_cleanup() {
        let mut session = Session::new(super::super::Driver::load().unwrap());
        // SAFETY: This test owns all CUDA handles on this thread. Session owns
        // each handle immediately after successful creation and cleans it up.
        unsafe {
            check(sys::cuInit(0), "test init").unwrap();
            check(sys::cuDeviceGet(&mut session.device, 0), "test device").unwrap();
            check(
                sys::cuCtxGetCurrent(&mut session.previous),
                "test previous context",
            )
            .unwrap();
            let mut context = ptr::null_mut();
            check(
                sys::cuDevicePrimaryCtxRetain(&mut context, session.device),
                "test context",
            )
            .unwrap();
            session.retained = true;
            check(sys::cuCtxSetCurrent(context), "test current context").unwrap();
            check(sys::cuStreamCreate(&mut session.stream, 1), "test stream").unwrap();
        }
        let path = std::env::temp_dir().join(format!("orin-stream-gpu-{}", std::process::id()));
        let data: Vec<_> = (0..SLOT_BYTES * 2 + 731).map(|i| (i * 37) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let source = TensorSource {
            path: path.clone(),
            offset: 173,
            bytes: data.len() - 173,
        };
        let mut destination = 0;
        // SAFETY: Validated test extent; ownership transfers to Session.
        unsafe {
            check(
                sys::cuMemAlloc_v2(&mut destination, source.bytes),
                "test buffer",
            )
            .unwrap();
        }
        session.buffers.push(destination);
        for workers in [1, 4, 12] {
            let bad = TensorSource {
                path: path.with_extension("missing"),
                offset: 0,
                bytes: 1024,
            };
            assert!(
                upload(
                    &session,
                    vec![
                        Upload {
                            source: source.clone(),
                            destination
                        },
                        Upload {
                            source: bad,
                            destination
                        }
                    ],
                    workers,
                )
                .is_err()
            );
            upload(
                &session,
                vec![Upload {
                    source: source.clone(),
                    destination,
                }],
                workers,
            )
            .unwrap();
            let mut readback = vec![0u8; source.bytes];
            // SAFETY: Upload synchronizes every staging event before returning;
            // source and destination cover the complete tensor extent.
            unsafe {
                check(
                    sys::cuMemcpyDtoH_v2(readback.as_mut_ptr().cast(), destination, readback.len()),
                    "test readback",
                )
                .unwrap();
            }
            assert_eq!(readback, data[173..]);
            // More jobs than staging slots, intentionally reversed in file
            // order, exercise descriptor reuse and event-gated slot recycling.
            let pieces = (0..97)
                .rev()
                .map(|i| Upload {
                    source: TensorSource {
                        path: path.clone(),
                        offset: 173 + i * 137,
                        bytes: 137,
                    },
                    destination: destination + (i * 137) as u64,
                })
                .collect();
            let stats = upload(&session, pieces, workers).unwrap();
            assert_eq!(stats.chunks, 97);
            let mut compact = vec![0u8; 97 * 137];
            // SAFETY: The complete upload drained all slot events and the
            // registered destination covers this compact readback extent.
            unsafe {
                check(
                    sys::cuMemcpyDtoH_v2(compact.as_mut_ptr().cast(), destination, compact.len()),
                    "test scattered readback",
                )
                .unwrap();
            }
            assert_eq!(compact, data[173..173 + 97 * 137]);
        }
        let cache = Arc::new(Mutex::new(host_weights::Cache::new(32 << 20)));
        for _ in 0..2 {
            upload_experts(
                &session,
                vec![Upload {
                    source: source.clone(),
                    destination,
                }],
                4,
                &cache,
            )
            .unwrap();
        }
        assert_eq!(cache.lock().unwrap().statistics.hits, 3);
        let changed: Vec<_> = data.iter().map(|b| b.wrapping_add(17)).collect();
        std::fs::write(&path, &changed).unwrap();
        upload_experts(
            &session,
            vec![Upload {
                source: source.clone(),
                destination,
            }],
            4,
            &cache,
        )
        .unwrap();
        let mut output = vec![0u8; source.bytes];
        // SAFETY: Cache-backed and direct uploads both completed their DMA.
        unsafe {
            check(
                sys::cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), destination, output.len()),
                "test cache readback",
            )
            .unwrap();
        }
        assert_eq!(output, changed[173..]);
        assert_eq!(
            cache.lock().unwrap().statistics.hits,
            3,
            "File changes invalidate cached bytes"
        );
        std::fs::remove_file(path).unwrap();
    }
}
