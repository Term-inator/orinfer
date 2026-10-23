//! Bounded JSONL input and an append-only, identity-checked result journal.
use super::*;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, Write},
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    path::Path,
};

const INPUT_LIMIT: u64 = 32 << 20;
const RESULT_LIMIT: u64 = 256 << 20;

fn line(reader: &mut impl BufRead, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(limit + 1)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err(format!("JSONL line exceeds {limit} bytes"));
    }
    Ok(bytes)
}

pub(super) struct Record {
    pub id: String,
    pub line: usize,
    pub bytes: usize,
    pub body: Result<Value>,
    pub prefix_hint: Option<usize>,
}
fn record(bytes: &[u8], line: usize) -> Record {
    let mut id = format!("line-{line}");
    let body = (|| {
        let mut value: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let object = value.as_object_mut().ok_or("Expected a JSON object")?;
        if let Some(custom) = object.remove("custom_id") {
            let custom = custom
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 256)
                .ok_or("custom_id must be a nonempty string of at most 256 bytes")?;
            id = custom.to_owned();
        }
        if object.contains_key("body")
            || object.contains_key("method")
            || object.contains_key("url")
        {
            if object.get("method").and_then(Value::as_str) != Some("POST")
                || object.get("url").and_then(Value::as_str) != Some("/v1/chat/completions")
            {
                return Err("Batch requests require POST /v1/chat/completions".into());
            }
            let body = object.remove("body").ok_or("Missing request body")?;
            object.remove("method");
            object.remove("url");
            if !object.is_empty() {
                return Err("Unknown batch envelope field".into());
            }
            Ok(body)
        } else {
            Ok(value)
        }
    })();
    Record {
        id,
        line,
        bytes: bytes.len(),
        body,
        prefix_hint: None,
    }
}

pub(super) struct Input {
    reader: BufReader<File>,
    locations: Vec<(u64, usize)>,
    order: Vec<usize>,
    prefix_hints: Vec<Option<usize>>,
    cursor: usize,
    read_at: u64,
    pub ids: BTreeMap<String, usize>,
    pub sha256: String,
    metadata: std::fs::Metadata,
}
impl Input {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| format!("Open input {}: {e}", path.display()))?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file() {
            return Err("Input must be a regular JSONL file".into());
        }
        let mut reader = BufReader::new(file);
        let mut digest = Sha256::new();
        let mut ids = BTreeMap::new();
        let mut n = 0;
        let mut offset = 0;
        let mut locations = vec![];
        loop {
            let bytes = line(&mut reader, INPUT_LIMIT)?;
            if bytes.is_empty() {
                break;
            }
            digest.update(&bytes);
            n += 1;
            let start = offset;
            offset += bytes.len() as u64;
            if bytes.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let r = record(&bytes, n);
            locations.push((start, n));
            if let Some(previous) = ids.insert(r.id.clone(), n) {
                return Err(format!(
                    "Duplicate custom_id {:?} at lines {previous} and {n}",
                    r.id
                ));
            }
        }
        reader.rewind().map_err(|e| e.to_string())?;
        Ok(Self {
            reader,
            locations,
            order: vec![],
            prefix_hints: vec![],
            cursor: 0,
            read_at: 0,
            ids,
            sha256: hex::encode(digest.finalize()),
            metadata,
        })
    }
    pub fn next(&mut self) -> Result<Option<Record>> {
        let current = self
            .reader
            .get_ref()
            .metadata()
            .map_err(|e| e.to_string())?;
        if current.len() != self.metadata.len()
            || current.modified().ok() != self.metadata.modified().ok()
        {
            return Err("Input changed during batch execution".into());
        }
        if self.cursor == self.locations.len() {
            return Ok(None);
        }
        let index = self.order.get(self.cursor).copied().unwrap_or(self.cursor);
        let (offset, physical_line) = self.locations[index];
        if offset != self.read_at {
            self.reader
                .seek(std::io::SeekFrom::Start(offset))
                .map_err(|e| e.to_string())?;
        }
        let bytes = line(&mut self.reader, INPUT_LIMIT)?;
        if bytes.is_empty() {
            return Err("Input ended before indexed record".into());
        }
        self.read_at = offset + bytes.len() as u64;
        self.cursor += 1;
        let mut record = record(&bytes, physical_line);
        record.prefix_hint = self.prefix_hints.get(index).copied().flatten();
        Ok(Some(record))
    }
    pub fn reorder(&mut self, order: Vec<usize>, prefix_hints: Vec<Option<usize>>) -> Result<()> {
        if !prefix_hints.is_empty() && prefix_hints.len() != self.locations.len() {
            return Err("Prefix hints must match the input index".into());
        }
        if !order.is_empty() {
            let mut seen = vec![false; self.locations.len()];
            if order.len() != seen.len()
                || order.iter().any(|&i| {
                    seen.get_mut(i)
                        .is_none_or(|value| std::mem::replace(value, true))
                })
            {
                return Err("Batch order must contain every input record exactly once".into());
            }
        }
        self.order = order;
        self.prefix_hints = prefix_hints;
        self.cursor = 0;
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Outcome {
    Success,
    Truncated,
    Failed,
}
fn outcome(value: &Value) -> Result<Outcome> {
    if !value["error"].is_null() && value["response"].is_null() {
        if !value["error"]["message"].is_string() {
            return Err("Malformed result error".into());
        }
        return Ok(Outcome::Failed);
    }
    if !value["error"].is_null() || value["response"]["status_code"] != 200 {
        return Err("Malformed result response".into());
    }
    match value["response"]["body"]["choices"][0]["finish_reason"].as_str() {
        Some("stop" | "tool_calls") => Ok(Outcome::Success),
        Some("length") => Ok(Outcome::Truncated),
        _ => Err("Missing or invalid result finish_reason".into()),
    }
}

pub(super) struct Journal {
    file: File,
    pub states: BTreeMap<String, Outcome>,
}
impl Journal {
    pub fn open(path: &Path, resume: bool, identity: &Value, input: &Input) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true);
        if !resume {
            options.create_new(true);
        }
        let mut file = options
            .open(path)
            .map_err(|e| format!("Open output {}: {e}", path.display()))?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file()
            || (metadata.dev() == input.metadata.dev() && metadata.ino() == input.metadata.ino())
        {
            return Err("Output must be a regular file distinct from input".into());
        }
        // SAFETY: A live descriptor owned for the whole run; concurrent writers
        // cannot inspect/repair/append this journal while the lock is held.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err("Batch output is locked by another process".into());
        }
        let mut meta = path.as_os_str().to_os_string();
        meta.push(".meta.json");
        let mut states = BTreeMap::new();
        if resume {
            let stored: Value = serde_json::from_reader(
                File::open(&meta).map_err(|e| format!("Open resume identity: {e}"))?,
            )
            .map_err(|e| e.to_string())?;
            if &stored != identity {
                return Err(
                    "Resume identity mismatch: input, model or request defaults changed".into(),
                );
            }
            let mut reader = BufReader::new(&mut file);
            let mut end = 0;
            loop {
                let bytes = line(&mut reader, RESULT_LIMIT)?;
                if bytes.is_empty() {
                    break;
                }
                if !bytes.ends_with(b"\n") {
                    break;
                }
                let value: Value = serde_json::from_slice(&bytes)
                    .map_err(|e| format!("Corrupt completed result: {e}"))?;
                let id = value["custom_id"]
                    .as_str()
                    .ok_or("Result custom_id missing")?;
                if !input.ids.contains_key(id) {
                    return Err("Result ID does not belong to input".into());
                }
                states.insert(id.to_owned(), outcome(&value)?);
                end = reader.stream_position().map_err(|e| e.to_string())?;
            }
            // An interrupted final write is uncommitted. Interior corruption
            // above is an error and never causes truncation of valid results.
            file.set_len(end).map_err(|e| e.to_string())?;
        } else {
            let mut meta = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&meta)
                .map_err(|e| format!("Create batch identity: {e}"))?;
            serde_json::to_writer(&mut meta, identity).map_err(|e| e.to_string())?;
            meta.write_all(b"\n").map_err(|e| e.to_string())?;
            meta.sync_all().map_err(|e| e.to_string())?;
        }
        file.seek(std::io::SeekFrom::End(0))
            .map_err(|e| e.to_string())?;
        Ok(Self { file, states })
    }
    pub fn done(&self, id: &str) -> bool {
        self.states.get(id).is_some_and(|s| *s != Outcome::Failed)
    }
    fn entry(value: &Value) -> Result<(String, Outcome, Vec<u8>)> {
        let id = value["custom_id"].as_str().ok_or("Result ID missing")?;
        let status = outcome(value)?;
        let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
        if bytes.len() as u64 >= RESULT_LIMIT {
            return Err("Result exceeds JSONL line limit".into());
        }
        let mut bytes = bytes;
        bytes.push(b'\n');
        Ok((id.to_owned(), status, bytes))
    }
    #[cfg(test)]
    pub fn append(&mut self, value: &Value) -> Result<()> {
        let (id, status, bytes) = Self::entry(value)?;
        self.file.write_all(&bytes).map_err(|e| e.to_string())?;
        self.states.insert(id, status);
        Ok(())
    }
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_data().map_err(|e| e.to_string())?;
        Ok(())
    }
}

struct WriteEntry {
    bytes: Vec<u8>,
    _memory: Option<Arc<crate::server::preparation::Lease>>,
}
/// One ordered writer overlaps serialization/durable I/O with further requests.
/// Its bounded queue retains each response's memory charge until the write.
pub(super) struct Writer {
    sender: mpsc::Sender<WriteEntry>,
    handle: tokio::task::JoinHandle<Result<()>>,
}
impl Writer {
    pub fn start(journal: &Journal, capacity: usize) -> Result<Self> {
        let mut file = journal.file.try_clone().map_err(|e| e.to_string())?;
        let (sender, mut receiver) = mpsc::channel::<WriteEntry>(capacity);
        let handle = tokio::task::spawn_blocking(move || {
            let mut synced = Instant::now();
            while let Some(entry) = receiver.blocking_recv() {
                file.write_all(&entry.bytes).map_err(|e| e.to_string())?;
                if synced.elapsed().as_secs() >= 1 {
                    file.sync_data().map_err(|e| e.to_string())?;
                    synced = Instant::now();
                }
            }
            let result = file.sync_data().map_err(|e| e.to_string());
            // Release the cloned descriptor before publishing task completion;
            // it shares the journal's flock and must not delay a resumed run.
            drop(file);
            result
        });
        Ok(Self { sender, handle })
    }
    pub async fn append(&self, journal: &mut Journal, completed: Completed) -> Result<()> {
        let (id, status, bytes) = Journal::entry(&completed.value)?;
        self.sender
            .send(WriteEntry {
                bytes,
                _memory: completed._memory,
            })
            .await
            .map_err(|_| "Batch writer disconnected".to_owned())?;
        // Exposed as completed only if finish() later confirms all writes/sync.
        journal.states.insert(id, status);
        Ok(())
    }
    pub async fn finish(self) -> Result<()> {
        drop(self.sender);
        self.handle
            .await
            .map_err(|e| format!("Batch writer failed: {e}"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn writer_error_after_queue_acceptance_is_not_success() {
        let path = std::env::temp_dir().join(format!(
            "orinfer-readonly-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"").unwrap();
        let mut journal = Journal {
            file: File::open(&path).unwrap(),
            states: Default::default(),
        };
        let writer = Writer::start(&journal, 1).unwrap();
        let _ = writer
            .append(
                &mut journal,
                Completed {
                    value: json!({"custom_id":"one","error":null,"response":{"status_code":200,
                "body":{"choices":[{"finish_reason":"stop"}]}}}),
                    bytes: 0,
                    _memory: None,
                },
            )
            .await;
        assert!(writer.finish().await.is_err());
        drop(journal);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn line_limit_is_checked_before_growing_to_the_rest_of_the_input() {
        let mut input = std::io::Cursor::new(b"123456789\nrest\n");
        assert!(line(&mut input, 8).is_err());
        assert_eq!(input.position(), 9);
        let mut input = std::io::Cursor::new(b"{}\r\n\n");
        assert_eq!(line(&mut input, 8).unwrap(), b"{}\r\n");
    }
}
