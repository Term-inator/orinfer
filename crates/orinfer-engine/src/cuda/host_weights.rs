//! Byte-bounded CPU cache of unchanged compressed expert payloads.
use crate::artifact::Result;
use memmap2::MmapMut;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::Metadata,
    os::unix::fs::MetadataExt,
    path::Path,
    sync::{Arc, Condvar},
};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Range {
    path: std::path::PathBuf,
    offset: usize,
    bytes: usize,
}
fn payload_charge(path: &Path, bytes: usize) -> usize {
    // Three identity keys, one range key and conservative index/node costs.
    bytes.next_multiple_of(4096)
        + 3 * std::mem::size_of::<Key>()
        + std::mem::size_of::<Range>()
        + 4 * path.as_os_str().len()
        + 768
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Key {
    path: std::path::PathBuf,
    identity: [u64; 3],
    modified: [i64; 4],
    offset: usize,
    bytes: usize,
}
impl Key {
    pub fn new(path: &Path, metadata: &Metadata, offset: usize, bytes: usize) -> Self {
        Self {
            path: path.into(),
            identity: [metadata.dev(), metadata.ino(), metadata.len()],
            modified: [
                metadata.mtime(),
                metadata.mtime_nsec(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            ],
            offset,
            bytes,
        }
    }
    fn charge(&self) -> usize {
        payload_charge(&self.path, self.bytes)
    }
    fn range(&self) -> Range {
        Range {
            path: self.path.clone(),
            offset: self.offset,
            bytes: self.bytes,
        }
    }
    pub fn extent(&self) -> (usize, usize) {
        (self.offset, self.bytes)
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Statistics {
    pub capacity_bytes: usize,
    pub resident_bytes: usize,
    pub pending_bytes: usize,
    pub peak_resident_bytes: usize,
    pub lookups: u64,
    pub hits: u64,
    pub coalesced_reads: u64,
    pub requested_bytes: u64,
    pub hit_bytes: u64,
    pub evictions: u64,
    pub gpu_priority_evictions: u64,
    pub gpu_covered_bytes: usize,
    /// Reader-side totals; concurrent workers can overlap these intervals.
    pub lock_wait_s: f64,
    pub metadata_s: f64,
    pub copy_s: f64,
    pub fill_s: f64,
    pub bank_metadata_bytes: usize,
}
impl Statistics {
    pub(crate) fn since(self, before: Self) -> Self {
        Self {
            lookups: self.lookups - before.lookups,
            hits: self.hits - before.hits,
            coalesced_reads: self.coalesced_reads - before.coalesced_reads,
            requested_bytes: self.requested_bytes - before.requested_bytes,
            hit_bytes: self.hit_bytes - before.hit_bytes,
            evictions: self.evictions - before.evictions,
            gpu_priority_evictions: self.gpu_priority_evictions - before.gpu_priority_evictions,
            lock_wait_s: self.lock_wait_s - before.lock_wait_s,
            metadata_s: self.metadata_s - before.metadata_s,
            copy_s: self.copy_s - before.copy_s,
            fill_s: self.fill_s - before.fill_s,
            ..self
        }
    }
}

struct Entry {
    gpu: bool,
    stamp: u64,
    payload: Arc<MmapMut>,
    charge: usize,
}
pub(super) struct Cache {
    entries: HashMap<Key, Entry>,
    order: BTreeMap<u64, Key>,
    covered: BTreeMap<u64, Key>,
    ranges: Option<HashMap<Range, Key>>,
    clock: u64,
    pending: HashSet<Key>,
    pub ready: Arc<Condvar>,
    banks: BTreeMap<(std::path::PathBuf, usize), usize>,
    pub statistics: Statistics,
}
impl Cache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: BTreeMap::new(),
            covered: BTreeMap::new(),
            ranges: None,
            clock: 0,
            pending: HashSet::new(),
            ready: Arc::new(Condvar::new()),
            banks: BTreeMap::new(),
            statistics: Statistics {
                capacity_bytes: capacity,
                ..Default::default()
            },
        }
    }
    /// Active readers retain immutable bytes outside the mutex. Leased entries
    /// remain resident and charged until the last reader releases its Arc.
    pub fn acquire(&mut self, key: &Key, requested_bytes: usize) -> Option<Arc<MmapMut>> {
        self.statistics.lookups += 1;
        self.statistics.requested_bytes += requested_bytes as u64;
        let payload = self.lease(key)?;
        self.statistics.hits += 1;
        self.statistics.hit_bytes += requested_bytes as u64;
        Some(payload)
    }
    /// A waiter already counted its request before joining an in-progress fill.
    pub fn lease(&mut self, key: &Key) -> Option<Arc<MmapMut>> {
        let entry = self.entries.get_mut(key)?;
        let queue = if entry.gpu {
            &mut self.covered
        } else {
            &mut self.order
        };
        queue.remove(&entry.stamp);
        self.clock += 1;
        entry.stamp = self.clock;
        queue.insert(entry.stamp, key.clone());
        Some(Arc::clone(&entry.payload))
    }
    pub fn is_pending(&self, key: &Key) -> bool {
        self.pending.contains(key)
    }
    /// Small declared expert tensors can share one cached bank. Register only
    /// extents that fit the uploader; the cache budget still decides admission.
    pub fn register_bank(&mut self, source: &crate::weights::TensorSource) {
        let hint = (source.path.clone(), source.offset);
        let charge = std::mem::size_of::<((std::path::PathBuf, usize), usize)>()
            + source.path.as_os_str().len()
            + 256;
        // The registry itself stays charged, including before the first fill.
        // A bank must fit together with this metadata to be worth registering.
        let payload_charge = payload_charge(&source.path, source.bytes);
        if source.bytes != 0
            && source.offset.checked_add(source.bytes).is_some()
            && !self.banks.contains_key(&hint)
            && self.statistics.resident_bytes
                + self.statistics.pending_bytes
                + charge
                + payload_charge
                <= self.statistics.capacity_bytes
        {
            self.banks.insert(hint, source.bytes);
            self.statistics.resident_bytes += charge;
            self.statistics.bank_metadata_bytes += charge;
            self.statistics.peak_resident_bytes = self
                .statistics
                .peak_resident_bytes
                .max(self.statistics.resident_bytes);
        }
    }
    fn bank_extent(&self, path: &Path, offset: usize, bytes: usize) -> Option<(usize, usize)> {
        let ((bank_path, first), len) = self
            .banks
            .range((path.into(), 0)..=(path.into(), offset))
            .next_back()?;
        (bank_path == path
            && offset
                .checked_add(bytes)
                .is_some_and(|end| end <= first + len))
        .then_some((*first, *len))
    }
    pub fn resolve(&self, key: Key) -> Key {
        if let Some((offset, bytes)) = self.bank_extent(&key.path, key.offset, key.bytes) {
            let bank = Key {
                offset,
                bytes,
                ..key.clone()
            };
            if bank.charge() + self.statistics.bank_metadata_bytes <= self.statistics.capacity_bytes
            {
                return bank;
            }
        }
        key
    }
    /// Coverage changes eviction priority, never payload identity or recency.
    pub fn set_gpu_resident(
        &mut self,
        path: &Path,
        offset: usize,
        bytes: usize,
        present: bool,
    ) -> Result<()> {
        let range = Range {
            path: path.into(),
            offset,
            bytes,
        };
        if self.ranges.is_none() && !present {
            return Ok(());
        }
        let ranges = self.ranges.get_or_insert_with(|| {
            self.entries
                .keys()
                .map(|key| (key.range(), key.clone()))
                .collect()
        });
        let Some(key) = ranges.get(&range).cloned() else {
            return Ok(());
        };
        let entry = self
            .entries
            .get_mut(&key)
            .ok_or("Lost host cache range index")?;
        if entry.gpu == present {
            return Ok(());
        }
        if entry.gpu {
            self.covered.remove(&entry.stamp);
            self.statistics.gpu_covered_bytes -= entry.charge;
        } else {
            self.order.remove(&entry.stamp);
        }
        entry.gpu = present;
        if present {
            self.covered.insert(entry.stamp, key);
            self.statistics.gpu_covered_bytes += entry.charge;
        } else {
            self.order.insert(entry.stamp, key);
        }
        Ok(())
    }
    fn make_room(&mut self, charge: usize) -> Result<bool> {
        while self.statistics.resident_bytes + self.statistics.pending_bytes + charge
            > self.statistics.capacity_bytes
        {
            let covered = self
                .covered
                .iter()
                .find(|(_, key)| Arc::strong_count(&self.entries[*key].payload) == 1);
            let host = self
                .order
                .iter()
                .find(|(_, key)| Arc::strong_count(&self.entries[*key].payload) == 1);
            let victim = match (covered, host) {
                (Some(gpu), Some(cpu)) => {
                    // GPU coverage biases age, but old host-only data must
                    // still expire when the workload moves to a new topic.
                    let gpu_age = self.clock.saturating_sub(*gpu.0).saturating_add(1);
                    let cpu_age = self.clock.saturating_sub(*cpu.0).saturating_add(1);
                    Some(if gpu_age.saturating_mul(4) >= cpu_age {
                        gpu
                    } else {
                        cpu
                    })
                }
                (gpu, cpu) => gpu.or(cpu),
            };
            let Some((&stamp, old)) = victim else {
                return Ok(false);
            };
            let old = old.clone();
            let entry = self
                .entries
                .remove(&old)
                .ok_or("Lost host weight cache entry")?;
            if entry.gpu {
                self.covered.remove(&stamp);
                self.statistics.gpu_covered_bytes -= entry.charge;
                self.statistics.gpu_priority_evictions += 1;
            } else {
                self.order.remove(&stamp);
            }
            let range = old.range();
            if let Some(ranges) = &mut self.ranges
                && ranges.get(&range) == Some(&old)
            {
                ranges.remove(&range);
            }
            self.statistics.resident_bytes -= entry.charge;
            self.statistics.evictions += 1;
        }
        Ok(true)
    }
    pub fn reserve(&mut self, key: &Key) -> Result<bool> {
        let charge = key.charge();
        if self.entries.contains_key(key)
            || self.pending.contains(key)
            || charge > self.statistics.capacity_bytes
        {
            return Ok(false);
        }
        if !self.make_room(charge)? {
            return Ok(false);
        }
        self.pending.insert(key.clone());
        self.statistics.pending_bytes += charge;
        self.statistics.peak_resident_bytes = self
            .statistics
            .peak_resident_bytes
            .max(self.statistics.resident_bytes + self.statistics.pending_bytes);
        Ok(true)
    }
    pub fn cancel(&mut self, key: &Key) {
        assert!(self.pending.remove(key));
        self.statistics.pending_bytes -= key.charge();
        self.ready.notify_all();
    }
    pub fn commit(&mut self, key: Key, data: MmapMut) -> Result<()> {
        let charge = key.charge();
        self.cancel(&key);
        if data.len() != key.bytes {
            return Err("Host weight cache extent mismatch".into());
        }
        if self.entries.contains_key(&key) {
            return Ok(());
        }
        self.clock += 1;
        self.order.insert(self.clock, key.clone());
        if let Some(ranges) = &mut self.ranges {
            ranges.insert(key.range(), key.clone());
        }
        self.entries.insert(
            key,
            Entry {
                gpu: false,
                stamp: self.clock,
                payload: Arc::new(data),
                charge,
            },
        );
        self.statistics.resident_bytes += charge;
        Ok(())
    }
    #[cfg(test)]
    fn insert(&mut self, key: Key, payload: &[u8]) -> Result<()> {
        if self.reserve(&key)? {
            let mut data = MmapMut::map_anon(payload.len()).map_err(|e| e.to_string())?;
            data.copy_from_slice(payload);
            self.commit(key, data)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_covered_payloads_are_evicted_before_host_only_payloads() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let a = Key::new(file.path(), &metadata, 0, 64);
        let b = Key::new(file.path(), &metadata, 64, 64);
        let c = Key::new(file.path(), &metadata, 128, 64);
        let mut cache = Cache::new(a.charge() * 2 + 4096);
        cache.insert(a.clone(), &[1; 64]).unwrap();
        cache.insert(b.clone(), &[2; 64]).unwrap();
        cache.set_gpu_resident(file.path(), 64, 64, true).unwrap();
        cache.insert(c.clone(), &[3; 64]).unwrap();
        assert!(
            cache.acquire(&a, 64).is_some(),
            "Keep the only copy of host-only data"
        );
        assert!(cache.acquire(&b, 64).is_none());
        assert!(cache.acquire(&c, 64).is_some());
        assert!(cache.statistics.resident_bytes <= cache.statistics.capacity_bytes);
    }

    #[test]
    fn covered_leases_stay_charged_and_releasing_coverage_preserves_recency() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let keys: Vec<_> = (0..4)
            .map(|i| Key::new(file.path(), &metadata, i * 64, 64))
            .collect();
        let mut cache = Cache::new(keys[0].charge() * 2 + 4096);
        cache.insert(keys[0].clone(), &[1; 64]).unwrap();
        cache.insert(keys[1].clone(), &[2; 64]).unwrap();
        cache.set_gpu_resident(file.path(), 64, 64, true).unwrap();
        let lease = cache.acquire(&keys[1], 64).unwrap();
        cache.insert(keys[2].clone(), &[3; 64]).unwrap();
        assert!(cache.acquire(&keys[0], 64).is_none());
        assert_eq!(&lease[..], &[2; 64]);
        assert_eq!(cache.statistics.gpu_covered_bytes, keys[1].charge());
        assert_eq!(cache.statistics.gpu_priority_evictions, 0);
        drop(lease);
        cache.set_gpu_resident(file.path(), 64, 64, false).unwrap();
        cache.insert(keys[3].clone(), &[4; 64]).unwrap();
        assert!(cache.acquire(&keys[1], 64).is_none());
        assert!(cache.acquire(&keys[2], 64).is_some());
        assert_eq!(cache.statistics.gpu_covered_bytes, 0);
        assert!(cache.statistics.resident_bytes <= cache.statistics.capacity_bytes);
    }

    #[test]
    fn replacing_file_identity_keeps_the_latest_range_index_after_old_eviction() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let old = Key::new(file.path(), &file.as_file().metadata().unwrap(), 0, 64);
        let mut cache = Cache::new(old.charge() * 2 + 4096);
        cache.insert(old.clone(), &[1; 64]).unwrap();
        cache.set_gpu_resident(file.path(), 0, 64, true).unwrap();
        std::fs::write(file.path(), [2; 256]).unwrap();
        let current = Key::new(file.path(), &file.as_file().metadata().unwrap(), 0, 64);
        cache.insert(current.clone(), &[2; 64]).unwrap();
        cache.set_gpu_resident(file.path(), 0, 64, true).unwrap();
        let another = Key::new(file.path(), &file.as_file().metadata().unwrap(), 64, 64);
        cache.insert(another, &[3; 64]).unwrap();
        assert!(cache.acquire(&old, 64).is_none());
        cache.set_gpu_resident(file.path(), 0, 64, false).unwrap();
        assert_eq!(cache.statistics.gpu_covered_bytes, 0);
        assert_eq!(&cache.acquire(&current, 64).unwrap()[..], &[2; 64]);
        assert!(cache.statistics.gpu_priority_evictions > 0);
    }

    #[test]
    fn stale_host_only_data_does_not_displace_recent_gpu_covered_data_forever() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let a = Key::new(file.path(), &metadata, 0, 64);
        let b = Key::new(file.path(), &metadata, 64, 64);
        let c = Key::new(file.path(), &metadata, 128, 64);
        let mut cache = Cache::new(a.charge() * 2 + 4096);
        cache.insert(a.clone(), &[1; 64]).unwrap();
        cache.insert(b.clone(), &[2; 64]).unwrap();
        for _ in 0..8 {
            cache.set_gpu_resident(file.path(), 64, 64, false).unwrap();
            assert!(cache.acquire(&b, 64).is_some());
            cache.set_gpu_resident(file.path(), 64, 64, true).unwrap();
        }
        cache.insert(c, &[3; 64]).unwrap();
        assert!(
            cache.acquire(&a, 64).is_none(),
            "Old host-only data must age out"
        );
        assert!(cache.acquire(&b, 64).is_some());
    }

    #[test]
    fn pending_fills_share_the_budget_and_release_reservations() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let a = Key::new(file.path(), &metadata, 0, 64);
        let b = Key::new(file.path(), &metadata, 64, 64);
        let c = Key::new(file.path(), &metadata, 128, 64);
        let charge = a.charge();
        let mut cache = Cache::new(charge * 2);
        assert!(cache.reserve(&a).unwrap());
        assert!(cache.reserve(&b).unwrap());
        assert!(!cache.reserve(&c).unwrap());
        assert_eq!(cache.statistics.pending_bytes, charge * 2);
        cache.cancel(&b);
        let mut data = MmapMut::map_anon(64).unwrap();
        data.fill(7);
        cache.commit(a.clone(), data).unwrap();
        assert_eq!(cache.statistics.pending_bytes, 0);
        assert_eq!(cache.statistics.resident_bytes, charge);
        assert!(cache.reserve(&b).unwrap());
        assert!(cache.reserve(&c).unwrap());
        assert_eq!(cache.statistics.resident_bytes, 0);
        assert_eq!(cache.statistics.evictions, 1);
        cache.cancel(&b);
        cache.cancel(&c);
        assert_eq!(cache.statistics.pending_bytes, 0);
        assert_eq!(cache.statistics.peak_resident_bytes, charge * 2);
    }

    #[test]
    fn recency_budget_and_file_identity_preserve_exact_payloads() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), [7u8; 128]).unwrap();
        let m = file.as_file().metadata().unwrap();
        let a = Key::new(file.path(), &m, 0, 64);
        let b = Key::new(file.path(), &m, 64, 64);
        let c = Key::new(file.path(), &m, 32, 64);
        let mut cache = Cache::new(a.charge() * 2);
        cache.insert(a.clone(), &[7; 64]).unwrap();
        cache.insert(b.clone(), &[9; 64]).unwrap();
        assert_eq!(&cache.acquire(&a, 64).unwrap()[..], &[7; 64]);
        cache.insert(c.clone(), &[11; 64]).unwrap();
        assert!(cache.acquire(&b, 64).is_none());
        assert!(cache.acquire(&a, 64).is_some());
        assert!(cache.statistics.resident_bytes <= cache.statistics.capacity_bytes);
        assert_eq!(cache.statistics.evictions, 1);
        std::fs::write(file.path(), [13u8; 256]).unwrap();
        let changed = Key::new(file.path(), &file.as_file().metadata().unwrap(), 0, 64);
        assert!(cache.acquire(&changed, 64).is_none());
        cache.insert(changed.clone(), &[13; 64]).unwrap();
        assert_eq!(&cache.acquire(&changed, 64).unwrap()[..], &[13; 64]);
        let mut tiny = Cache::new(32);
        tiny.insert(a.clone(), &[7; 64]).unwrap();
        assert_eq!(tiny.statistics.resident_bytes, 0);
    }

    #[test]
    fn active_readers_keep_payloads_charged_and_prevent_eviction() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let a = Key::new(file.path(), &metadata, 0, 64);
        let b = Key::new(file.path(), &metadata, 64, 64);
        let mut cache = Cache::new(a.charge());
        cache.insert(a.clone(), &[7; 64]).unwrap();
        let reader = cache.acquire(&a, 64).unwrap();
        assert!(!cache.reserve(&b).unwrap());
        assert_eq!(cache.statistics.resident_bytes, a.charge());
        assert_eq!(cache.statistics.pending_bytes, 0);
        assert_eq!(cache.statistics.evictions, 0);
        assert_eq!(&reader[..], &[7; 64]);
        drop(reader);
        assert!(cache.reserve(&b).unwrap());
        assert!(cache.acquire(&a, 64).is_none());
        assert_eq!(cache.statistics.resident_bytes, 0);
        assert_eq!(cache.statistics.pending_bytes, b.charge());
        cache.cancel(&b);
    }

    #[test]
    fn bank_ranges_preserve_row_accounting_and_budget_limits() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let metadata = file.as_file().metadata().unwrap();
        let source = crate::weights::TensorSource {
            path: file.path().into(),
            offset: 173,
            bytes: 8192,
        };
        let row = Key::new(file.path(), &metadata, 237, 64);
        let bank = Key::new(file.path(), &metadata, 173, 8192);
        let mut cache = Cache::new(bank.charge() * 2);
        cache.register_bank(&source);
        assert_eq!(cache.resolve(row.clone()), bank);
        let data = vec![7; bank.bytes];
        cache.insert(bank.clone(), &data).unwrap();
        assert_eq!(&cache.acquire(&bank, row.bytes).unwrap()[64..128], &[7; 64]);
        assert_eq!(cache.statistics.requested_bytes, 64);
        assert_eq!(cache.statistics.hit_bytes, 64);
        assert_eq!(
            cache.statistics.resident_bytes,
            bank.charge() + cache.statistics.bank_metadata_bytes
        );
        let outside = Key::new(file.path(), &metadata, 173 + 8192 - 32, 64);
        assert_eq!(cache.resolve(outside.clone()), outside);
        let mut small = Cache::new(row.charge());
        small.register_bank(&source);
        assert_eq!(small.resolve(row.clone()), row);
    }
}
