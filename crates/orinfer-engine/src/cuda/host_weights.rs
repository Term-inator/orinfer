//! Byte-bounded CPU cache of unchanged compressed expert payloads.
use crate::artifact::Result;
use memmap2::MmapMut;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap},
    fs::Metadata,
    os::unix::fs::MetadataExt,
    path::Path,
};

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
        // Anonymous mappings release their pages on eviction. Include page tails,
        // both key copies and conservative map/node/path bookkeeping.
        self.bytes.next_multiple_of(4096)
            + 2 * (std::mem::size_of::<Self>() + self.path.as_os_str().len())
            + 512
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
    pub requested_bytes: u64,
    pub hit_bytes: u64,
    pub evictions: u64,
}
impl Statistics {
    pub(crate) fn since(self, before: Self) -> Self {
        Self {
            lookups: self.lookups - before.lookups,
            hits: self.hits - before.hits,
            requested_bytes: self.requested_bytes - before.requested_bytes,
            hit_bytes: self.hit_bytes - before.hit_bytes,
            evictions: self.evictions - before.evictions,
            ..self
        }
    }
}

struct Entry {
    stamp: u64,
    payload: MmapMut,
    charge: usize,
}
pub(super) struct Cache {
    entries: HashMap<Key, Entry>,
    order: BTreeMap<u64, Key>,
    clock: u64,
    pub statistics: Statistics,
}
impl Cache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: BTreeMap::new(),
            clock: 0,
            statistics: Statistics {
                capacity_bytes: capacity,
                ..Default::default()
            },
        }
    }
    pub fn copy(&mut self, key: &Key, output: &mut [u8]) -> bool {
        self.statistics.lookups += 1;
        self.statistics.requested_bytes += key.bytes as u64;
        let Some(entry) = self.entries.get_mut(key) else {
            return false;
        };
        self.order.remove(&entry.stamp);
        self.clock += 1;
        entry.stamp = self.clock;
        self.order.insert(entry.stamp, key.clone());
        output[..key.bytes].copy_from_slice(&entry.payload);
        self.statistics.hits += 1;
        self.statistics.hit_bytes += key.bytes as u64;
        true
    }
    pub fn reserve(&mut self, key: &Key) -> Result<bool> {
        let charge = key.charge();
        if self.entries.contains_key(key) || charge > self.statistics.capacity_bytes {
            return Ok(false);
        }
        while self.statistics.resident_bytes + self.statistics.pending_bytes + charge
            > self.statistics.capacity_bytes
        {
            let Some((_, old)) = self.order.pop_first() else {
                return Ok(false);
            };
            let entry = self
                .entries
                .remove(&old)
                .ok_or("Lost host weight cache entry")?;
            self.statistics.resident_bytes -= entry.charge;
            self.statistics.evictions += 1;
        }
        self.statistics.pending_bytes += charge;
        self.statistics.peak_resident_bytes = self
            .statistics
            .peak_resident_bytes
            .max(self.statistics.resident_bytes + self.statistics.pending_bytes);
        Ok(true)
    }
    pub fn cancel(&mut self, key: &Key) {
        self.statistics.pending_bytes -= key.charge();
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
        self.entries.insert(
            key,
            Entry {
                stamp: self.clock,
                payload: data,
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
        let mut target = [0u8; 64];
        assert!(cache.copy(&a, &mut target));
        assert_eq!(target, [7; 64]);
        cache.insert(c.clone(), &[11; 64]).unwrap();
        assert!(!cache.copy(&b, &mut target));
        assert!(cache.copy(&a, &mut target));
        assert!(cache.statistics.resident_bytes <= cache.statistics.capacity_bytes);
        assert_eq!(cache.statistics.evictions, 1);
        std::fs::write(file.path(), [13u8; 256]).unwrap();
        let changed = Key::new(file.path(), &file.as_file().metadata().unwrap(), 0, 64);
        assert!(!cache.copy(&changed, &mut target));
        cache.insert(changed.clone(), &[13; 64]).unwrap();
        assert!(cache.copy(&changed, &mut target));
        assert_eq!(target, [13; 64]);
        let mut tiny = Cache::new(32);
        tiny.insert(a.clone(), &[7; 64]).unwrap();
        assert_eq!(tiny.statistics.resident_bytes, 0);
    }
}
