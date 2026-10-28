//! Whole compressed-expert SLRU with model-declared slot or direct storage.
use super::{Session, Value, check, expert_slots::Slots, host_weights, sys, weight_upload};
use crate::{
    artifact::{Argument, Dtype, Kernel, Result},
    execution::ResidentExperts,
    weights::TensorSource,
};
use orinfer_model_sdk::abi::ExpertBank;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Instant,
};

type Key = (usize, usize);
#[derive(Default)]
struct Queue {
    keys: BTreeMap<Key, u64>,
    order: BTreeSet<(u64, Key)>,
}
impl Queue {
    fn remove(&mut self, key: Key) -> bool {
        if let Some(tick) = self.keys.remove(&key) {
            self.order.remove(&(tick, key));
            true
        } else {
            false
        }
    }
    fn insert(&mut self, key: Key, tick: u64) {
        self.remove(key);
        self.keys.insert(key, tick);
        self.order.insert((tick, key));
    }
    fn victim(&self, pinned: &BTreeSet<Key>) -> Option<Key> {
        self.order
            .iter()
            .map(|(_, k)| *k)
            .find(|k| !pinned.contains(k))
    }
}

pub use crate::backend::report::ExpertCacheStatistics as Statistics;

impl Statistics {
    /// Interval counters with gauges retained at the end of the interval.
    pub(crate) fn since(self, before: Self) -> Self {
        Self {
            lookups: self.lookups - before.lookups,
            hits: self.hits - before.hits,
            evictions: self.evictions - before.evictions,
            loaded_bytes: self.loaded_bytes - before.loaded_bytes,
            ensure_calls: self.ensure_calls - before.ensure_calls,
            route_wait_s: self.route_wait_s - before.route_wait_s,
            transition_s: self.transition_s - before.transition_s,
            upload_s: self.upload_s - before.upload_s,
            upload_staging_s: self.upload_staging_s - before.upload_staging_s,
            read_wait_s: self.read_wait_s - before.read_wait_s,
            dma_wait_s: self.dma_wait_s - before.dma_wait_s,
            upload_chunks: self.upload_chunks - before.upload_chunks,
            reader_service_s: self.reader_service_s - before.reader_service_s,
            file_read_s: self.file_read_s - before.file_read_s,
            page_reuses: self.page_reuses - before.page_reuses,
            host_cache: self.host_cache.since(before.host_cache),
            ..self
        }
    }
}

pub(super) struct Plan {
    pub banks: Vec<ExpertBank>,
    pub names: BTreeSet<String>,
    pub capacity: usize,
    pub slot_layout: Option<super::expert_slots::Layout>,
}
impl Plan {
    pub fn new(
        prepared: &crate::loader::PreparedModel,
        option: ResidentExperts,
    ) -> Result<Option<Self>> {
        let ResidentExperts::Count(count) = option else {
            return Ok(None);
        };
        if count == 0 || prepared.expert_banks.is_empty() {
            return Err(
                "Finite residency requires a model execution package with expert-cache contracts"
                    .into(),
            );
        }
        if prepared.expert_banks.iter().any(|b| count > b.experts) {
            return Err("Resident expert count exceeds the model's experts per bank".into());
        }
        if prepared.expert_banks.iter().all(|b| count == b.experts) {
            return Ok(None);
        }
        let mut names = BTreeSet::new();
        let buffers: BTreeMap<_, _> = prepared
            .plan
            .buffers
            .iter()
            .map(|b| (b.name.as_str(), b))
            .collect();
        let kernels: BTreeMap<_, _> = prepared
            .plan
            .kernels
            .iter()
            .map(|k| (k.name.as_str(), k))
            .collect();
        let mut consumers = BTreeSet::new();
        let mut banks = prepared.expert_banks.clone();
        // MTP=0 removes verification consumers from the active manifest.
        for bank in &mut banks {
            bank.consumers
                .retain(|name, _| kernels.contains_key(name.as_str()));
        }
        for bank in &banks {
            if bank.experts == 0 || bank.buffers.is_empty() || bank.consumers.is_empty() {
                return Err("Empty expert residency contract".into());
            }
            for name in &bank.buffers {
                let b = buffers.get(name.as_str()).ok_or("Missing expert tensor")?;
                if b.data.is_none()
                    || b.access != crate::artifact::Access::Read
                    || b.shape.first() != Some(&bank.experts)
                    || !names.insert(name.clone())
                    || bank.addresses.is_empty()
                        && !kernels.values().any(|k| {
                            k.args.iter().any(
                                |a| matches!(a, Argument::Buffer { name: used } if used == name),
                            )
                        })
                {
                    return Err("Invalid or overlapping expert tensor bank".into());
                }
            }
            if !bank.addresses.is_empty() {
                if bank.addresses.len() != bank.buffers.len() {
                    return Err("Incomplete expert address tables".into());
                }
                for name in &bank.addresses {
                    let b = buffers
                        .get(name.as_str())
                        .ok_or("Missing expert address table")?;
                    if b.dtype != Dtype::U64 || b.shape != [bank.experts] || b.data.is_some() {
                        return Err("Invalid expert address table".into());
                    }
                }
                if kernels.values().any(|k| {
                    k.args.iter().any(
                        |a| matches!(a, Argument::Buffer {name} if bank.buffers.contains(name)),
                    )
                }) {
                    return Err("Compact expert consumer bypasses address tables".into());
                }
            }
            for (name, index) in &bank.consumers {
                let k = kernels
                    .get(name.as_str())
                    .ok_or("Unknown expert consumer")?;
                let Some(Argument::Buffer { name: counts }) = k.args.get(*index) else {
                    return Err("Expert histogram must be a buffer argument".into());
                };
                let b = buffers
                    .get(counts.as_str())
                    .ok_or("Missing expert counts")?;
                if b.dtype != Dtype::I32
                    || b.shape != [bank.experts]
                    || b.data.is_some()
                    || !consumers.insert(name.clone())
                {
                    return Err("Invalid expert histogram contract".into());
                }
                if !k
                    .args
                    .iter()
                    .any(|a| matches!(a, Argument::Buffer { name } if bank.buffers.contains(name) || bank.addresses.contains(name)))
                {
                    return Err("Expert consumer does not use its bank".into());
                }
            }
        }
        let capacity = count
            .checked_mul(prepared.expert_banks.len())
            .ok_or("Expert capacity overflow")?;
        let sizes = prepared
            .plan
            .buffers
            .iter()
            .map(|b| Ok((b.name.clone(), b.bytes()?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let slot_layout = super::expert_slots::Layout::new(&banks, capacity, &sizes)?;
        Ok(Some(Self {
            slot_layout,
            banks,
            names,
            capacity,
        }))
    }
}

#[derive(Default)]
struct Page {
    owners: usize,
    /// Large tensors are populated by expert range, not by physical page.
    initialized: BTreeSet<usize>,
}

pub(super) struct Pager {
    banks: Vec<ExpertBank>,
    sources: BTreeMap<String, TensorSource>,
    consumers: BTreeMap<String, (usize, usize)>,
    refs: BTreeMap<(String, usize), Page>,
    slots: Option<Slots>,
    prob: Queue,
    hot: Queue,
    transient: BTreeSet<Key>,
    clock: u64,
    workers: usize,
    host_cache: Arc<Mutex<host_weights::Cache>>,
    host_priority: bool,
    pub statistics: Statistics,
    poisoned: bool,
}
impl Pager {
    pub(super) fn consumes(&self, name: &str) -> bool {
        self.consumers.contains_key(name)
    }
    pub(super) fn uses_bank(&self, kernel: &Kernel) -> bool {
        kernel.args.iter().any(|argument| {
            matches!(argument, Argument::Buffer { name } | Argument::BufferSlice { name, .. }
                if self.sources.contains_key(name) || self.banks.iter().any(|b| b.addresses.contains(name)))
        })
    }
    pub fn new(
        plan: Plan,
        sources: BTreeMap<String, TensorSource>,
        workers: usize,
        host_cache_bytes: usize,
        slots: Option<Slots>,
    ) -> Self {
        let consumers = plan
            .banks
            .iter()
            .enumerate()
            .flat_map(|(i, b)| b.consumers.iter().map(move |(k, n)| (k.clone(), (i, *n))))
            .collect();
        let mapped_bytes = slots.as_ref().map_or(0, |slots| slots.bytes);
        let mut host_cache = host_weights::Cache::new(host_cache_bytes);
        for source in sources
            .values()
            .filter(|source| source.bytes <= weight_upload::PAYLOAD_BYTES)
        {
            host_cache.register_bank(source);
        }
        Self {
            banks: plan.banks,
            sources,
            consumers,
            refs: BTreeMap::new(),
            slots,
            prob: Queue::default(),
            hot: Queue::default(),
            transient: BTreeSet::new(),
            clock: 0,
            workers,
            host_cache: Arc::new(Mutex::new(host_cache)),
            host_priority: false,
            statistics: Statistics {
                capacity_experts: plan.capacity,
                mapped_bytes,
                peak_mapped_bytes: mapped_bytes,
                ..Default::default()
            },
            poisoned: false,
        }
    }
    fn pages(&self, s: &Session, key: Key) -> Result<Vec<(String, usize)>> {
        let reservations = s.expert_buffers.borrow();
        let mut pages = vec![];
        for name in &self.banks[key.0].buffers {
            let r = &reservations[name];
            let stride = self.sources[name].bytes / self.banks[key.0].experts;
            let start = key.1 * stride;
            let end = start + stride;
            for offset in (start / r.granularity * r.granularity..end).step_by(r.granularity) {
                pages.push((name.clone(), offset));
            }
        }
        Ok(pages)
    }
    fn host_residency(&self, key: Key, present: bool) -> Result<()> {
        let bank = &self.banks[key.0];
        let mut cache = self.host_cache.lock().unwrap();
        for name in &bank.buffers {
            let source = &self.sources[name];
            // Whole small banks (scales) are shared by many GPU owners.
            if source.bytes <= weight_upload::PAYLOAD_BYTES {
                continue;
            }
            let stride = source.bytes / bank.experts;
            let first = source
                .offset
                .checked_add(key.1 * stride)
                .ok_or("Host residency range overflow")?;
            for offset in (0..stride).step_by(weight_upload::PAYLOAD_BYTES) {
                cache.set_gpu_resident(
                    &source.path,
                    first + offset,
                    (stride - offset).min(weight_upload::PAYLOAD_BYTES),
                    present,
                )?;
            }
        }
        Ok(())
    }
    fn release(&mut self, s: &Session, key: Key) -> Result<()> {
        if let Some(slots) = &mut self.slots {
            slots.release(key)?;
        } else {
            for page in self.pages(s, key)? {
                let n = self
                    .refs
                    .get_mut(&page)
                    .ok_or("Lost expert page ownership")?;
                if n.owners == 1 {
                    s.expert_buffers
                        .borrow_mut()
                        .get_mut(&page.0)
                        .unwrap()
                        .unmap_page(s, page.1)?;
                    self.refs.remove(&page);
                } else {
                    n.owners -= 1;
                }
            }
        }
        if self.host_priority {
            self.host_residency(key, false)?;
        }
        Ok(())
    }
    pub fn ensure(&mut self, s: &Session, kernel: &Kernel, values: &[Value]) -> Result<()> {
        let Some(&(bank, argument)) = self.consumers.get(&kernel.name) else {
            return Ok(());
        };
        if self.poisoned {
            return Err("Expert cache is quarantined after a failed transition".into());
        }
        let Some(Value::Pointer(address)) = values.get(argument) else {
            return Err("Unbound expert histogram".into());
        };
        let route_at = Instant::now();
        self.statistics.ensure_calls += 1;
        let mut counts = vec![0i32; self.banks[bank].experts];
        // SAFETY: Producer and all previous consumers finish before page eviction;
        // the model-declared histogram has this exact validated byte extent.
        unsafe {
            check(sys::cuStreamSynchronize(s.stream), "expert route barrier")?;
            check(
                sys::cuMemcpyDtoH_v2(counts.as_mut_ptr().cast(), *address, counts.len() * 4),
                "expert counts",
            )?;
        }
        self.statistics.route_wait_s += route_at.elapsed().as_secs_f64();
        let transition_at = Instant::now();
        if counts.iter().any(|&n| n < 0) {
            return Err("Negative expert route count".into());
        }
        let wanted: BTreeSet<_> = counts
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(e, _)| (bank, e))
            .collect();
        self.poisoned = true;
        for key in std::mem::take(&mut self.transient) {
            self.release(s, key)?;
        }
        let mut uploads = vec![];
        let mut loaded = vec![];
        for &key in &wanted {
            self.clock = self
                .clock
                .checked_add(1)
                .ok_or("Expert cache clock overflow")?;
            self.statistics.lookups += 1;
            if self.hot.keys.contains_key(&key) || self.prob.keys.contains_key(&key) {
                self.statistics.hits += 1;
                self.prob.remove(key);
                self.hot.insert(key, self.clock);
                if self.hot.keys.len() > self.statistics.capacity_experts * 3 / 4 {
                    let old = self.hot.order.first().unwrap().1;
                    self.hot.remove(old);
                    self.prob.insert(old, self.clock);
                }
                continue;
            }
            if self.host_priority {
                loaded.push(key);
            }
            let capacity = self.statistics.capacity_experts;
            if self.prob.keys.len() + self.hot.keys.len() >= capacity
                && let Some(victim) = self
                    .prob
                    .victim(&wanted)
                    .or_else(|| self.hot.victim(&wanted))
            {
                self.release(s, victim)?;
                self.prob.remove(victim);
                self.hot.remove(victim);
                self.statistics.evictions += 1;
            }
            if let Some(slots) = &mut self.slots {
                self.statistics.loaded_bytes +=
                    slots.load(key, &self.banks[bank], &self.sources, &mut uploads)? as u64;
            } else {
                let pages = self.pages(s, key)?;
                for page in &pages {
                    if !self.refs.contains_key(page) {
                        let mut reservations = s.expert_buffers.borrow_mut();
                        let r = reservations.get_mut(&page.0).unwrap();
                        r.map_page(s, page.1)?;
                        let source = &self.sources[&page.0];
                        // Small banks (such as row scales) fit one physical page.
                        // Populate them once, rather than issue many tiny reads.
                        if source.bytes <= r.granularity {
                            uploads.push(weight_upload::Upload {
                                source: source.clone(),
                                destination: r.address,
                            });
                            self.statistics.loaded_bytes += source.bytes as u64;
                        }
                    }
                    self.refs.entry(page.clone()).or_default().owners += 1;
                }
                for name in &self.banks[bank].buffers {
                    let reservations = s.expert_buffers.borrow();
                    let r = &reservations[name];
                    let source = &self.sources[name];
                    if source.bytes <= r.granularity {
                        continue;
                    }
                    let covered: Vec<_> = pages.iter().filter(|p| &p.0 == name).collect();
                    if covered
                        .iter()
                        .all(|p| self.refs[*p].initialized.contains(&key.1))
                    {
                        continue;
                    }
                    let stride = source.bytes / self.banks[bank].experts;
                    let offset = key.1 * stride;
                    uploads.push(weight_upload::Upload {
                        source: TensorSource {
                            path: source.path.clone(),
                            offset: source.offset + offset,
                            bytes: stride,
                        },
                        destination: r.address + offset as u64,
                    });
                    self.statistics.loaded_bytes += stride as u64;
                    // A failed upload quarantines the pager. These markers are only
                    // consumed after the complete upload succeeds. Unmapping a page
                    // discards its markers, even if a neighboring page stays mapped.
                    for page in covered {
                        self.refs.get_mut(page).unwrap().initialized.insert(key.1);
                    }
                }
            }
            if self.prob.keys.len() + self.hot.keys.len() < capacity {
                self.prob.insert(key, self.clock);
            } else {
                self.transient.insert(key);
            }
        }
        s.release_expert_spares()?;
        self.statistics.page_reuses = s.expert_page_reuses.get();
        self.statistics.transition_s += transition_at.elapsed().as_secs_f64();
        let upload_at = Instant::now();
        let uploaded = weight_upload::upload_experts(s, uploads, self.workers, &self.host_cache)?;
        if let Some(slots) = &self.slots {
            slots.publish(s, bank, self.banks[bank].experts)?;
        }
        // Only staged sources are GPU-covered. First enable priority after
        // actual host eviction pressure; fitting CPU working sets stay on LRU.
        if !self.host_priority && self.host_cache.lock().unwrap().statistics.evictions > 0 {
            self.host_priority = true;
            for &key in self
                .prob
                .keys
                .keys()
                .chain(self.hot.keys.keys())
                .chain(self.transient.iter())
            {
                self.host_residency(key, true)?;
            }
        } else if self.host_priority {
            for key in loaded {
                self.host_residency(key, true)?;
            }
        }
        self.statistics.host_cache = self.host_cache.lock().unwrap().statistics;
        self.statistics.upload_s += upload_at.elapsed().as_secs_f64();
        self.statistics.upload_staging_s += uploaded.staging_s;
        self.statistics.read_wait_s += uploaded.read_wait_s;
        self.statistics.dma_wait_s += uploaded.dma_wait_s;
        self.statistics.upload_chunks += uploaded.chunks;
        self.statistics.reader_service_s += uploaded.reader_service_s;
        self.statistics.file_read_s += uploaded.file_read_s;
        self.statistics.resident_experts = self.prob.keys.len() + self.hot.keys.len();
        self.statistics.transient_experts = self.transient.len();
        self.statistics.peak_transient_experts = self
            .statistics
            .peak_transient_experts
            .max(self.transient.len());
        self.statistics.mapped_bytes =
            self.slots
                .as_ref()
                .map(|slots| slots.bytes)
                .unwrap_or_else(|| {
                    s.expert_buffers
                        .borrow()
                        .values()
                        .map(|r| r.physical_bytes())
                        .sum()
                });
        self.statistics.peak_mapped_bytes = self
            .statistics
            .peak_mapped_bytes
            .max(self.statistics.mapped_bytes);
        self.poisoned = false;
        Ok(())
    }
}

impl super::executor::Executor {
    #[cfg(test)]
    pub(crate) fn fail_expert_reads_for_test(&self, directory: &std::path::Path) {
        let mut cache = self.expert_cache.borrow_mut();
        for source in cache.as_mut().unwrap().sources.values_mut() {
            source.path = directory.join("missing-expert");
        }
    }
    pub(crate) fn expert_statistics(&self) -> Option<Statistics> {
        self.expert_cache.borrow().as_ref().map(|p| p.statistics)
    }
    pub(super) fn ensure_experts(&self, kernel: &Kernel, values: &[Value]) -> Result<()> {
        if let Some(pager) = self.expert_cache.borrow_mut().as_mut() {
            pager.ensure(&self.session, kernel, values)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interval_statistics_separate_counters_and_current_gauges() {
        let before = Statistics {
            lookups: 11,
            hits: 3,
            loaded_bytes: 1024,
            upload_s: 2.,
            ..Default::default()
        };
        let end = Statistics {
            lookups: 20,
            hits: 8,
            loaded_bytes: 4096,
            upload_s: 5.,
            resident_experts: 7,
            mapped_bytes: 8192,
            ..Default::default()
        };
        let interval = end.since(before);
        assert_eq!(interval.lookups, 9);
        assert_eq!(interval.hits, 5);
        assert_eq!(interval.loaded_bytes, 3072);
        assert_eq!(interval.upload_s, 3.);
        assert_eq!(interval.resident_experts, 7);
        assert_eq!(interval.mapped_bytes, 8192);
    }

    #[test]
    fn victim_selection_preserves_the_entire_current_union() {
        let mut q = Queue::default();
        for i in 0..4 {
            q.insert((0, i), i as u64);
        }
        let pinned = BTreeSet::from([(0, 0), (0, 1), (0, 3)]);
        assert_eq!(q.victim(&pinned), Some((0, 2)));
        q.insert((0, 2), 7);
        assert_eq!(q.keys.len(), 4);
        assert_eq!(q.order.len(), 4);
        assert_eq!(q.victim(&q.keys.keys().copied().collect()), None);
        assert!(q.remove((0, 2)));
        assert!(!q.remove((0, 2)));
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::super::{Driver, virtual_memory::Reservation};
    use super::*;
    use std::ptr;

    #[test]
    #[ignore = "requires the serialized SM120 GPU"]
    fn sparse_ranges_preserve_neighbors_tails_and_eviction() {
        let mut s = Session::new(Driver::load().unwrap());
        // SAFETY: This test owns its retained context, stream and allocations.
        // Session records every handle and drains outstanding work on cleanup.
        unsafe {
            check(sys::cuInit(0), "test init").unwrap();
            check(sys::cuDeviceGet(&mut s.device, 0), "test device").unwrap();
            check(sys::cuCtxGetCurrent(&mut s.previous), "test previous").unwrap();
            let mut context = ptr::null_mut();
            check(
                sys::cuDevicePrimaryCtxRetain(&mut context, s.device),
                "test context",
            )
            .unwrap();
            s.retained = true;
            check(sys::cuCtxSetCurrent(context), "test current").unwrap();
            check(sys::cuStreamCreate(&mut s.stream, 1), "test stream").unwrap();
        }
        let probe = Reservation::reserve(&s, 1, 1).unwrap();
        let granularity = probe.granularity;
        s.expert_buffers.borrow_mut().insert("probe".into(), probe);
        let stride = granularity / 2 + 173;
        let weight_bytes = stride * 5;
        let scale_bytes = 85;
        let file = tempfile::NamedTempFile::new().unwrap();
        let data: Vec<_> = (0..173 + weight_bytes + 97 + scale_bytes)
            .map(|i| ((i * 37 + i / 251) % 256) as u8)
            .collect();
        std::fs::write(file.path(), &data).unwrap();
        let mut sources = BTreeMap::new();
        for (name, offset, bytes) in [
            ("weights", 173, weight_bytes),
            ("scales", 173 + weight_bytes + 97, scale_bytes),
        ] {
            sources.insert(
                name.into(),
                TensorSource {
                    path: file.path().into(),
                    offset,
                    bytes,
                },
            );
            let reservation = Reservation::reserve(&s, bytes, 1).unwrap();
            s.expert_buffers
                .borrow_mut()
                .insert(name.into(), reservation);
        }
        let mut counts_address = 0;
        // SAFETY: The session owns this complete five-element histogram.
        unsafe {
            check(sys::cuMemAlloc_v2(&mut counts_address, 20), "test counts").unwrap();
        }
        s.buffers.push(counts_address);
        let identity = serde_json::json!({"file":"unused","sha256":"0".repeat(64)});
        let kernel: Kernel = serde_json::from_value(serde_json::json!({
            "name":"test", "symbol":"unused", "module":identity, "source":identity,
            "host_abi":identity, "grid":[1,1,1], "block":[1,1,1],
            "shared_memory_bytes":0,"cooperative":false,"args":[]
        }))
        .unwrap();
        let mut pager = Pager::new(
            Plan {
                banks: vec![ExpertBank {
                    experts: 5,
                    buffers: vec!["weights".into(), "scales".into()],
                    addresses: vec![],
                    consumers: BTreeMap::from([("test".into(), 0)]),
                }],
                names: BTreeSet::from(["weights".into(), "scales".into()]),
                capacity: 2,
                slot_layout: None,
            },
            sources,
            4,
            32 << 20,
            None,
        );
        for (round, wanted) in [
            vec![0],
            vec![1],
            vec![1],
            vec![2],
            vec![0],
            vec![4],
            vec![0, 1, 4],
            vec![3],
            vec![0],
        ]
        .into_iter()
        .enumerate()
        {
            let mut counts = [0i32; 5];
            for &expert in &wanted {
                counts[expert] = 1;
            }
            // SAFETY: Complete host/device histogram extents; synchronous copy.
            unsafe {
                check(
                    sys::cuMemcpyHtoD_v2(counts_address, counts.as_ptr().cast(), 20),
                    "test route",
                )
                .unwrap();
            }
            let loaded_before = pager.statistics.loaded_bytes;
            pager
                .ensure(&s, &kernel, &[Value::Pointer(counts_address)])
                .unwrap();
            if round == 0 {
                assert_eq!(
                    pager.statistics.loaded_bytes,
                    (stride + scale_bytes) as u64,
                    "The first expert must not read unused neighboring weights"
                );
            }
            if round == 4 {
                assert_eq!(
                    pager.statistics.loaded_bytes, loaded_before,
                    "Initialized bytes survive logical eviction while a neighbor owns the page"
                );
            }
            // SAFETY: Uploaded weights are queued on this owned stream.
            unsafe {
                check(
                    sys::cuStreamSynchronize(s.stream),
                    "test expert upload ready",
                )
                .unwrap();
            }
            for expert in wanted {
                let reservations = s.expert_buffers.borrow();
                for (name, width) in [("weights", stride), ("scales", 17)] {
                    let mut result = vec![0u8; width];
                    // SAFETY: ensure mapped and initialized this selected range;
                    // compare full ranges, including shared-page and EOF tails.
                    unsafe {
                        check(
                            sys::cuMemcpyDtoH_v2(
                                result.as_mut_ptr().cast(),
                                reservations[name].address + (expert * width) as u64,
                                width,
                            ),
                            "test readback",
                        )
                        .unwrap();
                    }
                    let offset = pager.sources[name].offset + expert * width;
                    assert_eq!(
                        result,
                        data[offset..offset + width],
                        "round {round}, expert {expert}, {name}"
                    );
                }
            }
        }
        assert!(pager.statistics.evictions > 0);
        assert!(pager.statistics.peak_transient_experts > 0);
        assert!(pager.statistics.page_reuses > 0);
        assert!(s.expert_spares.borrow().is_empty());
        s.cleanup().unwrap();
        s.cleanup().unwrap();
    }
}

#[cfg(test)]
mod compact_gpu_tests {
    use super::super::Driver;
    use super::*;
    use std::ptr;

    #[test]
    #[ignore = "requires the serialized SM120 GPU"]
    fn compact_slots_preserve_all_banks_tails_transients_and_reuse() {
        let mut s = Session::new(Driver::load().unwrap());
        // SAFETY: This test exclusively owns the retained context and stream.
        unsafe {
            check(sys::cuInit(0), "test init").unwrap();
            check(sys::cuDeviceGet(&mut s.device, 0), "test device").unwrap();
            check(sys::cuCtxGetCurrent(&mut s.previous), "test previous").unwrap();
            let mut context = ptr::null_mut();
            check(
                sys::cuDevicePrimaryCtxRetain(&mut context, s.device),
                "test retain",
            )
            .unwrap();
            s.retained = true;
            check(sys::cuCtxSetCurrent(context), "test current").unwrap();
            check(sys::cuStreamCreate(&mut s.stream, 1), "test stream").unwrap();
        }
        let file = tempfile::NamedTempFile::new().unwrap();
        let payload: Vec<u8> = (0..8192).map(|i| ((i * 37 + i / 31) % 256) as u8).collect();
        std::fs::write(file.path(), &payload).unwrap();
        let mut banks = Vec::new();
        let mut sources = BTreeMap::new();
        let mut pointers = BTreeMap::new();
        let mut cursor = 173;
        for bank in 0..2 {
            let mut buffers = Vec::new();
            let mut addresses = Vec::new();
            for (column, stride) in [397, 17].into_iter().enumerate() {
                let name = format!("bank{bank}-{column}");
                sources.insert(
                    name.clone(),
                    TensorSource {
                        path: file.path().to_path_buf(),
                        offset: cursor,
                        bytes: stride * 5,
                    },
                );
                cursor += stride * 5 + 23;
                let table = name.clone() + "_addresses";
                let mut address = 0;
                // SAFETY: Five complete U64 entries, owned before a later failure.
                unsafe {
                    check(sys::cuMemAlloc_v2(&mut address, 40), "test table").unwrap();
                }
                s.buffers.push(address);
                pointers.insert(table.clone(), address);
                buffers.push(name);
                addresses.push(table);
            }
            banks.push(ExpertBank {
                experts: 5,
                buffers,
                addresses,
                consumers: BTreeMap::from([(format!("test{bank}"), 0)]),
            });
        }
        let sizes = sources
            .iter()
            .map(|(name, source)| (name.clone(), source.bytes))
            .collect();
        let slot_layout = super::super::expert_slots::Layout::new(&banks, 2, &sizes).unwrap();
        let plan = Plan {
            slot_layout,
            names: sources.keys().cloned().collect(),
            banks: banks.clone(),
            capacity: 2,
        };
        let slots = Slots::new(&mut s, &plan, &pointers).unwrap().unwrap();
        assert_eq!(slots.bytes, 5 * 768);
        let expected_bytes = slots.bytes;
        let mut pager = Pager::new(plan, sources.clone(), 2, 1 << 20, Some(slots));
        let mut counts_address = 0;
        // SAFETY: Five I32 counts, retained by the test session.
        unsafe {
            check(sys::cuMemAlloc_v2(&mut counts_address, 20), "test counts").unwrap();
        }
        s.buffers.push(counts_address);
        let identity = serde_json::json!({"file":"unused","sha256":"0".repeat(64)});
        for (round, (bank, wanted)) in [
            (0, vec![0]),
            (1, vec![4]),
            (0, vec![0, 1, 2, 3, 4]),
            (1, vec![0, 4]),
            (0, vec![3]),
            (1, vec![1, 2, 3]),
            (0, vec![]),
        ]
        .into_iter()
        .enumerate()
        {
            let counts: Vec<i32> = (0..5).map(|e| i32::from(wanted.contains(&e))).collect();
            // SAFETY: Both complete histogram buffers are live until this copy returns.
            unsafe {
                check(
                    sys::cuMemcpyHtoD_v2(counts_address, counts.as_ptr().cast(), 20),
                    "test route",
                )
                .unwrap();
            }
            let kernel:Kernel=serde_json::from_value(serde_json::json!({"name":format!("test{bank}"),"symbol":"unused","module":identity,"source":identity,"host_abi":identity,"grid":[1,1,1],"block":[1,1,1],"shared_memory_bytes":0,"cooperative":false,"args":[]})).unwrap();
            pager
                .ensure(&s, &kernel, &[Value::Pointer(counts_address)])
                .unwrap();
            // SAFETY: Drain the table updates before synchronous readback.
            unsafe {
                check(sys::cuStreamSynchronize(s.stream), "test ready").unwrap();
            }
            for (name, table) in banks[bank].buffers.iter().zip(&banks[bank].addresses) {
                let mut table_values = [0u64; 5];
                // SAFETY: Exactly five initialized entries in this bank's published table.
                unsafe {
                    check(
                        sys::cuMemcpyDtoH_v2(table_values.as_mut_ptr().cast(), pointers[table], 40),
                        "test addresses",
                    )
                    .unwrap();
                }
                for &e in &wanted {
                    assert_ne!(table_values[e], 0);
                    let source = &sources[name];
                    let stride = source.bytes / 5;
                    let mut actual = vec![0u8; stride];
                    // SAFETY: Each published expert owns a complete matrix in the pool.
                    unsafe {
                        check(
                            sys::cuMemcpyDtoH_v2(
                                actual.as_mut_ptr().cast(),
                                table_values[e],
                                stride,
                            ),
                            "test expert",
                        )
                        .unwrap();
                    }
                    assert_eq!(
                        actual,
                        payload[source.offset + e * stride..source.offset + (e + 1) * stride],
                        "round {round} bank {bank} expert {e}"
                    );
                }
            }
            assert_eq!(pager.statistics.mapped_bytes, expected_bytes);
            assert!(s.expert_buffers.borrow().is_empty());
        }
        assert!(pager.statistics.peak_transient_experts >= 3);
        assert!(pager.statistics.evictions > 0);
        assert_eq!(pager.statistics.page_reuses, 0);
        std::fs::remove_file(file.path()).unwrap();
        let counts = [0i32, 0, 0, 0, 1];
        // SAFETY: The live source and destination cover the complete histogram.
        unsafe {
            check(
                sys::cuMemcpyHtoD_v2(counts_address, counts.as_ptr().cast(), 20),
                "failed route",
            )
            .unwrap();
        }
        let kernel: Kernel = serde_json::from_value(serde_json::json!({"name":"test0","symbol":"unused","module":identity,"source":identity,"host_abi":identity,"grid":[1,1,1],"block":[1,1,1],"shared_memory_bytes":0,"cooperative":false,"args":[]})).unwrap();
        assert!(
            pager
                .ensure(&s, &kernel, &[Value::Pointer(counts_address)])
                .is_err()
        );
        assert!(
            pager
                .ensure(&s, &kernel, &[Value::Pointer(counts_address)])
                .unwrap_err()
                .contains("quarantined")
        );
        s.cleanup().unwrap();
        s.cleanup().unwrap();
    }
}
