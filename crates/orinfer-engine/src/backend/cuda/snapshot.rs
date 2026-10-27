//! Shared immutable KV extents plus private endpoint state. Restore keeps graph VAs.
use super::executor::Executor;
use super::{Result, check, sys};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    time::Instant,
};

use crate::backend::snapshot::size;
pub(crate) use crate::backend::snapshot::{Allocation, Piece, Plan, Range, Snapshot, Workspace};
impl Executor {
    #[cfg(test)]
    pub(crate) fn snapshot_resident_bytes(&self) -> usize {
        self.snapshot_allocations.iter().map(|a| a.bytes).sum()
    }
    pub(crate) fn plan_snapshot(
        &self,
        ranges: BTreeMap<String, Range>,
        kv_names: &BTreeSet<String>,
        base: &[Piece],
    ) -> Result<Plan> {
        for piece in base {
            self.session.owner.require(piece.allocation.owner)?;
        }
        let (mut state_ranges, mut new_kv, mut kv) = (BTreeMap::new(), BTreeMap::new(), vec![]);
        for (name, range) in ranges {
            let end = range
                .offset
                .checked_add(range.bytes)
                .ok_or("Snapshot extent overflow")?;
            if self.sizes.get(&name).is_none_or(|&bytes| end > bytes)
                || self
                    .session
                    .virtual_buffers
                    .borrow()
                    .get(&name)
                    .is_some_and(|v| end > v.mapped)
            {
                return Err(format!("{name}: snapshot outside resident allocation"));
            }
            if !kv_names.contains(&name) {
                state_ranges.insert(name, range);
                continue;
            }
            if range.offset != 0 {
                return Err("KV snapshot must start at zero".into());
            }
            let reuse_end = reusable_prefix_end(
                base.iter().filter(|p| p.name == name).map(|p| &p.range),
                range.bytes,
                self.session
                    .virtual_buffers
                    .borrow()
                    .get(&name)
                    .map(|v| v.granularity),
            )?;
            let mut covered = 0;
            for p in base.iter().filter(|p| p.name == name) {
                if covered == reuse_end {
                    break;
                }
                if p.range.offset != covered {
                    return Err("Non-contiguous shared KV extent".into());
                }
                let mut p = p.clone();
                p.range.bytes = p.range.bytes.min(reuse_end - covered);
                covered += p.range.bytes;
                kv.push(p);
            }
            if covered < range.bytes {
                new_kv.insert(
                    name,
                    Range {
                        offset: covered,
                        bytes: range.bytes - covered,
                    },
                );
            }
        }
        let new_bytes = size(&state_ranges)?
            .checked_add(self.kv_snapshot_size(&new_kv)?)
            .ok_or("Snapshot size overflow")?;
        Ok(Plan {
            owner: self.session.owner,
            state: state_ranges,
            tail: new_kv,
            kv,
            new_bytes,
        })
    }
    fn kv_snapshot_size(&self, ranges: &BTreeMap<String, Range>) -> Result<usize> {
        let buffers = self.session.virtual_buffers.borrow();
        ranges.iter().try_fold(0usize, |sum, (name, range)| {
            let parts = buffers
                .get(name)
                .and_then(|v| aligned_interior(range, v.granularity));
            let bytes = if let Some((offset, bytes)) = parts {
                bytes
                    .checked_add(aligned_size(offset - range.offset)?)
                    .and_then(|n| {
                        n.checked_add(
                            aligned_size(range.offset + range.bytes - offset - bytes).ok()?,
                        )
                    })
                    .ok_or("KV snapshot size overflow")?
            } else {
                aligned_size(range.bytes)?
            };
            sum.checked_add(bytes)
                .ok_or("KV snapshot size overflow".into())
        })
    }
    pub(crate) fn snapshot(&mut self, plan: &Plan) -> Result<Option<Snapshot>> {
        self.session.owner.require(plan.owner)?;
        self.sync()?;
        self.collect_snapshots()?;
        let Some(state) = self.save_ranges(plan.state.clone())? else {
            return Ok(None);
        };
        let Some(tail) = self.save_kv_ranges(&plan.tail)? else {
            drop(state);
            self.collect_snapshots()?;
            return Ok(None);
        };
        let mut kv = plan.kv.clone();
        kv.extend(tail);
        let bytes = state.iter().chain(&kv).map(|p| p.range.bytes).sum();
        self.sync()?;
        Ok(Some(Snapshot {
            owner: self.session.owner,
            bytes,
            state,
            kv,
        }))
    }
    fn save_ranges(&mut self, ranges: BTreeMap<String, Range>) -> Result<Option<Vec<Piece>>> {
        self.allocate_ranges(ranges, true)
    }
    fn allocate_ranges(
        &mut self,
        ranges: BTreeMap<String, Range>,
        copy: bool,
    ) -> Result<Option<Vec<Piece>>> {
        let bytes = size(&ranges)?;
        if bytes == 0 {
            return Ok(Some(vec![]));
        }
        let (mut free, mut total, mut address) = (0, 0, 0);
        // SAFETY: Live context, checked extents. Ownership transfers to Session
        // before any fallible copy; its destructor also handles error paths.
        unsafe {
            check(
                sys::cuMemGetInfo_v2(&mut free, &mut total),
                "snapshot memory",
            )?;
            if bytes > free.saturating_sub(64 * 1024 * 1024) {
                return Ok(None);
            }
            let status = sys::cuMemAlloc_v2(&mut address, bytes);
            if status == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
                return Ok(None);
            }
            check(status, "allocate prefix snapshot")?;
        }
        self.session.buffers.push(address);
        let allocation = Rc::new(Allocation {
            owner: self.session.owner,
            address,
            bytes,
            handle: None,
        });
        self.snapshot_allocations.push(Rc::clone(&allocation));
        let (mut pieces, mut source) = (vec![], 0);
        for (name, range) in ranges {
            if range.bytes != 0 {
                if copy {
                    // SAFETY: Source/destination extents are validated; one stream
                    // orders all writes before the immutable snapshot is published.
                    unsafe {
                        check(
                            sys::cuMemcpyDtoDAsync_v2(
                                address + source as u64,
                                self.pointers[&name] + range.offset as u64,
                                range.bytes,
                                self.session.stream,
                            ),
                            "save prefix extent",
                        )?;
                    }
                }
                let count = range.bytes;
                pieces.push(Piece {
                    allocation: Rc::clone(&allocation),
                    name,
                    range,
                    source,
                });
                source = (source + count + 255) & !255;
            }
        }
        Ok(Some(pieces))
    }
    pub(crate) fn reserve_workspace(
        &mut self,
        ranges: BTreeMap<String, Range>,
    ) -> Result<Option<Rc<RefCell<Workspace>>>> {
        let plan = self.plan_snapshot(ranges, &BTreeSet::new(), &[])?;
        self.collect_snapshots()?;
        let Some(state) = self.allocate_ranges(plan.state, false)? else {
            return Ok(None);
        };
        let bytes = state.iter().map(|p| p.range.bytes).sum();
        Ok(Some(Rc::new(RefCell::new(Workspace {
            snapshot: Snapshot {
                owner: self.session.owner,
                bytes,
                state,
                kv: vec![],
            },
            resident: true,
            saved_s: 0.,
        }))))
    }
    pub(crate) fn retain_workspace(&mut self, workspace: &Rc<RefCell<Workspace>>) -> Result<()> {
        workspace
            .borrow()
            .snapshot
            .require_owner(self.session.owner)?;
        if self
            .live_workspace
            .as_ref()
            .and_then(|(_, w)| w.upgrade())
            .is_some()
        {
            return Err("Shared workspace already has a live continuation".into());
        }
        self.live_workspace = Some((self.active_sequence, Rc::downgrade(workspace)));
        Ok(())
    }
    pub(super) fn save_live_workspace(&mut self) -> Result<()> {
        let Some((owner, weak)) = &self.live_workspace else {
            return Ok(());
        };
        let Some(workspace) = weak.upgrade() else {
            self.live_workspace = None;
            return Ok(());
        };
        if *owner != self.active_sequence {
            return Err("Live workspace belongs to an inactive sequence".into());
        }
        let at = Instant::now();
        let mut workspace = workspace.borrow_mut();
        if !workspace.resident {
            return Err("Live workspace has already been saved".into());
        }
        for p in &workspace.snapshot.state {
            // SAFETY: The backing and source extents were validated at reserve;
            // they are private, live, and ordered with preceding layer writes.
            unsafe {
                check(
                    sys::cuMemcpyDtoDAsync_v2(
                        p.allocation.address + p.source as u64,
                        self.pointers[&p.name] + p.range.offset as u64,
                        p.range.bytes,
                        self.session.stream,
                    ),
                    "save live workspace",
                )?;
            }
        }
        self.sync()?;
        workspace.resident = false;
        let seconds = at.elapsed().as_secs_f64();
        workspace.saved_s += seconds;
        self.workspace_save_s += seconds;
        self.live_workspace = None;
        Ok(())
    }
    pub(crate) fn resume_workspace(&mut self, workspace: &Rc<RefCell<Workspace>>) -> Result<f64> {
        workspace
            .borrow()
            .snapshot
            .require_owner(self.session.owner)?;
        if let Some((_, live)) = &self.live_workspace
            && live
                .upgrade()
                .is_some_and(|live| Rc::ptr_eq(&live, workspace))
        {
            self.live_workspace = None;
        } else {
            self.save_live_workspace()?;
        }
        let mut workspace = workspace.borrow_mut();
        let at = Instant::now();
        let seconds = if workspace.resident {
            0.
        } else {
            self.restore_snapshot(&workspace.snapshot)?;
            workspace.resident = true;
            let seconds = at.elapsed().as_secs_f64();
            self.workspace_restore_s += seconds;
            seconds
        };
        Ok(seconds + std::mem::take(&mut workspace.saved_s))
    }
    fn save_kv_ranges(&mut self, ranges: &BTreeMap<String, Range>) -> Result<Option<Vec<Piece>>> {
        let (mut pieces, mut copied) = (vec![], BTreeMap::new());
        for (name, range) in ranges {
            let granularity = self
                .session
                .virtual_buffers
                .borrow()
                .get(name)
                .map(|v| v.granularity);
            let Some((offset, bytes)) = granularity.and_then(|g| aligned_interior(range, g)) else {
                copied.insert(name.clone(), range.clone());
                continue;
            };
            let (mut free, mut total) = (0, 0);
            // SAFETY: Live context and writable scalar outputs.
            unsafe {
                check(
                    sys::cuMemGetInfo_v2(&mut free, &mut total),
                    "shared prefix memory",
                )?;
            }
            if bytes > free.saturating_sub(64 << 20) {
                return Ok(None);
            }
            let reservation = super::virtual_memory::Reservation::reserve(&self.session, bytes, 1)?;
            let address = reservation.address;
            let key = address.to_string();
            // Register before any fallible physical mapping or copy; cleanup owns all transitions.
            self.session
                .prefix_buffers
                .borrow_mut()
                .insert(key.clone(), reservation);
            let mapped = self
                .session
                .prefix_buffers
                .borrow_mut()
                .get_mut(&key)
                .unwrap()
                .grow_bytes(&self.session, bytes);
            if let Err(error) = mapped {
                let mut buffers = self.session.prefix_buffers.borrow_mut();
                let v = buffers.get_mut(&key).unwrap();
                v.release_slabs(&self.session.driver)?;
                // SAFETY: No snapshot refers to this failed allocation; its mappings are retired.
                unsafe {
                    check(
                        sys::cuMemAddressFree(v.address, v.bytes),
                        "rollback shared prefix address",
                    )?;
                }
                buffers.remove(&key);
                if error.contains("CUDA_ERROR_OUT_OF_MEMORY") {
                    return Ok(None);
                }
                return Err(error);
            }
            let handle = self.session.prefix_buffers.borrow()[&key].allocation_handle()?;
            let allocation = Rc::new(Allocation {
                owner: self.session.owner,
                address,
                bytes,
                handle: Some(handle),
            });
            self.snapshot_allocations.push(Rc::clone(&allocation));
            // SAFETY: Both complete aligned extents are resident and owned on this stream.
            unsafe {
                check(
                    sys::cuMemcpyDtoDAsync_v2(
                        address,
                        self.pointers[name] + offset as u64,
                        bytes,
                        self.session.stream,
                    ),
                    "save immutable VMM prefix",
                )?;
            }
            pieces.push(Piece {
                allocation,
                name: name.clone(),
                range: Range { offset, bytes },
                source: 0,
            });
            // Boundary fragments cannot alias a whole VMM allocation. Keep their exact bytes private.
            for (start, end) in [
                (range.offset, offset),
                (offset + bytes, range.offset + range.bytes),
            ] {
                if start < end {
                    let Some(part) = self.save_ranges(BTreeMap::from([(
                        name.clone(),
                        Range {
                            offset: start,
                            bytes: end - start,
                        },
                    )]))?
                    else {
                        return Ok(None);
                    };
                    pieces.extend(part);
                }
            }
        }
        let Some(tail) = self.save_ranges(copied)? else {
            return Ok(None);
        };
        pieces.extend(tail);
        pieces.sort_by(|a, b| (&a.name, a.range.offset).cmp(&(&b.name, b.range.offset)));
        Ok(Some(pieces))
    }
    pub(crate) fn restore_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        snapshot.require_owner(self.session.owner)?;
        self.sync()?;
        // Map complete immutable KV allocations at fixed request VAs. Partial
        // allocation slices and endpoint state retain the byte-exact copy path.
        let mut shared = BTreeMap::<&str, Vec<&Piece>>::new();
        for p in &snapshot.kv {
            shared.entry(&p.name).or_default().push(p);
        }
        let mut aliased = BTreeSet::<(&str, usize)>::new();
        for (name, pieces) in &mut shared {
            pieces.sort_by_key(|p| p.range.offset);
            let mut buffers = self.session.virtual_buffers.borrow_mut();
            let Some(v) = buffers.get_mut(*name) else {
                continue;
            };
            if !pieces.iter().any(|p| can_alias(p, v.granularity)) {
                continue;
            }
            v.release_slabs(&self.session.driver)?;
            for p in pieces {
                if can_alias(p, v.granularity) {
                    v.grow_bytes(&self.session, p.range.offset)?;
                    v.map_shared(&self.session, p.range.offset, Rc::clone(&p.allocation))?;
                    aliased.insert((name, p.range.offset));
                } else {
                    v.grow_bytes(&self.session, p.range.offset + p.range.bytes)?;
                }
            }
        }
        // Grow once to each final extent, then restore private fragments/state.
        let mut extents: BTreeMap<&str, usize> = BTreeMap::new();
        for p in snapshot.kv.iter().chain(&snapshot.state) {
            if aliased.contains(&(p.name.as_str(), p.range.offset)) {
                continue;
            }
            if let Some(v) = self.session.virtual_buffers.borrow_mut().get_mut(&p.name) {
                v.make_writable(&self.session, p.range.offset, p.range.bytes)?;
            }
            let end = p
                .range
                .offset
                .checked_add(p.range.bytes)
                .ok_or("Restore extent overflow")?;
            extents
                .entry(&p.name)
                .and_modify(|e| *e = (*e).max(end))
                .or_insert(end);
        }
        for (name, count) in extents {
            if let Some(v) = self.session.virtual_buffers.borrow_mut().get_mut(name) {
                v.grow(&self.session, count.div_ceil(v.stride))?;
            }
        }
        for p in snapshot.kv.iter().chain(&snapshot.state) {
            if aliased.contains(&(p.name.as_str(), p.range.offset)) {
                continue;
            }
            // SAFETY: Immutable allocations remain referenced; ranges were checked
            // at capture, and graph-bound destinations retain their original VAs.
            unsafe {
                check(
                    sys::cuMemcpyDtoDAsync_v2(
                        self.pointers[&p.name] + p.range.offset as u64,
                        p.allocation.address + p.source as u64,
                        p.range.bytes,
                        self.session.stream,
                    ),
                    "restore prefix extent",
                )?;
            }
        }
        self.sync()
    }
    pub(crate) fn release_snapshot(&mut self, snapshot: Snapshot) -> Result<()> {
        drop(snapshot);
        self.collect_snapshots()
    }
    pub(crate) fn collect_snapshots(&mut self) -> Result<()> {
        if self
            .snapshot_allocations
            .iter()
            .all(|a| Rc::strong_count(a) > 1)
        {
            return Ok(());
        }
        self.sync()?;
        let mut i = 0;
        while i < self.snapshot_allocations.len() {
            let a = &self.snapshot_allocations[i];
            if Rc::strong_count(a) != 1 {
                i += 1;
                continue;
            }
            if a.handle.is_some() {
                let key = a.address.to_string();
                let mut buffers = self.session.prefix_buffers.borrow_mut();
                let v = buffers
                    .get_mut(&key)
                    .ok_or("Unowned shared prefix allocation")?;
                v.release_slabs(&self.session.driver)?;
                // SAFETY: Registry is the only owner, mappings and consumers are retired.
                unsafe {
                    check(
                        sys::cuMemAddressFree(v.address, v.bytes),
                        "release shared prefix address",
                    )?;
                }
                buffers.remove(&key);
                self.snapshot_allocations.swap_remove(i);
                continue;
            }
            let index = self
                .session
                .buffers
                .iter()
                .position(|&p| p == a.address)
                .ok_or("Unowned snapshot allocation")?;
            // SAFETY: Stream completed and registry is the only remaining owner.
            unsafe {
                check(sys::cuMemFree_v2(a.address), "release prefix allocation")?;
            }
            self.session.buffers.swap_remove(index);
            self.snapshot_allocations.swap_remove(i);
        }
        Ok(())
    }
}
/// Recopy an incomplete physical granule with the new tail. Otherwise small
/// periodic checkpoints would forever retain fragments too small to alias.
fn reusable_prefix_end<'a>(
    ranges: impl Iterator<Item = &'a Range>,
    bytes: usize,
    granularity: Option<usize>,
) -> Result<usize> {
    let mut end = 0usize;
    for range in ranges {
        if range.offset != end {
            return Err("Non-contiguous shared KV extent".into());
        }
        end = end
            .checked_add(range.bytes)
            .ok_or("Shared KV extent overflow")?;
        if end >= bytes {
            break;
        }
    }
    let end = end.min(bytes);
    Ok(granularity.map_or(end, |g| end / g * g))
}
fn aligned_interior(range: &Range, granularity: usize) -> Option<(usize, usize)> {
    let start = range.offset.checked_add(granularity.checked_sub(1)?)? / granularity * granularity;
    let end = range.offset.checked_add(range.bytes)? / granularity * granularity;
    (end > start).then_some((start, end.saturating_sub(start)))
}
fn aligned_size(bytes: usize) -> Result<usize> {
    bytes
        .checked_add(255)
        .map(|n| n & !255)
        .ok_or("Snapshot size overflow".into())
}
fn can_alias(p: &Piece, granularity: usize) -> bool {
    p.allocation.handle.is_some()
        && p.source == 0
        && p.range.bytes == p.allocation.bytes
        && p.range.offset.is_multiple_of(granularity)
        && p.range.bytes.is_multiple_of(granularity)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn periodic_checkpoints_coalesce_incomplete_physical_granules() {
        let ranges = [
            Range {
                offset: 0,
                bytes: 1024,
            },
            Range {
                offset: 1024,
                bytes: 3072,
            },
            Range {
                offset: 4096,
                bytes: 1024,
            },
        ];
        assert_eq!(
            reusable_prefix_end(ranges.iter(), 8192, Some(4096)).unwrap(),
            4096
        );
        assert_eq!(
            reusable_prefix_end(ranges[..1].iter(), 8192, Some(4096)).unwrap(),
            0
        );
        assert_eq!(
            reusable_prefix_end(ranges.iter(), 3072, Some(4096)).unwrap(),
            0
        );
        assert_eq!(
            reusable_prefix_end(ranges.iter(), 8192, None).unwrap(),
            5120
        );
        assert!(reusable_prefix_end(ranges[1..].iter(), 8192, Some(4096)).is_err());
    }
    #[test]
    fn shared_prefix_only_maps_complete_aligned_allocations() {
        assert_eq!(
            aligned_interior(
                &Range {
                    offset: 1,
                    bytes: 8191
                },
                4096
            ),
            Some((4096, 4096))
        );
        assert_eq!(
            aligned_interior(
                &Range {
                    offset: 5,
                    bytes: 4090
                },
                4096
            ),
            None
        );
        assert_eq!(
            aligned_interior(
                &Range {
                    offset: usize::MAX,
                    bytes: 1
                },
                4096
            ),
            None
        );
        let mut piece = Piece {
            allocation: Rc::new(Allocation {
                owner: orinfer_device::DeviceOwner::new(orinfer_device::Backend::Cuda, 0),
                address: 0,
                bytes: 8192,
                handle: Some(7),
            }),
            name: "kv".into(),
            range: Range {
                offset: 4096,
                bytes: 8192,
            },
            source: 0,
        };
        assert!(can_alias(&piece, 4096));
        piece.range.bytes = 4096;
        assert!(!can_alias(&piece, 4096));
        piece.range.bytes = 8192;
        piece.source = 4096;
        assert!(!can_alias(&piece, 4096));
    }
    #[test]
    fn range_accounting_checks_overflow_and_alignment() {
        let mut r = BTreeMap::new();
        r.insert(
            "a".into(),
            Range {
                offset: 100,
                bytes: 1,
            },
        );
        r.insert(
            "b".into(),
            Range {
                offset: 0,
                bytes: 257,
            },
        );
        assert_eq!(size(&r).unwrap(), 768);
        r.insert(
            "c".into(),
            Range {
                offset: 0,
                bytes: usize::MAX,
            },
        );
        assert!(size(&r).is_err());
    }
}
