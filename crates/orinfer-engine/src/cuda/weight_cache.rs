//! Bounded dense-weight residency over stable VMM addresses.
//!
//! Small assets and a fixed hot set stay resident. Cold tensors are read through
//! the existing bounded disk/DMA pipeline before each consumer. No weight values
//! or layouts change, and CUDA graphs are disabled while mappings can change.
use super::{Session, check, sys, virtual_memory::Reservation, weight_upload};
use crate::{
    artifact::{Argument, Kernel, Result},
    model::Manifest,
    weights::TensorSource,
};
use std::collections::{BTreeMap, BTreeSet};

const MIN_TENSOR: usize = 8 << 20;
pub(super) const REQUEST_RESERVE: usize = 512 << 20;

pub(super) struct Plan {
    pub cold: BTreeSet<String>,
    pub cold_bytes: usize,
    pub capacity: usize,
}

fn names(kernel: &Kernel) -> BTreeSet<&str> {
    kernel
        .args
        .iter()
        .filter_map(|a| match a {
            Argument::Buffer { name } | Argument::BufferSlice { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect()
}

impl Plan {
    pub fn new(
        manifest: &Manifest,
        used: &BTreeSet<&str>,
        budget: usize,
        granularity: usize,
    ) -> Result<Option<Self>> {
        let weights: Vec<_> = manifest
            .buffers
            .iter()
            .filter(|b| {
                b.data.is_some()
                    && b.access == crate::artifact::Access::Read
                    && used.contains(b.name.as_str())
            })
            .map(|b| Ok((b.name.as_str(), b.bytes()?)))
            .collect::<Result<_>>()?;
        let total = weights.iter().try_fold(0usize, |n, (_, bytes)| {
            n.checked_add(*bytes).ok_or("Weight sum overflow")
        })?;
        if total <= budget {
            return Ok(None);
        }
        let copied: BTreeSet<_> = manifest
            .programs
            .values()
            .flatten()
            .filter_map(|op| match op {
                crate::model::Operation::Copy { source, .. } => Some(source.as_str()),
                _ => None,
            })
            .collect();
        let candidates: BTreeMap<_, _> = weights
            .iter()
            .filter(|(name, bytes)| *bytes >= MIN_TENSOR && !copied.contains(name))
            .copied()
            .collect();
        let round = |bytes: usize| -> Result<usize> {
            bytes
                .checked_add(granularity - 1)
                .map(|n| n / granularity * granularity)
                .ok_or("Weight page sum overflow".into())
        };
        let capacity = manifest
            .kernels
            .iter()
            .map(|k| {
                names(k)
                    .iter()
                    .filter_map(|n| candidates.get(n))
                    .try_fold(0usize, |sum, &bytes| {
                        sum.checked_add(round(bytes)?)
                            .ok_or("Weight working set overflow".into())
                    })
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let mut pinned = weights
            .iter()
            .filter(|(name, _)| !candidates.contains_key(name))
            .try_fold(0usize, |sum, (_, bytes)| {
                sum.checked_add(*bytes).ok_or("Pinned weight sum overflow")
            })?;
        let hot_budget = budget
            .checked_sub(capacity)
            .ok_or("Device memory cannot hold one weight working set")?;
        if pinned > hot_budget {
            return Err(
                "Device memory cannot hold mandatory weights and one weight working set".into(),
            );
        }
        let mut cold = BTreeSet::new();
        let mut cold_bytes = 0usize;
        for (name, bytes) in weights {
            if !candidates.contains_key(name) {
                continue;
            }
            if bytes <= hot_budget - pinned {
                pinned += bytes;
            } else {
                cold.insert(name.into());
                cold_bytes += bytes;
            }
        }
        Ok(Some(Self {
            cold,
            cold_bytes,
            capacity,
        }))
    }
}

pub(super) struct Pager {
    pub capacity: usize,
    pub mapped: usize,
    sources: BTreeMap<String, TensorSource>,
    workers: usize,
    poisoned: bool,
}
impl Pager {
    pub fn new(capacity: usize, sources: BTreeMap<String, TensorSource>, workers: usize) -> Self {
        Self {
            capacity,
            mapped: 0,
            sources,
            workers,
            poisoned: false,
        }
    }
    pub fn ensure(&mut self, session: &Session, kernel: &Kernel) -> Result<()> {
        if self.poisoned {
            return Err("Cold weight cache is unusable after a failed load".into());
        }
        let result = self.ensure_inner(session, kernel);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn ensure_inner(&mut self, session: &Session, kernel: &Kernel) -> Result<()> {
        let required: BTreeSet<_> = names(kernel)
            .into_iter()
            .filter(|n| self.sources.contains_key(*n))
            .collect();
        let missing = {
            let reservations = session.weight_buffers.borrow();
            required.iter().any(|n| reservations[*n].mapped == 0)
        };
        if !missing {
            return Ok(());
        }
        // SAFETY: All streams belong to this worker's retained context. Finish
        // consumers before unmapping any tensor, including branch-stream work.
        unsafe {
            check(sys::cuCtxSynchronize(), "finish cold weight consumers")?;
        }
        let mut buffers = session.weight_buffers.borrow_mut();
        for (name, buffer) in buffers.iter_mut() {
            if buffer.mapped > 0 && !required.contains(name.as_str()) {
                let bytes = buffer.mapped;
                buffer.release_slabs(&session.driver)?;
                self.mapped -= bytes;
            }
        }
        let needed = required.iter().try_fold(0usize, |n, name| {
            n.checked_add(buffers[*name].bytes)
                .ok_or("Cold weight working set overflow")
        })?;
        if needed > self.capacity {
            return Err("Kernel exceeds reserved cold-weight capacity".into());
        }
        let mut uploads = Vec::new();
        for name in required {
            let buffer = buffers.get_mut(name).unwrap();
            if buffer.mapped > 0 {
                continue;
            }
            buffer.grow_bytes(session, self.sources[name].bytes)?;
            self.mapped += buffer.mapped;
            uploads.push(weight_upload::Upload {
                source: self.sources[name].clone(),
                destination: buffer.address,
            });
        }
        drop(buffers);
        // Dense consumers may run on either stream. Finish the main-stream
        // upload before handing this bank to those consumers.
        weight_upload::upload(session, uploads, self.workers)?;
        // SAFETY: The session owns this stream and all pending upload slots.
        unsafe {
            check(
                sys::cuStreamSynchronize(session.stream),
                "dense weight upload",
            )?;
        }
        Ok(())
    }
}

pub(super) fn granularity(session: &Session) -> Result<usize> {
    let reservation = Reservation::reserve(session, 1, 1)?;
    let result = reservation.granularity;
    // SAFETY: This empty reservation has no mappings or consumers.
    unsafe {
        check(
            sys::cuMemAddressFree(reservation.address, reservation.bytes),
            "release weight granularity probe",
        )?;
    }
    Ok(result)
}

impl super::executor::Executor {
    pub(super) fn ensure_weights(&self, kernel: &Kernel) -> Result<()> {
        if let Some(pager) = self.weight_cache.borrow_mut().as_mut() {
            pager.ensure(&self.session, kernel)?;
        }
        Ok(())
    }
    pub(super) fn cold_weight_reserve(&self) -> usize {
        self.weight_cache
            .borrow()
            .as_ref()
            .map_or(0, |p| p.capacity.saturating_sub(p.mapped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest() -> Manifest {
        let id = serde_json::json!({"file":"unused","sha256":"0".repeat(64)});
        let buffers: Vec<_> = ["A", "B", "C"].iter().map(|name| serde_json::json!({"name":name,"dtype":"u8","shape":[MIN_TENSOR],"layout":"contiguous","alignment":256,"access":"read","data":{"tensor":name,"sha256":"0".repeat(64)}})).collect();
        let kernels: Vec<_> = ["A", "B", "C"].iter().map(|name| serde_json::json!({"name":name,"module":id,"source":id,"host_abi":id,"symbol":"unused","grid":[1,1,1],"block":[32,1,1],"shared_memory_bytes":0,"cooperative":false,"args":[{"kind":"buffer","name":name}]})).collect();
        serde_json::from_value(serde_json::json!({"schema_version":2,"target":"sm_120","model":"cache-test","chunk_tokens":128,"max_context":4096,"vocab":4,"toolchain":{},"buffers":buffers,"kernels":kernels,"programs":{},"reset_buffers":[],"input":"Input","token":"Token","status":"Status","logits":"Logits","position":"Step","weight_bytes":3*MIN_TENSOR,"weight_parameters":3*MIN_TENSOR,"weight_scope":"test"})).unwrap()
    }
    #[test]
    fn residency_accounts_for_simultaneous_consumers_and_rejects_too_small_budgets() {
        let mut m = manifest();
        let used = BTreeSet::from(["A", "B", "C"]);
        assert!(
            Plan::new(&m, &used, 3 * MIN_TENSOR, 65536)
                .unwrap()
                .is_none()
        );
        let plan = Plan::new(&m, &used, 20 << 20, 65536).unwrap().unwrap();
        assert_eq!(plan.cold, BTreeSet::from(["B".into(), "C".into()]));
        assert_eq!(plan.capacity, MIN_TENSOR);
        m.kernels[0].args.push(Argument::BufferSlice {
            name: "B".into(),
            offset: 0,
        });
        assert!(Plan::new(&m, &used, 15 << 20, 65536).is_err());
        let plan = Plan::new(&m, &used, 20 << 20, 65536).unwrap().unwrap();
        assert_eq!(plan.capacity, 2 * MIN_TENSOR);
        assert_eq!(plan.cold.len(), 3);
    }

    #[test]
    #[ignore = "Requires the shared GPU experiment lock"]
    fn cold_weights_reload_without_changing_addresses_and_poison_failed_uploads() {
        let mut session = Session::new(super::super::Driver::load().unwrap());
        // SAFETY: This test owns the context and records each CUDA handle in Session.
        unsafe {
            check(sys::cuInit(0), "init").unwrap();
            check(sys::cuDeviceGet(&mut session.device, 0), "device").unwrap();
            check(
                sys::cuCtxGetCurrent(&mut session.previous),
                "previous context",
            )
            .unwrap();
            let mut context = std::ptr::null_mut();
            check(
                sys::cuDevicePrimaryCtxRetain(&mut context, session.device),
                "retain",
            )
            .unwrap();
            session.retained = true;
            check(sys::cuCtxSetCurrent(context), "current").unwrap();
            check(sys::cuStreamCreate(&mut session.stream, 1), "stream").unwrap();
        }
        let m = manifest();
        let directory = tempfile::tempdir().unwrap();
        let mut sources = BTreeMap::new();
        let mut addresses = BTreeMap::new();
        for (name, byte) in [("B", 37u8), ("C", 91u8)] {
            let path = directory.path().join(name);
            std::fs::write(&path, vec![byte; MIN_TENSOR]).unwrap();
            let buffer = Reservation::reserve(&session, MIN_TENSOR, 1).unwrap();
            addresses.insert(name.to_string(), buffer.address);
            session
                .weight_buffers
                .borrow_mut()
                .insert(name.into(), buffer);
            sources.insert(
                name.into(),
                TensorSource {
                    path,
                    offset: 0,
                    bytes: MIN_TENSOR,
                },
            );
        }
        let capacity = session.weight_buffers.borrow()["B"].bytes;
        let mut pager = Pager::new(capacity, sources, 2);
        for (index, name, byte) in [(1, "B", 37u8), (2, "C", 91), (1, "B", 37)] {
            pager.ensure(&session, &m.kernels[index]).unwrap();
            assert_eq!(
                session.weight_buffers.borrow()[name].address,
                addresses[name]
            );
            assert_eq!(pager.mapped, capacity);
            let mut output = vec![0u8; MIN_TENSOR];
            // SAFETY: Live fully mapped tensor and writable output with exact extent.
            unsafe {
                check(
                    sys::cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), addresses[name], output.len()),
                    "read reloaded weight",
                )
                .unwrap();
            }
            assert!(output.iter().all(|&value| value == byte));
        }
        std::fs::remove_file(directory.path().join("C")).unwrap();
        assert!(pager.ensure(&session, &m.kernels[2]).is_err());
        assert!(pager.poisoned);
        assert!(pager.ensure(&session, &m.kernels[1]).is_err());
    }
}
