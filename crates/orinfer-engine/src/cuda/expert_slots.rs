//! Packed compressed experts in a single session-owned allocation.
use super::{Session, check, experts::Plan, sys, weight_upload};
use crate::{artifact::Result, weights::TensorSource};
use orinfer_model_sdk::abi::ExpertBank;
use std::collections::BTreeMap;

type Key = (usize, usize);

pub(super) struct Slots {
    base: u64,
    pub bytes: usize,
    stride: usize,
    layouts: Vec<Vec<(usize, usize)>>,
    tables: Vec<Vec<u64>>,
    positions: BTreeMap<Key, usize>,
    free: Vec<usize>,
}

fn align(bytes: usize) -> Result<usize> {
    bytes
        .checked_add(255)
        .map(|n| n & !255)
        .ok_or("Expert slot alignment overflow".into())
}

pub(super) struct Layout {
    pub bytes: usize,
    stride: usize,
    layouts: Vec<Vec<(usize, usize)>>,
    count: usize,
}
impl Layout {
    pub fn new(
        banks: &[ExpertBank],
        capacity: usize,
        sizes: &BTreeMap<String, usize>,
    ) -> Result<Option<Self>> {
        if banks.iter().all(|b| b.addresses.is_empty()) {
            return Ok(None);
        }
        if banks.iter().any(|b| b.addresses.len() != b.buffers.len()) {
            return Err("Every compact expert bank needs complete address tables".into());
        }
        let mut stride = 0;
        let mut layouts = Vec::new();
        for bank in banks {
            if bank.experts == 0 {
                return Err("Empty expert bank".into());
            }
            let mut offset = 0usize;
            let mut layout = Vec::new();
            for name in &bank.buffers {
                let source_bytes = *sizes.get(name).ok_or("Missing expert tensor size")?;
                if source_bytes == 0 || !source_bytes.is_multiple_of(bank.experts) {
                    return Err("Expert tensor has a fractional stride".into());
                }
                let bytes = source_bytes / bank.experts;
                layout.push((offset, bytes));
                offset = align(
                    offset
                        .checked_add(bytes)
                        .ok_or("Expert slot size overflow")?,
                )?;
            }
            stride = stride.max(offset);
            layouts.push(layout);
        }
        // All experts wanted by one bank must stay alive through its down projection.
        // Above the resident budget, only that bounded current union is transient.
        let count = capacity.max(banks.iter().map(|b| b.experts).max().unwrap_or(0));
        let bytes = stride
            .checked_mul(count)
            .ok_or("Expert slot pool overflow")?;
        Ok(Some(Self {
            bytes,
            stride,
            layouts,
            count,
        }))
    }
}

impl Slots {
    pub fn new(
        session: &mut Session,
        plan: &Plan,
        pointers: &BTreeMap<String, u64>,
    ) -> Result<Option<Self>> {
        let Some(layout) = &plan.slot_layout else {
            return Ok(None);
        };
        let Layout {
            bytes,
            stride,
            layouts,
            count,
        } = layout;
        let mut base = 0;
        // SAFETY: Checked nonzero extent, immediately transferred to Session ownership.
        unsafe {
            check(
                sys::cuMemAlloc_v2(&mut base, *bytes),
                "allocate compact expert slots",
            )?;
        }
        session.buffers.push(base);
        let tables = plan
            .banks
            .iter()
            .map(|b| b.addresses.iter().map(|n| pointers[n]).collect())
            .collect();
        Ok(Some(Self {
            base,
            bytes: *bytes,
            stride: *stride,
            layouts: layouts.clone(),
            tables,
            positions: BTreeMap::new(),
            free: (0..*count).rev().collect(),
        }))
    }

    pub fn release(&mut self, key: Key) -> Result<()> {
        let slot = self
            .positions
            .remove(&key)
            .ok_or("Lost expert slot ownership")?;
        self.free.push(slot);
        Ok(())
    }

    pub fn load(
        &mut self,
        key: Key,
        bank: &ExpertBank,
        sources: &BTreeMap<String, TensorSource>,
        uploads: &mut Vec<weight_upload::Upload>,
    ) -> Result<usize> {
        if self.positions.contains_key(&key) {
            return Err("Expert already owns a compact slot".into());
        }
        let slot = self
            .free
            .pop()
            .ok_or("Current expert union exceeds the slot pool")?;
        self.positions.insert(key, slot);
        let mut total = 0;
        for (name, &(offset, bytes)) in bank.buffers.iter().zip(&self.layouts[key.0]) {
            let original = &sources[name];
            let source = TensorSource {
                path: original.path.clone(),
                offset: original.offset + key.1 * bytes,
                bytes,
            };
            uploads.push(weight_upload::Upload {
                source,
                destination: self.base + (slot * self.stride + offset) as u64,
            });
            total += bytes;
        }
        Ok(total)
    }

    pub fn publish(&self, session: &Session, bank: usize, experts: usize) -> Result<()> {
        for (column, &destination) in self.tables[bank].iter().enumerate() {
            let mut addresses = vec![0u64; experts];
            for (&(_, expert), &slot) in self.positions.range((bank, 0)..(bank, experts)) {
                addresses[expert] =
                    self.base + (slot * self.stride + self.layouts[bank][column].0) as u64;
            }
            upload_addresses(session, destination, &addresses)?;
        }
        Ok(())
    }
}

fn upload_addresses(session: &Session, destination: u64, addresses: &[u64]) -> Result<()> {
    // SAFETY: u64 has no padding; the bounded pinned uploader takes its own copy.
    let bytes =
        unsafe { std::slice::from_raw_parts(addresses.as_ptr().cast(), addresses.len() * 8) };
    session
        .transfers
        .borrow_mut()
        .upload(session.stream, destination, bytes)
}

pub(super) fn initialize_full(
    session: &Session,
    banks: &[ExpertBank],
    pointers: &BTreeMap<String, u64>,
    sizes: &BTreeMap<String, usize>,
) -> Result<()> {
    for bank in banks {
        for (name, table) in bank.buffers.iter().zip(&bank.addresses) {
            if bank.addresses.len() != bank.buffers.len() || sizes[table] != bank.experts * 8 {
                return Err("Invalid full expert address contract".into());
            }
            let stride = sizes[name] / bank.experts;
            let addresses: Vec<_> = (0..bank.experts)
                .map(|e| pointers[name] + (e * stride) as u64)
                .collect();
            upload_addresses(session, pointers[table], &addresses)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_budget_is_checked_before_cuda_allocation() {
        let bank = ExpertBank {
            experts: 5,
            buffers: vec!["packed".into(), "scale".into()],
            addresses: vec!["packed_addresses".into(), "scale_addresses".into()],
            consumers: BTreeMap::new(),
        };
        let mut sizes = BTreeMap::from([("packed".into(), 397 * 5), ("scale".into(), 17 * 5)]);
        // A full routed bank fits even when the persistent budget is only two.
        assert_eq!(
            Layout::new(std::slice::from_ref(&bank), 2, &sizes)
                .unwrap()
                .unwrap()
                .bytes,
            5 * 768
        );
        assert!(Layout::new(std::slice::from_ref(&bank), usize::MAX, &sizes).is_err());
        sizes.insert("packed".into(), 398 * 5 - 1);
        assert!(Layout::new(std::slice::from_ref(&bank), 2, &sizes).is_err());
        sizes.insert("packed".into(), 0);
        assert!(Layout::new(std::slice::from_ref(&bank), 2, &sizes).is_err());
        sizes.remove("packed");
        assert!(Layout::new(&[bank], 2, &sizes).is_err());
    }
}
