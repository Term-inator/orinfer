//! Common packet operand-table contract for ordinary and speculative rows.
use crate::{
    artifact::{Dtype, Result},
    model::Manifest,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PacketTable {
    pub buffer: String,
    pub columns: Vec<String>,
    pub kernels: BTreeMap<String, String>,
    /// Original host argument index -> wrapped host argument index.
    #[serde(default)]
    pub scalar_arguments: BTreeMap<String, Vec<(usize, usize)>>,
    pub copies: Vec<PacketCopy>,
}

impl PacketTable {
    pub fn validate(&self, model: &Manifest) -> Result<()> {
        let table = self;
        let columns: BTreeSet<_> = table.columns.iter().collect();
        let runtime_buffers: BTreeSet<_> = model
            .buffers
            .iter()
            .filter(|buffer| buffer.data.is_none())
            .map(|buffer| buffer.name.as_str())
            .collect();
        let kernels: BTreeSet<_> = model.kernels.iter().map(|k| k.name.as_str()).collect();
        if columns.is_empty()
            || columns.len() != table.columns.len()
            || !model.buffers.iter().any(|b| {
                b.name == table.buffer
                    && b.dtype == Dtype::U64
                    && b.data.is_none()
                    && b.shape == [128 * columns.len()]
            })
            || table
                .columns
                .iter()
                .any(|name| !runtime_buffers.contains(name.as_str()))
            || table.copies.iter().any(|copy| {
                copy.row_bytes == 0 || !model.dynamic_batch_kernels.contains_key(&copy.kernel)
            })
            || table
                .kernels
                .values()
                .chain(table.copies.iter().map(|c| &c.kernel))
                .any(|name| !kernels.contains(name.as_str()))
        {
            return Err("Invalid packet address table".into());
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PacketCopy {
    pub source: String,
    pub destination: String,
    pub kernel: String,
    pub row_bytes: usize,
}
