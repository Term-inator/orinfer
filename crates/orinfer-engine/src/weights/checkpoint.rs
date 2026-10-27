//! Read-only original checkpoint tensors, before physical execution transforms.
use super::{Index, Shard};
use crate::artifact::{Result, resolve_file};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use std::{
    collections::BTreeMap,
    fs::File,
    path::{Path, PathBuf},
};

pub(crate) struct Checkpoint {
    base: PathBuf,
    index: Index,
    shards: BTreeMap<String, Shard>,
}
pub(crate) struct Tensor<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}
impl Checkpoint {
    pub fn open(base: &Path, index: &str) -> Result<Self> {
        let base = base.canonicalize().map_err(|e| e.to_string())?;
        let index = crate::model::read(&resolve_file(&base, index)?)?;
        Ok(Self {
            base,
            index,
            shards: BTreeMap::new(),
        })
    }
    pub fn read(&mut self, name: &str) -> Result<Tensor<'_>> {
        let file = self
            .index
            .weight_map
            .get(name)
            .ok_or_else(|| format!("Checkpoint tensor missing: {name}"))?
            .clone();
        self.file_tensor(&file, name)
    }
    pub fn path(&self, name: &str) -> Result<PathBuf> {
        let file = self
            .index
            .weight_map
            .get(name)
            .ok_or_else(|| format!("Checkpoint tensor missing: {name}"))?;
        resolve_file(&self.base, file)
    }
    pub fn file_tensor(&mut self, file: &str, name: &str) -> Result<Tensor<'_>> {
        if !self.shards.contains_key(file) {
            let path = resolve_file(&self.base, file)?;
            let input = File::open(&path).map_err(|e| e.to_string())?;
            // SAFETY: Checkpoint assets are immutable throughout execution. This
            // owner retains each read-only mapping for every borrowed tensor.
            let mapping = unsafe { MmapOptions::new().map(&input) }.map_err(|e| e.to_string())?;
            let (header, metadata) =
                SafeTensors::read_metadata(&mapping).map_err(|e| e.to_string())?;
            self.shards.insert(
                file.to_owned(),
                Shard {
                    path,
                    mapping,
                    metadata,
                    data_start: 8 + header,
                },
            );
        }
        let shard = &self.shards[file];
        let info = shard
            .metadata
            .info(name)
            .ok_or_else(|| format!("{file}: missing {name}"))?;
        let (start, end) = info.data_offsets;
        Ok(Tensor {
            dtype: info.dtype,
            shape: &info.shape,
            data: &shard.mapping[shard.data_start + start..shard.data_start + end],
        })
    }
}
