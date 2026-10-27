//! Backend-neutral load and residency statistics. Legacy report field names stay stable.
use super::DeviceInfo;
use crate::execution::GraphMode as CudaGraphMode;
use serde::Serialize;
pub(crate) struct LoadStats {
    pub(crate) resident_experts: crate::execution::ResidentExperts,
    pub(crate) manifest_sha256: String,
    pub(crate) device: DeviceInfo,
    pub(crate) load_to_ready_s: f64,
    pub(crate) load_stages: crate::model::LoadStages,
    pub(crate) verify_weights: bool,
    pub(crate) load_workers: usize,
    pub(crate) preloaded_modules: usize,
    pub(crate) registered_modules: usize,
    pub(crate) weight_io_hash_s: f64,
    pub(crate) weight_upload_s: f64,
    pub(crate) module_load_bind_s: f64,
    pub(crate) graph_capture_s: f64,
    pub(crate) cuda_graph: CudaGraphMode,
    pub(crate) captured_programs: Vec<String>,
    pub(crate) buffer_bytes: usize,
    pub(crate) buffer_capacity_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ExpertCacheStatistics {
    pub capacity_experts: usize,
    pub resident_experts: usize,
    pub transient_experts: usize,
    pub peak_transient_experts: usize,
    pub lookups: u64,
    pub hits: u64,
    pub evictions: u64,
    pub loaded_bytes: u64,
    /// Physical VMM pages include allocation-granule overhead and shared neighbors.
    pub mapped_bytes: usize,
    pub peak_mapped_bytes: usize,
}
