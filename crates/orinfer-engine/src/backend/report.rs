//! Backend-neutral load and residency statistics. Legacy report field names stay stable.
use super::DeviceInfo;
use crate::execution::GraphMode;
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
    pub(crate) cuda_graph: GraphMode,
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
    pub ensure_calls: u64,
    /// Completed prior GPU work plus the route histogram download.
    pub route_wait_s: f64,
    /// Cache bookkeeping and physical page mapping/unmapping, excluding upload.
    pub transition_s: f64,
    pub upload_s: f64,
    pub upload_staging_s: f64,
    pub read_wait_s: f64,
    pub dma_wait_s: f64,
    pub upload_chunks: usize,
    /// Sum of reader work/wait intervals across concurrent workers.
    pub reader_service_s: f64,
    pub file_read_s: f64,
    pub page_reuses: u64,
    pub host_cache: crate::execution::HostWeightCacheStatistics,
    /// Physical VMM pages include allocation-granule overhead and shared neighbors.
    pub mapped_bytes: usize,
    pub peak_mapped_bytes: usize,
}
