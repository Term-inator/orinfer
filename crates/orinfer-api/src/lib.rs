//! CPU-side OpenAI Chat protocol adapter. GPU execution belongs to orinfer-engine.
mod config;
pub mod embeddings;
pub mod manager;
mod server;
#[cfg(test)]
mod test_fixtures;
pub use config::{BatchConfig, BatchOrder, PreprocessingLimits, ServerConfig};
pub use server::batch::{BatchReport, run as run_batch};
pub use server::run;
