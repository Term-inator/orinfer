//! Bounded checks against real immutable Engram assets, without NPU allocation.
use super::{Int8Rows, RowPlan, Shard};
use crate::{
    artifact::{Result, sha256},
    model,
};
use orinfer_model_sdk::conditional_memory::{HashLayout, Hasher};
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
pub struct Report {
    pub history: Vec<u32>,
    pub tokens: Vec<u32>,
    pub hashes_sha256: String,
    pub chunked_history_verified: bool,
    pub layers: Vec<Layer>,
    pub scope: &'static str,
}
#[derive(Serialize)]
pub struct Layer {
    layer: usize,
    queries: usize,
    unique_rows: usize,
    row_bytes: usize,
    staged_bytes: usize,
    rank_rows: Vec<usize>,
}
pub fn run(metadata: &Path, model_dir: &Path) -> Result<Report> {
    if std::fs::metadata(metadata)
        .map_err(|e| e.to_string())?
        .len()
        > 16 << 20
    {
        return Err("Engram metadata exceeds 16 MiB".into());
    }
    let layout: HashLayout = model::read(metadata)?;
    for (file, digest) in [
        ("config.json", &layout.config_sha256),
        ("tokenizer.json", &layout.tokenizer_sha256),
    ] {
        let raw = std::fs::read(model_dir.join(file)).map_err(|e| e.to_string())?;
        if sha256(&raw) != *digest {
            return Err(format!("Engram metadata source identity differs: {file}"));
        }
    }
    let hasher = Hasher::new(layout)?;
    let history = vec![10, 20, 30];
    let mut tokens = vec![40, 50, 60, 70, 40, 50, 60, 70];
    tokens.extend(&hasher.layout().barriers);
    tokens.extend([80, 90, 100, 40, 50, 60, 70]);
    let (mut hashes, mut valid) = (vec![], vec![]);
    hasher.hash_into(&history, &tokens, &mut hashes, &mut valid)?;
    for split in 1..tokens.len() {
        let (mut first, mut fm, mut second, mut sm) = (vec![], vec![], vec![], vec![]);
        hasher.hash_into(&history, &tokens[..split], &mut first, &mut fm)?;
        let mut prefix = history.clone();
        prefix.extend(&tokens[..split]);
        hasher.hash_into(&prefix, &tokens[split..], &mut second, &mut sm)?;
        first.extend(second);
        fm.extend(sm);
        if first != hashes || fm != valid {
            return Err("Engram chunk/prefix hash mismatch".into());
        }
    }
    let digest = sha256(
        &hashes
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let width = hasher.layout().heads * 3;
    let mut layers = vec![];
    for (index, layer) in hasher.layout().layers.iter().enumerate() {
        let mut ids = vec![];
        for (row, &is_valid) in valid.iter().enumerate() {
            if is_valid {
                let first = (row * hasher.layout().layers.len() + index) * width;
                ids.extend_from_slice(&hashes[first..first + width]);
            }
        }
        let plan = RowPlan::build(&ids, layer.rows, 8, 4096)?;
        if plan
            .restore
            .iter()
            .map(|&slot| plan.rows[slot] as i64)
            .collect::<Vec<_>>()
            != ids
        {
            return Err("Engram deduplication changed lookup order".into());
        }
        let mut staged_bytes = 0;
        let mut row_bytes = 0;
        let mut rank_rows = vec![];
        for rank in 0..8 {
            let root = model_dir.join("engram_int8");
            let name = format!("layers_{}_engram_embed", layer.layer);
            let tensor = format!("layers.{}.engram.embed", layer.layer);
            let rows = Int8Rows::open(
                &root.join(format!("{name}.weight.safetensors")),
                &format!("{tensor}.weight"),
                &root.join(format!("{name}.scale.safetensors")),
                &format!("{tensor}.scale"),
                Shard::for_rank(layer.rows, 8, rank)?,
            )?;
            let ids = plan.for_rank(rank)?;
            let mut packet = vec![];
            rows.gather_into(ids, &mut packet, 1 << 20)?;
            row_bytes = rows.row_bytes();
            staged_bytes += packet.len();
            rank_rows.push(ids.len());
        }
        layers.push(Layer {
            layer: layer.layer,
            queries: ids.len(),
            unique_rows: plan.rows.len(),
            row_bytes,
            staged_bytes,
            rank_rows,
        });
    }
    Ok(Report {
        history,
        tokens,
        hashes_sha256: digest,
        chunked_history_verified: true,
        layers,
        scope: "CPU Engram hashing, stable ownership/deduplication and real INT8 row/scales staging; no model forward",
    })
}
