//! Complete bidirectional embedding requests through the shared CUDA executor.
use crate::{
    artifact::Result,
    cuda::executor::Executor,
    execution::{ExecutionPhase, LoadOptions},
};
use serde::Serialize;
use std::{collections::BTreeMap, path::Path, time::Instant};

pub use orinfer_model_sdk::embedding::Spec;

#[derive(Debug, Serialize)]
pub struct Statistics {
    pub load_to_ready_s: f64,
    pub weight_bytes: usize,
    pub weight_arena_padding_bytes: usize,
    pub resident_weight_bytes: usize,
    pub effective_weight_bits: f64,
    pub allocated_bytes: usize,
}

pub struct Model {
    execution: Executor,
    // Keep the native library alive through executor teardown.
    _package: crate::model_package::ModelPackage,
    spec: Spec,
    context: usize,
    vocab: usize,
    input: String,
    frontend_assets: BTreeMap<String, String>,
    statistics: Statistics,
}

impl Model {
    pub fn load(path: &Path, options: LoadOptions) -> Result<Self> {
        let started = Instant::now();
        let prepared = crate::loader::load_with_experts(path, options.verify_weights, false)?;
        let m = &prepared.plan;
        let spec = m
            .embedding
            .clone()
            .ok_or("Model package does not provide embeddings")?;
        let preload = m.kernels.iter().map(|k| k.name.clone()).collect();
        let (execution, stats) = Executor::load(&prepared, options, &preload)?;
        let padding = stats.load_stages.weight_arena_padding_bytes;
        let resident_weight_bytes = m
            .weight_bytes
            .checked_add(padding)
            .ok_or("Embedding weight accounting overflow")?;
        let statistics = Statistics {
            load_to_ready_s: started.elapsed().as_secs_f64(),
            weight_bytes: m.weight_bytes,
            weight_arena_padding_bytes: padding,
            resident_weight_bytes,
            effective_weight_bits: resident_weight_bytes as f64 * 8.0 / m.weight_parameters as f64,
            allocated_bytes: stats.buffer_bytes,
        };
        Ok(Self {
            execution,
            _package: prepared.execution_model,
            spec,
            context: m.max_context,
            vocab: m.vocab,
            input: m.input.clone(),
            frontend_assets: prepared.frontend_assets,
            statistics,
        })
    }
    pub fn spec(&self) -> &Spec {
        &self.spec
    }
    pub fn max_context(&self) -> usize {
        self.context
    }
    pub fn frontend_assets(&self) -> &BTreeMap<String, String> {
        &self.frontend_assets
    }
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    pub fn embed(
        &mut self,
        tokens: &[u32],
        images: &[crate::vision::ImageInput],
        feature_index: &[i32],
        dimensions: Option<usize>,
    ) -> Result<Vec<f32>> {
        let dimension = dimensions.unwrap_or(self.spec.dimension);
        if dimension == 0
            || dimension > self.spec.dimension
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= self.vocab)
            || feature_index.len() != tokens.len()
        {
            return Err("Invalid embedding length, tokens, dimensions or feature index".into());
        }
        let mut features = 0usize;
        for image in images {
            let patches = image
                .grid_height
                .checked_mul(image.grid_width)
                .ok_or("Image grid overflow")?;
            if image.grid_height == 0
                || image.grid_width == 0
                || image.grid_height % self.spec.merge_size != 0
                || image.grid_width % self.spec.merge_size != 0
                || patches > self.spec.max_patches
                || image.pixels.len() != patches * self.spec.patch_size.pow(2) * 3
                || image
                    .pixels
                    .iter()
                    .any(|x| !x.is_finite() || x.abs() > 65504.0)
            {
                return Err("Invalid embedding image patches".into());
            }
            features += patches / self.spec.merge_size.pow(2);
        }
        if features > self.context {
            return Err("Image features exceed embedding context".into());
        }
        let mut next = 0usize;
        for (&token, &index) in tokens.iter().zip(feature_index) {
            if index >= 0 {
                if token != self.spec.image_token_id || index as usize != next || next >= features {
                    return Err("Invalid embedding image token ownership".into());
                }
                next += 1;
            } else if index != -1 || token == self.spec.image_token_id {
                return Err("Unbound embedding image token".into());
            }
        }
        if next != features {
            return Err("Unused embedding image features".into());
        }
        let e = &self.execution;
        let mut offset = 0;
        for image in images {
            let patches = image.grid_height * image.grid_width;
            let pixels: Vec<u8> = image
                .pixels
                .iter()
                .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
                .collect();
            e.upload_bytes(&self.spec.pixels, &pixels)?;
            e.upload_ids(
                &self.spec.grid,
                &[image.grid_height as u32, image.grid_width as u32],
            )?;
            e.upload_ids(&self.spec.vision_length, &[patches as u32])?;
            e.submit_program_rows(&self.spec.vision_program, patches, ExecutionPhase::Vision)?;
            let count = patches / self.spec.merge_size.pow(2);
            e.copy_range_async(
                &self.spec.vision_output,
                0,
                &self.spec.features,
                offset * self.spec.dimension * 2,
                count * self.spec.dimension * 2,
            )?;
            offset += count;
        }
        e.upload_ids(&self.input, tokens)?;
        e.upload_ids(&self.spec.length, &[tokens.len() as u32])?;
        let index: Vec<u8> = feature_index.iter().flat_map(|i| i.to_le_bytes()).collect();
        e.upload_bytes(&self.spec.feature_index, &index)?;
        e.submit_program_rows(&self.spec.program, tokens.len(), ExecutionPhase::Prefill)?;
        let bytes = e.download_bytes(&self.spec.output, self.spec.dimension * 4)?;
        let mut output: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(dimension)
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let norm = output
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= 1e-12 {
            return Err("Embedding execution produced a nonfinite or zero vector".into());
        }
        for x in &mut output {
            *x = (*x as f64 / norm) as f32;
        }
        Ok(output)
    }
}
