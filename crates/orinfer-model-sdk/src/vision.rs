//! Image feature layout and Qwen interleaved multimodal positions.
use crate::{
    artifact::{Access, Dtype, Result},
    model::Manifest,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    #[default]
    F16,
    Bf16,
}
impl Precision {
    fn storage(self) -> Dtype {
        match self {
            Self::F16 => Dtype::F16,
            Self::Bf16 => Dtype::Bf16,
        }
    }
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisionPlan {
    pub patches: usize,
    #[serde(default, skip_serializing_if = "crate::model::is_zero")]
    pub min_patches: usize,
    pub program: String,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisionSpec {
    #[serde(default)]
    pub dtype: Precision,
    pub hidden: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub max_patches: usize,
    pub max_features: usize,
    pub pixels: String,
    pub grid: String,
    pub length: String,
    pub output: String,
    pub features: String,
    pub feature_index: String,
    pub mrope_positions: String,
    pub plans: Vec<VisionPlan>,
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageInput {
    pub grid_height: usize,
    pub grid_width: usize,
    /// Normalized RGB, in checkpoint patch/channel/temporal/spatial order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pixels: Vec<f32>,
    /// Resized interleaved RGB8. Expanded only when this image is encoded.
    /// Exactly one of `pixels` and `rgb` must be populated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rgb: Vec<u8>,
}
impl ImageInput {
    pub fn host_bytes(&self) -> usize {
        self.pixels.len() * std::mem::size_of::<f32>() + self.rgb.len()
    }
    /// Preserve the checkpoint's merge/patch/channel/temporal/spatial order.
    /// Callers validate the grid and storage with `feature_count` first.
    pub fn normalized<'a>(&'a self, v: &'a VisionSpec) -> impl Iterator<Item = f32> + 'a {
        let patch = v.patch_size;
        let merge = v.merge_size;
        let width = self.grid_width * patch;
        let blocks = if self.rgb.is_empty() {
            0
        } else {
            self.grid_height / merge
        };
        self.pixels
            .iter()
            .copied()
            .chain((0..blocks).flat_map(move |br| {
                (0..self.grid_width / merge).flat_map(move |bc| {
                    (0..merge * merge).flat_map(move |inner| {
                        (0..3).flat_map(move |ch| {
                            (0..v.temporal_patch_size).flat_map(move |_| {
                                (0..patch * patch).map(move |xy| {
                                    let y = (br * merge + inner / merge) * patch + xy / patch;
                                    let x = (bc * merge + inner % merge) * patch + xy % patch;
                                    self.rgb[(y * width + x) * 3 + ch] as f32 / 127.5 - 1.0
                                })
                            })
                        })
                    })
                })
            }))
    }
}
impl VisionSpec {
    pub fn patch_values(&self) -> Result<usize> {
        self.patch_size
            .checked_mul(self.patch_size)
            .and_then(|n| n.checked_mul(self.temporal_patch_size))
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| "Patch size overflow".into())
    }
    pub fn feature_count(&self, image: &ImageInput) -> Result<usize> {
        let n = image
            .grid_height
            .checked_mul(image.grid_width)
            .ok_or("Image grid overflow")?;
        if image.grid_height == 0
            || image.grid_width == 0
            || !image.grid_height.is_multiple_of(self.merge_size)
            || !image.grid_width.is_multiple_of(self.merge_size)
            || n > self.max_patches
            || self.plans.iter().all(|p| p.patches < n)
            || if image.rgb.is_empty() {
                image.pixels.len()
                    != n.checked_mul(self.patch_values()?)
                        .ok_or("Pixel size overflow")?
            } else {
                !image.pixels.is_empty()
                    || image.rgb.len()
                        != n.checked_mul(self.patch_size * self.patch_size * 3)
                            .ok_or("Pixel size overflow")?
            }
            || image
                .pixels
                .iter()
                .any(|p| !p.is_finite() || !(-1.001..=1.001).contains(p))
        {
            return Err("Invalid image grid, pixel data or vision capacity".into());
        }
        Ok(n / self.merge_size / self.merge_size)
    }
    pub fn validate(&self, m: &Manifest) -> Result<()> {
        // This adapter is images-only. Future adapters declare their own layout.
        if self.hidden == 0
            || self.patch_size != 16
            || self.temporal_patch_size != 2
            || self.merge_size != 2
            || self.max_patches == 0
            || !self.max_patches.is_multiple_of(4)
            || self.max_features == 0
            || self.max_features > m.max_context
            || self.max_patches / 4 > self.max_features
        {
            return Err("Invalid image adapter dimensions".into());
        }
        let names = [
            &self.pixels,
            &self.grid,
            &self.length,
            &self.output,
            &self.features,
            &self.feature_index,
            &self.mrope_positions,
        ];
        if names.into_iter().collect::<BTreeSet<_>>().len() != 7 {
            return Err("Vision control and feature buffers must not alias".into());
        }
        let ids = [
            self.image_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        ];
        if ids.iter().any(|&id| id as usize >= m.vocab)
            || ids.into_iter().collect::<BTreeSet<_>>().len() != 3
        {
            return Err("Invalid vision token IDs".into());
        }
        let mut sizes = BTreeSet::new();
        for p in &self.plans {
            if p.patches == 0
                || p.min_patches > p.patches
                || !p.min_patches.is_multiple_of(4)
                || (p.min_patches > 0
                    && m.row_execution.as_ref().is_none_or(|rows| {
                        !rows.programs.contains(&p.program)
                            || rows.program_capacity(&p.program) < p.patches
                    }))
                || !p.patches.is_multiple_of(4)
                || p.patches > self.max_patches
                || !sizes.insert(p.patches)
                || m.programs.get(&p.program).is_none_or(Vec::is_empty)
            {
                return Err("Invalid vision graph plan".into());
            }
        }
        if !sizes.contains(&self.max_patches) {
            return Err("Missing maximum vision plan".into());
        }
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| "Vision size overflow".to_owned())
        };
        for (name, dtype, count) in [
            (
                &self.pixels,
                self.dtype.storage(),
                product(self.max_patches, self.patch_values()?)?,
            ),
            (&self.grid, Dtype::I32, 2),
            (&self.length, Dtype::I32, 1),
            (
                &self.output,
                Dtype::F16,
                product(self.max_patches / 4, self.hidden)?,
            ),
            (
                &self.features,
                Dtype::F16,
                product(self.max_features, self.hidden)?,
            ),
            (&self.feature_index, Dtype::I32, m.max_context),
            (
                &self.mrope_positions,
                Dtype::I32,
                product(m.max_context, 3)?,
            ),
        ] {
            let b = m
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .ok_or("Missing vision buffer")?;
            if b.dtype != dtype
                || b.access == Access::Read
                || b.bytes()? != product(count, dtype.bytes())?
            {
                return Err(format!("Invalid vision buffer {name}"));
            }
        }
        Ok(())
    }

    /// Expand one native image marker per image, preserving message order.
    pub fn expand(
        &self,
        tokens: &[u32],
        images: &[ImageInput],
        context: usize,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut next = 0;
        for &id in tokens {
            if id == self.image_token_id {
                let image = images.get(next).ok_or("Image marker has no image")?;
                out.extend(std::iter::repeat_n(id, self.feature_count(image)?));
                next += 1;
            } else {
                out.push(id);
            }
            if out.len() > context {
                return Err("Image tokens exceed context capacity".into());
            }
        }
        if next != images.len() {
            return Err("Images missing from rendered prompt".into());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> VisionSpec {
        serde_json::from_value(
            serde_json::json!({"hidden":8,"patch_size":16,"temporal_patch_size":2,
            "merge_size":2,"max_patches":64,"max_features":16,"pixels":"p","grid":"g",
            "length":"l","output":"o","features":"f","feature_index":"i","mrope_positions":"r",
            "plans":[{"patches":64,"program":"v"}],"image_token_id":10,"vision_start_token_id":11,
            "vision_end_token_id":12}),
        )
        .unwrap()
    }
    #[test]
    fn compact_rgb_preserves_patch_channels_and_duplicate_temporal_planes() {
        let v = spec();
        let image = ImageInput {
            grid_height: 2,
            grid_width: 4,
            rgb: (0..32 * 64 * 3).map(|n| (n * 37 % 256) as u8).collect(),
            ..Default::default()
        };
        assert_eq!(v.feature_count(&image).unwrap(), 2);
        let values: Vec<_> = image.normalized(&v).collect();
        assert_eq!(values.len(), 8 * 1536);
        for (patch, (y, x)) in [
            (0, 0),
            (0, 16),
            (16, 0),
            (16, 16),
            (0, 32),
            (0, 48),
            (16, 32),
            (16, 48),
        ]
        .into_iter()
        .enumerate()
        {
            for ch in 0..3 {
                for row in 0..16 {
                    for col in 0..16 {
                        let expected =
                            image.rgb[((y + row) * 64 + x + col) * 3 + ch] as f32 / 127.5 - 1.0;
                        for frame in 0..2 {
                            assert_eq!(
                                values[patch * 1536 + ch * 512 + frame * 256 + row * 16 + col]
                                    .to_bits(),
                                expected.to_bits()
                            );
                        }
                    }
                }
            }
        }
        let old: ImageInput = serde_json::from_value(
            serde_json::json!({"grid_height":2,"grid_width":4,"pixels":values}),
        )
        .unwrap();
        assert_eq!(v.feature_count(&old).unwrap(), 2);
        assert_eq!(old.normalized(&v).collect::<Vec<_>>(), values);
        assert_eq!(image.host_bytes() * 8, old.host_bytes());
        let mut invalid = image.clone();
        invalid.rgb.pop();
        assert!(v.feature_count(&invalid).is_err());
        invalid = image;
        invalid.pixels.push(0.0);
        assert!(v.feature_count(&invalid).is_err());
    }
}
