//! Independent bidirectional embedding requests; no autoregressive state.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub dimension: usize,
    pub output: String,
    pub program: String,
    pub length: String,
    pub feature_index: String,
    pub vision_program: String,
    pub pixels: String,
    pub grid: String,
    pub vision_length: String,
    pub vision_output: String,
    pub features: String,
    pub image_token_id: u32,
    pub max_patches: usize,
    pub patch_size: usize,
    pub merge_size: usize,
}

impl Spec {
    pub fn validate(&self, manifest: &crate::model::Manifest) -> crate::artifact::Result<()> {
        use crate::artifact::{Access, Dtype};
        if self.dimension == 0
            || self.dimension > 16384
            || !(1..=64).contains(&self.patch_size)
            || !(1..=8).contains(&self.merge_size)
            || !(4..=32768).contains(&self.max_patches)
            || !self.max_patches.is_multiple_of(self.merge_size.pow(2))
            || self.image_token_id as usize >= manifest.vocab
            || manifest.kv_cache.is_some()
            || manifest.mtp.is_some()
            || manifest.vision.is_some()
            || manifest.audio.is_some()
        {
            return Err("Invalid independent embedding contract".into());
        }
        for (name, dtype, elements) in [
            (&self.output, Dtype::F32, self.dimension),
            (&self.length, Dtype::I32, 1),
            (&self.feature_index, Dtype::I32, manifest.chunk_tokens),
            (&self.grid, Dtype::I32, 2),
            (&self.vision_length, Dtype::I32, 1),
            (
                &self.features,
                Dtype::F16,
                manifest.chunk_tokens * self.dimension,
            ),
            (
                &self.vision_output,
                Dtype::F16,
                self.max_patches / self.merge_size.pow(2) * self.dimension,
            ),
            (
                &self.pixels,
                Dtype::F16,
                self.max_patches * 3 * self.patch_size.pow(2),
            ),
        ] {
            let buffer = manifest
                .buffers
                .iter()
                .find(|b| b.name == *name)
                .ok_or("Missing embedding buffer")?;
            if buffer.dtype != dtype
                || buffer.bytes()? != elements * dtype.bytes()
                || buffer.access == Access::Read
                || buffer.data.is_some()
            {
                return Err(format!("Invalid embedding buffer {name}"));
            }
        }
        for program in [&self.program, &self.vision_program] {
            if manifest.programs.get(program).is_none_or(Vec::is_empty) {
                return Err("Missing embedding execution program".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    fn manifest() -> crate::model::Manifest {
        let buffers: Vec<_> = [
            ("Input", "i32", vec![512]),
            ("Control", "i32", vec![1]),
            ("Output", "f32", vec![8]),
            ("Length", "i32", vec![1]),
            ("Index", "i32", vec![512]),
            ("Grid", "i32", vec![2]),
            ("VisionLength", "i32", vec![1]),
            ("Features", "f16", vec![512, 8]),
            ("VisionOutput", "f16", vec![1, 8]),
            ("Pixels", "f16", vec![4, 588]),
        ]
        .into_iter()
        .map(|(name, dtype, shape)| {
            json!({"name":name,"dtype":dtype,"shape":shape,
            "layout":"native_contiguous","alignment":256,"access":"read_write"})
        })
        .collect();
        serde_json::from_value(json!({
            "schema_version":2,"target":"sm_87","model":"embedding-test","chunk_tokens":512,
            "max_context":512,"vocab":16,"toolchain":{"test":"1"},"buffers":buffers,
            "reset_buffers":[],"input":"Input","token":"Control","status":"Control",
            "position":"Control","logits":"Output","weight_bytes":0,"weight_parameters":1,"weight_scope":"test",
            "programs":{"embed":[{"kind":"zero","destination":"Output","bytes":32}],
                        "vision":[{"kind":"zero","destination":"VisionOutput","bytes":16}]},
            "embedding":{"dimension":8,"output":"Output","program":"embed","length":"Length",
                "feature_index":"Index","vision_program":"vision","pixels":"Pixels","grid":"Grid",
                "vision_length":"VisionLength","vision_output":"VisionOutput","features":"Features",
                "image_token_id":10,"max_patches":4,"patch_size":14,"merge_size":2}
        })).unwrap()
    }
    #[test]
    fn embedding_outputs_and_workspace_have_no_autoregressive_reset_requirement() {
        let m = manifest();
        m.validate().unwrap();
        let mut generation = m.clone();
        generation.embedding = None;
        assert!(
            generation
                .validate()
                .unwrap_err()
                .contains("full-vocabulary")
        );
        generation.vocab = 8;
        for name in ["prefill", "head", "decode"] {
            generation
                .programs
                .insert(name.into(), generation.programs["embed"].clone());
        }
        assert!(
            generation
                .validate()
                .unwrap_err()
                .contains("Position must reset")
        );
    }
    #[test]
    fn image_and_output_extents_are_validated() {
        for name in ["Output", "Features", "Pixels", "Grid", "Index"] {
            let mut m = manifest();
            m.buffers.iter_mut().find(|b| b.name == name).unwrap().shape[0] += 1;
            assert!(m.validate().is_err(), "{name}");
        }
    }
}
