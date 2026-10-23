//! Complete causal sequence scoring with a two-class final-position output.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<crate::encoder_rows::Spec>,
    pub program: String,
    pub length: String,
    pub output: String,
    pub false_token_id: u32,
    pub true_token_id: u32,
}
impl Spec {
    pub fn validate(&self, m: &crate::model::Manifest) -> crate::artifact::Result<()> {
        use crate::artifact::{Access, Dtype};
        if let Some(rows) = &self.rows {
            rows.validate(m)?;
        }
        if self.false_token_id == self.true_token_id
            || [self.false_token_id, self.true_token_id]
                .iter()
                .any(|&id| id as usize >= m.vocab)
            || m.kv_cache.is_some()
            || m.mtp.is_some()
            || m.vision.is_some()
            || m.audio.is_some()
            || m.embedding.is_some()
            || m.programs.get(&self.program).is_none_or(Vec::is_empty)
        {
            return Err("Invalid independent reranking contract".into());
        }
        for (name, dtype, shape) in [
            (
                &self.output,
                Dtype::F32,
                self.rows
                    .as_ref()
                    .map_or_else(|| vec![2], |r| vec![r.max_sequences, 2]),
            ),
            (&self.length, Dtype::I32, vec![1]),
        ] {
            let buffer = m
                .buffers
                .iter()
                .find(|b| b.name == *name)
                .ok_or("Missing reranking buffer")?;
            if buffer.dtype != dtype
                || buffer.shape != shape
                || buffer.access == Access::Read
                || buffer.data.is_some()
            {
                return Err(format!("Invalid reranking buffer {name}"));
            }
        }
        Ok(())
    }
}
