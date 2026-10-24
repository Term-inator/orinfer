//! Manifest-defined MTP programs and distribution-preserving speculation.
use crate::{
    artifact::{Access, Dtype, Result},
    model::Manifest,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub position: String,
    pub input: String,
    pub token: String,
    pub status: String,
    pub verification_tokens: String,
    pub verification_status: String,
    pub draft_logits: String,
    pub verification_logits: String,
    pub feature_index: Option<String>,
    pub accepted_inputs: String,
    pub target_length: String,
    pub draft_program: String,
    /// When present, consume each target prefill chunk before reusing this ring.
    #[serde(default)]
    pub hidden_ring: Option<String>,
    /// Compact recurrent verification needs a commit even on full acceptance.
    #[serde(default)]
    pub commit_always: bool,
    /// Draft physical state that cannot be repaired by rewinding its cursor.
    #[serde(default)]
    pub draft_snapshot_program: Option<String>,
    #[serde(default)]
    pub draft_restore_program: Option<String>,
    pub default_verification_tokens: usize,
    pub warm_plans: Vec<WarmPlan>,
    pub capture_plans: Vec<CapturePlan>,
    pub verification_plans: Vec<VerificationPlan>,
    /// Shapes whose target verification can share one batch forward.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch_verification_tokens: Vec<usize>,
    /// Independent one-token draft packets; a padded fixed capacity is allowed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch_draft_profiles: Vec<usize>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub batch_commit: bool,
    /// Optional private pointer table consumed by an owner-parallel commit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_state_table: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_table: Option<CausalTable>,
    /// Immutable table bindings are shared across per-round policy clones.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_table: Option<std::sync::Arc<PacketTable>>,
    /// Joined token rows for warm; independent request/head capacity is separate.
    #[serde(default, skip_serializing_if = "crate::model::is_zero")]
    pub batch_warm_capacity: usize,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub batch_warm: bool,
    /// Target row counts validated for joint prefill/speculative execution.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mixed_verification_tokens: Vec<usize>,
    /// Architecture-owned speculative batch programs to retire when MTP is off.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch_programs: Vec<String>,
    /// Prefix/factor capacity is shared between widths within each owner arena.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub shared_verification_state: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PacketTable {
    pub buffer: String,
    pub columns: Vec<String>,
    pub kernels: std::collections::BTreeMap<String, String>,
    /// Original host argument index -> wrapped host argument index.
    #[serde(default)]
    pub scalar_arguments: std::collections::BTreeMap<String, Vec<(usize, usize)>>,
    pub copies: Vec<PacketCopy>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PacketCopy {
    pub source: String,
    pub destination: String,
    pub kernel: String,
    pub row_bytes: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalTable {
    pub buffer: String,
    pub columns: Vec<String>,
    pub row_offset: String,
    /// Verification and accepted-prefix commit retain exact pending GDN updates.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub deferred: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attention: Vec<CausalAttention>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalAttention {
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub program: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WarmPlan {
    pub tokens: usize,
    #[serde(default, skip_serializing_if = "crate::model::is_zero")]
    pub min_tokens: usize,
    pub program: String,
    pub head_program: String,
}

impl WarmPlan {
    pub fn accepts(&self, tokens: usize) -> bool {
        tokens == self.tokens
            || (self.min_tokens > 0 && (self.min_tokens..=self.tokens).contains(&tokens))
    }
}

#[cfg(test)]
mod warm_tests {
    use super::WarmPlan;
    #[test]
    fn dynamic_warm_accepts_real_tails_without_padding() {
        let p = WarmPlan {
            tokens: 512,
            min_tokens: 1,
            program: "warm".into(),
            head_program: "head".into(),
        };
        for n in [1, 3, 17, 129, 257, 511, 512] {
            assert!(p.accepts(n));
        }
        for n in [0, 513] {
            assert!(!p.accepts(n));
        }
        let exact = WarmPlan { min_tokens: 0, ..p };
        assert!(!exact.accepts(511));
        assert!(exact.accepts(512));
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturePlan {
    pub tokens: usize,
    pub program: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationPlan {
    pub tokens: usize,
    #[serde(default, skip_serializing_if = "crate::model::is_zero")]
    pub min_tokens: usize,
    pub program: String,
    pub restore_program: String,
    pub capture_program: String,
}

impl VerificationPlan {
    pub fn accepts(&self, tokens: usize) -> bool {
        tokens == self.tokens
            || (self.min_tokens >= 2 && (self.min_tokens..=self.tokens).contains(&tokens))
    }
}

impl Spec {
    pub fn verification_plan(&self, tokens: usize) -> Option<VerificationPlan> {
        self.verification_plans
            .iter()
            .find(|p| p.accepts(tokens))
            .map(|p| VerificationPlan {
                tokens,
                ..p.clone()
            })
    }
    pub fn verification_widths(&self) -> Vec<usize> {
        self.verification_plans
            .iter()
            .flat_map(|p| {
                (if p.min_tokens > 0 {
                    p.min_tokens
                } else {
                    p.tokens
                })..=p.tokens
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    pub fn best_verification_plan(&self, limit: usize) -> Option<VerificationPlan> {
        self.verification_widths()
            .into_iter()
            .rev()
            .find(|&n| n <= limit)
            .and_then(|n| self.verification_plan(n))
    }
    pub fn warm_plan(&self, remaining: usize) -> Result<(&WarmPlan, usize)> {
        self.warm_plans
            .iter()
            .filter_map(|p| {
                let live = p.tokens.min(remaining);
                p.accepts(live).then_some((p, live))
            })
            .max_by_key(|(_, live)| *live)
            .ok_or_else(|| "No compatible MTP warm plan".into())
    }
    pub fn validate(&self, model: &Manifest) -> Result<()> {
        if let Some(table) = &self.packet_table {
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
                return Err("Invalid MTP packet address table".into());
            }
        }
        let program = |name: &str| -> Result<()> {
            if model.programs.get(name).is_none_or(Vec::is_empty) {
                return Err(format!("Missing MTP program {name}"));
            }
            Ok(())
        };
        program(&self.draft_program)?;
        match (&self.draft_snapshot_program, &self.draft_restore_program) {
            (Some(save), Some(restore)) => {
                program(save)?;
                program(restore)?;
            }
            (None, None) => {}
            _ => return Err("MTP draft snapshot/restore must be paired".into()),
        }
        let mut capture_sizes = BTreeSet::new();
        for p in &self.capture_plans {
            if p.tokens == 0 || p.tokens > model.chunk_tokens || !capture_sizes.insert(p.tokens) {
                return Err("Invalid/duplicate MTP capture plan".into());
            }
            program(&p.program)?;
        }
        if !capture_sizes.contains(&1)
            || (model.prefill_plans.is_empty() && !capture_sizes.contains(&model.chunk_tokens))
            || model
                .prefill_plans
                .iter()
                .any(|p| !capture_sizes.contains(&p.chunk_tokens))
        {
            return Err("MTP capture must cover every prefill plan and single-token decode".into());
        }
        let mut warm_sizes = BTreeSet::new();
        for p in &self.warm_plans {
            if p.tokens == 0
                || p.tokens > model.max_context
                || p.min_tokens > p.tokens
                || !warm_sizes.insert(p.tokens)
            {
                return Err("Invalid/duplicate MTP warm plan".into());
            }
            program(&p.program)?;
            program(&p.head_program)?;
        }
        if !self.warm_plans.iter().any(|p| p.accepts(1)) {
            return Err("MTP needs a one-token warm fallback".into());
        }
        let mut verify_sizes = BTreeSet::new();
        for p in &self.verification_plans {
            if p.tokens < 2
                || p.tokens > model.max_context
                || (p.min_tokens > 0 && !(2..=p.tokens).contains(&p.min_tokens))
                || !verify_sizes.insert(p.tokens)
                || (p.min_tokens > 0
                    && model.row_execution.as_ref().is_none_or(|rows| {
                        [&p.program, &p.capture_program, &p.restore_program]
                            .iter()
                            .any(|name| {
                                !rows.programs.contains(*name)
                                    || rows.program_capacity(name) < p.tokens
                            })
                    }))
            {
                return Err("Invalid/duplicate MTP verification plan".into());
            }
            for name in [&p.program, &p.restore_program, &p.capture_program] {
                program(name)?;
            }
        }
        if self
            .verification_plan(self.default_verification_tokens)
            .is_none()
        {
            return Err("Missing default MTP verification shape".into());
        }
        if self
            .batch_verification_tokens
            .windows(2)
            .any(|p| p[0] >= p[1])
            || self
                .batch_verification_tokens
                .iter()
                .any(|&n| self.verification_plan(n).is_none())
        {
            return Err("Invalid batched MTP verification shapes".into());
        }
        if self.batch_draft_profiles.windows(2).any(|p| p[0] >= p[1])
            || self
                .batch_draft_profiles
                .iter()
                .any(|n| !(2..=128).contains(n))
        {
            return Err("Invalid batched MTP draft profiles".into());
        }
        if (self.batch_warm && self.batch_draft_profiles.is_empty())
            || (self.batch_commit && self.batch_verification_tokens.is_empty())
            || (!self.mixed_verification_tokens.is_empty()
                && self.batch_verification_tokens.is_empty())
        {
            return Err("Batched MTP warm/commit lacks execution profiles".into());
        }
        if !self.mixed_verification_tokens.is_empty()
            && (model.prefill_batch_profiles.is_empty()
                || model
                    .batch_layout
                    .as_ref()
                    .is_none_or(|l| l.profiles.is_empty()))
        {
            return Err("Mixed MTP verification lacks prefill row profiles".into());
        }
        if self
            .mixed_verification_tokens
            .windows(2)
            .any(|p| p[0] >= p[1])
            || self
                .mixed_verification_tokens
                .iter()
                .any(|n| !self.batch_verification_tokens.contains(n))
        {
            return Err("Invalid mixed MTP verification shapes".into());
        }
        let mut batch_programs = BTreeSet::new();
        for name in &self.batch_programs {
            if !batch_programs.insert(name) {
                return Err("Duplicate MTP batch program".into());
            }
            program(name)?;
        }
        let max_warm = *warm_sizes.last().ok_or("No MTP warm plans")?;
        let max_verify = *verify_sizes.last().ok_or("No MTP verification plans")?;
        if let Some(name) = &self.hidden_ring {
            let buffer = model
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .ok_or("Missing MTP hidden ring")?;
            if buffer.dtype != Dtype::F16
                || buffer.access == Access::Read
                || buffer.shape.len() != 2
                || buffer.shape[1] == 0
                || buffer.shape[0] < model.chunk_tokens.max(max_warm).max(max_verify)
                || buffer.data.is_some()
            {
                return Err(
                    "MTP hidden ring must hold every capture/warm/verification chunk".into(),
                );
            }
        }
        for (name, rows) in [
            (&self.draft_logits, 1),
            (&self.verification_logits, max_verify),
        ] {
            let buffer = model
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .ok_or("Missing MTP logits buffer")?;
            if buffer.dtype != Dtype::F32
                || buffer.access == Access::Read
                || buffer.shape != [rows, model.vocab]
            {
                return Err(format!("Invalid MTP logits {name}"));
            }
        }
        match (&model.vision, &self.feature_index) {
            (Some(vision), Some(name)) => {
                let buffer = model
                    .buffers
                    .iter()
                    .find(|b| &b.name == name)
                    .ok_or("Missing shifted MTP feature index")?;
                if name == &vision.feature_index
                    || buffer.dtype != Dtype::I32
                    || buffer.access == Access::Read
                    || buffer.shape != [model.max_context]
                {
                    return Err("Invalid shifted MTP feature index".into());
                }
            }
            (None, None) => {}
            _ => return Err("MTP multimodal inputs must match the vision adapter".into()),
        }
        let mut controls = BTreeSet::new();
        for (name, words) in [
            (&self.position, 1),
            (&self.input, max_warm),
            (&self.token, 1),
            (&self.status, 1),
            (&self.accepted_inputs, 1),
            (&self.verification_tokens, max_verify),
            (&self.verification_status, max_verify),
        ] {
            let buffer = model
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .ok_or("Missing MTP control buffer")?;
            if buffer.dtype != Dtype::I32
                || buffer.access == Access::Read
                || buffer.bytes()? < words.checked_mul(4).ok_or("MTP size overflow")?
                || !controls.insert(name)
                || [&model.position, &model.input, &model.token, &model.status].contains(&name)
            {
                return Err(format!("Invalid/aliased MTP control {name}"));
            }
        }
        if !model.reset_buffers.contains(&self.position) {
            return Err("MTP position must reset between requests".into());
        }
        let length = model
            .buffers
            .iter()
            .find(|b| b.name == self.target_length)
            .ok_or("Missing target sequence length")?;
        if length.dtype != Dtype::I32
            || length.access == Access::Read
            || length.bytes()? != 4
            || controls.contains(&self.target_length)
            || [&model.position, &model.input, &model.token, &model.status]
                .contains(&&self.target_length)
        {
            return Err("Invalid/aliased target sequence length".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod verification_tests {
    use super::*;
    #[test]
    fn policy_clones_share_packet_bindings_and_preserve_the_wire_contract() {
        let spec: Spec = serde_json::from_value(serde_json::json!({
            "position":"Position", "input":"Input", "token":"Token", "status":"Status",
            "verification_tokens":"VerificationTokens", "verification_status":"VerificationStatus",
            "draft_logits":"DraftLogits", "verification_logits":"VerificationLogits",
            "accepted_inputs":"AcceptedInputs", "target_length":"TargetLength",
            "draft_program":"draft", "default_verification_tokens":8,
            "warm_plans":[], "capture_plans":[], "verification_plans":[],
            "packet_table":{
                "buffer":"Pointers", "columns":["State"],
                "kernels":{"original":"indirect"},
                "scalar_arguments":{"original":[[0,2]]}, "copies":[]
            }
        }))
        .unwrap();
        let wire = serde_json::to_value(&spec).unwrap();
        let mut policy = spec.clone();
        assert!(std::sync::Arc::ptr_eq(
            spec.packet_table.as_ref().unwrap(),
            policy.packet_table.as_ref().unwrap()
        ));
        assert_eq!(serde_json::to_value(&policy).unwrap(), wire);
        policy.default_verification_tokens = 4;
        assert_eq!(spec.default_verification_tokens, 8);
        let restored: Spec = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(&restored).unwrap(), wire);
    }

    #[test]
    fn one_verification_entry_preserves_every_draft_width_and_short_tail() {
        let p = VerificationPlan {
            tokens: 8,
            min_tokens: 2,
            program: "verify".into(),
            restore_program: "restore".into(),
            capture_program: "capture".into(),
        };
        for n in 2..=8 {
            assert!(p.accepts(n));
        }
        for n in [0, 1, 9] {
            assert!(!p.accepts(n));
        }
        let exact = VerificationPlan { min_tokens: 0, ..p };
        assert!(!exact.accepts(7));
        assert!(exact.accepts(8));
    }
}
