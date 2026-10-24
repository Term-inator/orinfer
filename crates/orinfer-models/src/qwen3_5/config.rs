//! Qwen configuration semantics and compiled arithmetic signature.
//! Weight files contain data; operator packages contain bindings. Control flow
//! belongs to the registered architecture implementation.
use crate::artifact::Result;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub(crate) struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub linear_num_key_heads: usize,
    #[serde(default)]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub linear_key_head_dim: usize,
    #[serde(default)]
    pub linear_value_head_dim: usize,
    #[serde(default)]
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "query_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}

fn query_heads() -> usize {
    24
}
fn kv_heads() -> usize {
    4
}
fn head_dim() -> usize {
    256
}

impl TextConfig {
    pub fn qkv(&self) -> usize {
        (2 * self.linear_num_key_heads + self.linear_num_value_heads) * self.linear_key_head_dim
    }
    pub fn mixer(&self) -> usize {
        if self.linear_num_value_heads == 0 {
            self.full_q()
        } else {
            self.linear_num_value_heads * self.linear_value_head_dim
        }
    }
    pub fn full_q(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn full_x(&self) -> usize {
        (if self.linear_num_value_heads == 0 {
            1
        } else {
            2
        }) * self.full_q()
            + 2 * self.kv()
    }
}

fn parse_text(text_value: &Value, standard: bool) -> Result<TextConfig> {
    let text: TextConfig =
        serde_json::from_value(text_value.clone()).map_err(|e| format!("Text config: {e}"))?;
    if text.hidden_size == 0
        || text.vocab_size == 0
        || text.num_hidden_layers == 0
        || text.layer_types.len() != text.num_hidden_layers
        || text
            .layer_types
            .iter()
            .any(|s| !matches!(s.as_str(), "linear_attention" | "full_attention"))
        || (!standard
            && (text.linear_num_key_heads == 0
                || text.linear_num_value_heads == 0
                || text.linear_key_head_dim == 0
                || text.linear_value_head_dim == 0
                || text.linear_conv_kernel_dim < 2))
    {
        return Err("Invalid Qwen3_5 layer/state configuration".into());
    }
    Ok(text)
}

pub(crate) struct Configuration {
    text: TextConfig,
    pub vision_depth: usize,
    pub standard: bool,
    pub rotated_w8_layers: Vec<usize>,
    pub rotated_w8_matrices: Vec<String>,
    pub signature: Value,
}

impl Configuration {
    pub(super) fn text(&self) -> &TextConfig {
        &self.text
    }

    pub fn parse(config: Value) -> Result<Self> {
        if !matches!(
            config["model_type"].as_str(),
            Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_asr")
        ) {
            return Err("Unsupported Qwen model configuration".into());
        }
        let text_value = config.get("text_config").unwrap_or(&config);
        let standard = config["model_type"] == "qwen3_asr";
        let text = parse_text(text_value, standard)?;
        if standard
            && (text.layer_types.iter().any(|s| s != "full_attention")
                || config["audio_config"]["encoder_layers"] != 24
                || config["audio_config"]["d_model"] != 1024
                || text_value["attn_output_gate"] != false
                || text_value["rope_parameters"]["rope_theta"] != 1000000)
        {
            return Err("Unsupported ASR audio or decoder contract".into());
        }
        if standard {
            for (key, value) in [
                ("encoder_attention_heads", 16),
                ("encoder_ffn_dim", 4096),
                ("num_mel_bins", 128),
                ("output_dim", 2048),
                ("downsample_hidden_size", 480),
                ("n_window", 50),
                ("n_window_infer", 800),
            ] {
                if config["audio_config"][key] != value {
                    return Err(format!("Unsupported ASR audio {key}"));
                }
            }
        }
        if (config["model_type"] == "qwen3_5_moe") != (text.num_experts > 0) {
            return Err("Qwen dense/MoE configuration mismatch".into());
        }
        // These values affect compiled arithmetic, weight contracts or state.
        // Checkpoint identity, naming, dtype labels and quantization provenance
        // deliberately do not determine architecture/operator compatibility.
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../configs/architecture-contract.json"
        ))
        .map_err(|e| e.to_string())?;
        let family = if standard { "qwen3_asr" } else { "qwen3_5" };
        let rotated_w8_layers = if standard {
            let expected = &contract["families"][family]["int8_quality"]["rotated_w8_layers"];
            if config["quantization_config"]["rotated_w8_layers"] != *expected {
                return Err("ASR requires its quality-first W8 layer policy".into());
            }
            serde_json::from_value(expected.clone()).map_err(|e| e.to_string())?
        } else {
            Vec::new()
        };
        let rotated_w8_matrices: Vec<String> = config["quantization_config"]
            .get("rotated_w8_matrices")
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()
            .map_err(|e| format!("Invalid rotated W8 projections: {e}"))?
            .unwrap_or_default();
        let mut seen = std::collections::BTreeSet::new();
        for name in &rotated_w8_matrices {
            let (layer, role) = name
                .strip_prefix("model.language_model.layers.")
                .and_then(|s| s.split_once('.'))
                .ok_or("Invalid rotated W8 projection name")?;
            let index = layer
                .parse::<usize>()
                .map_err(|_| "Invalid rotated W8 layer")?;
            if standard
                || text.num_experts > 0
                || index >= text.num_hidden_layers
                || layer != index.to_string()
                || !seen.insert(name)
            {
                return Err("Invalid or duplicate dense rotated W8 projection".into());
            }
            let valid = match role {
                "self_attn.q_proj.weight"
                | "self_attn.k_proj.weight"
                | "self_attn.v_proj.weight" => text.layer_types[index] == "full_attention",
                "linear_attn.in_proj_qkv.weight" | "linear_attn.in_proj_z.weight" => {
                    text.layer_types[index] == "linear_attention"
                }
                "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.down_proj.weight" => true,
                _ => false,
            };
            if !valid {
                return Err(format!("Unsupported rotated W8 projection: {name}"));
            }
        }
        let text_keys = contract["families"][family]["text_keys"]
            .as_array()
            .ok_or("Missing architecture signature contract")?;
        let text_signature: serde_json::Map<String, Value> = text_keys
            .iter()
            .filter_map(|k| k.as_str())
            .filter_map(|k| text_value.get(k).map(|v| (k.into(), v.clone())))
            .collect();
        let vision_depth = config["vision_config"]["depth"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0);
        let mut signature = serde_json::json!({"text": text_signature,
            "vision": config.get("vision_config").cloned().unwrap_or(Value::Null)});
        if standard {
            signature["audio"] = config["audio_config"].clone();
        }
        Ok(Self {
            standard,
            rotated_w8_layers,
            rotated_w8_matrices,
            text,
            vision_depth,
            signature,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_rotated_w8_selection_rejects_unknown_roles_and_duplicates() {
        let name = "model.language_model.layers.1.self_attn.k_proj.weight";
        let mut raw = config();
        raw["quantization_config"] = serde_json::json!({"rotated_w8_matrices":[name]});
        let parsed = Configuration::parse(raw.clone()).unwrap();
        assert_eq!(parsed.rotated_w8_matrices, [name]);
        for selected in [
            serde_json::json!([name, name]),
            serde_json::json!(["model.language_model.layers.0.self_attn.k_proj.weight"]),
            serde_json::json!(["model.language_model.layers.2.self_attn.k_proj.weight"]),
            serde_json::json!(["model.language_model.layers.1.self_attn.o_proj.weight"]),
            serde_json::json!(null),
        ] {
            raw["quantization_config"]["rotated_w8_matrices"] = selected;
            assert!(Configuration::parse(raw.clone()).is_err());
        }
    }
    fn config() -> Value {
        serde_json::json!({"model_type":"qwen3_5_text","hidden_size":8,"intermediate_size":16,"vocab_size":16,
            "num_hidden_layers":2,"layer_types":["linear_attention","full_attention"],
            "linear_num_key_heads":1,"linear_num_value_heads":2,
            "linear_key_head_dim":2,"linear_value_head_dim":3,"linear_conv_kernel_dim":4})
    }
    #[test]
    fn configuration_rejects_unknown_or_incomplete_architectures() {
        let mut raw = config();
        raw["model_type"] = "other".into();
        assert!(Configuration::parse(raw).is_err());
        let mut raw = config();
        raw["layer_types"] = serde_json::json!(["linear_attention"]);
        assert!(Configuration::parse(raw).is_err());
        let mut raw = config();
        raw["linear_num_value_heads"] = 0.into();
        assert!(Configuration::parse(raw).is_err());
    }
    #[test]
    fn kernel_compatibility_tracks_math_not_checkpoint_provenance() {
        let original = Configuration::parse(config()).unwrap();
        let mut raw = config();
        raw["quantization_config"] = serde_json::json!({"source":"another checkpoint"});
        raw["dtype"] = "float16".into();
        assert_eq!(
            original.signature,
            Configuration::parse(raw).unwrap().signature
        );
        let mut raw = config();
        raw["rms_norm_eps"] = serde_json::json!(0.1);
        assert_ne!(
            original.signature,
            Configuration::parse(raw).unwrap().signature
        );
    }
}
