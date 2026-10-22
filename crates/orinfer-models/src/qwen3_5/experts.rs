//! Qwen whole-expert storage and routing dependencies for the shared VMM cache.
use crate::{
    artifact::{Argument, Dtype, Kernel, Result},
    model::Manifest,
};
use orinfer_model_sdk::abi::ExpertBank;
use std::collections::BTreeMap;

pub(super) fn banks(
    m: &Manifest,
    kernels: &[Kernel],
    layers: usize,
    experts: usize,
) -> Result<Vec<ExpertBank>> {
    let buffers: BTreeMap<_, _> = m.buffers.iter().map(|b| (b.name.as_str(), b)).collect();
    (0..layers)
        .map(|layer| {
            let prefix = format!("model.language_model.layers.{layer}.mlp.experts.");
            let names: Vec<_> = ["gate_up_proj", "down_proj"]
                .into_iter()
                .flat_map(|role| {
                    ["packed", "scale"].into_iter().map({
                        let prefix = prefix.clone();
                        move |part| format!("{prefix}{role}.{part}")
                    })
                })
                .collect();
            for name in &names {
                let b = buffers
                    .get(name.as_str())
                    .ok_or("Missing Qwen expert bank tensor")?;
                if b.data.is_none() || b.shape.first() != Some(&experts) {
                    return Err("Invalid Qwen expert bank tensor".into());
                }
            }
            let mut consumers = BTreeMap::new();
            for k in kernels {
                if !k
                    .args
                    .iter()
                    .any(|a| matches!(a, Argument::Buffer { name } if names.contains(name)))
                {
                    continue;
                }
                let counts: Vec<_> = k
                    .args
                    .iter()
                    .enumerate()
                    .filter_map(|(i, a)| match a {
                        Argument::Buffer { name } if name == "MoeCounts" => Some(i),
                        _ => None,
                    })
                    .collect();
                if counts.len() != 1 {
                    return Err(format!(
                        "{}: expert consumer needs one routing histogram",
                        k.name
                    ));
                }
                let b = buffers
                    .get("MoeCounts")
                    .ok_or("Missing Qwen expert routing histogram")?;
                if b.dtype != Dtype::I32 || b.shape != [experts] || b.data.is_some() {
                    return Err("Invalid Qwen expert routing histogram".into());
                }
                consumers.insert(k.name.clone(), counts[0]);
            }
            if consumers.is_empty() && !kernels.is_empty() {
                return Err(format!("Missing Qwen layer {layer} expert consumers"));
            }
            // Up/down and their scales share residency: both projections consume
            // the same route, including expansion and all selected row variants.
            Ok(ExpertBank {
                experts,
                buffers: names,
                consumers,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> (Manifest, Vec<Kernel>) {
        let names: Vec<_> = ["gate_up_proj", "down_proj"]
            .into_iter()
            .flat_map(|role| {
                ["packed", "scale"].map(move |part| {
                    format!("model.language_model.layers.0.mlp.experts.{role}.{part}")
                })
            })
            .collect();
        let mut buffers: Vec<_> = names.iter().map(|name| json!({"name":name,"dtype":"u32","shape":[4,128],"layout":"test","alignment":256,"access":"read","data":{"tensor":name,"sha256":"0".repeat(64)}})).collect();
        buffers.push(json!({"name":"MoeCounts","dtype":"i32","shape":[4],"layout":"contiguous","alignment":256,"access":"read_write"}));
        let m = serde_json::from_value(json!({"schema_version":2,"target":"sm_120","model":"test","chunk_tokens":128,"max_context":4096,"vocab":4,"toolchain":{},"buffers":buffers,"reset_buffers":[],"input":"Input","token":"Token","status":"Status","logits":"Logits","position":"Step","weight_bytes":0,"weight_parameters":0,"weight_scope":"test"})).unwrap();
        let id = json!({"file":"test","sha256":"0".repeat(64)});
        let kernels = [0,2].into_iter().enumerate().map(|(i,n)| serde_json::from_value(json!({"name":format!("expert/{i}"),"symbol":"test","module":id,"source":id,"host_abi":id,"grid":[1,1,1],"block":[32,1,1],"shared_memory_bytes":0,"cooperative":false,"args":[{"kind":"buffer","name":names[n]},{"kind":"buffer","name":"MoeCounts"},{"kind":"buffer","name":names[n+1]}]})).unwrap()).collect();
        (m, kernels)
    }
    #[test]
    fn up_down_and_scales_share_one_bank_and_use_exported_histogram_order() {
        let (m, mut kernels) = fixture();
        kernels[1].args.swap(0, 1);
        let banks = banks(&m, &kernels, 1, 4).unwrap();
        assert_eq!(banks.len(), 1);
        assert_eq!(banks[0].buffers.len(), 4);
        assert_eq!(
            banks[0].consumers,
            BTreeMap::from([("expert/0".into(), 1), ("expert/1".into(), 0)])
        );
    }
    #[test]
    fn malformed_counts_and_missing_consumers_are_rejected() {
        let (mut m, mut kernels) = fixture();
        kernels[0].args.remove(1);
        assert!(banks(&m, &kernels, 1, 4).unwrap_err().contains("histogram"));
        let (_, kernels) = fixture();
        m.buffers.last_mut().unwrap().shape = vec![5];
        assert!(banks(&m, &kernels, 1, 4).unwrap_err().contains("histogram"));
    }
}
