//! BF16 HC prefill operands retain the original FP16 materialization boundaries.
use crate::{
    artifact::{Access, Dtype, Kernel, Result},
    model::{Manifest, Operation},
};
use std::collections::{BTreeMap, BTreeSet};

fn binding(kernel: &Kernel, index: usize) -> Result<&str> {
    match kernel.args.get(index) {
        Some(crate::artifact::Argument::Buffer { name }) => Ok(name),
        _ => Err("HC prefill fusion requires whole-buffer arguments".into()),
    }
}

pub(super) fn build(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let fused: BTreeSet<_> = kernels
        .iter()
        .filter(|k| {
            k.symbol == "hc_prefill_up_mix_kernel"
                && !k.name.ends_with("/aligned_rows")
                && !k.name.contains("/choice/")
        })
        .map(|k| k.name.clone())
        .collect();
    if fused.is_empty() {
        return Ok(());
    }
    let mut registered = BTreeSet::new();
    let buffers: BTreeMap<_, _> = manifest
        .buffers
        .iter()
        .map(|b| (b.name.as_str(), b))
        .collect();
    let check = |name: &str, dtype: Dtype, elements: usize, immutable: bool| -> Result<()> {
        let b = buffers.get(name).ok_or("Missing HC prefill buffer")?;
        if b.dtype != dtype
            || b.bytes()? < elements * dtype.bytes()
            || b.data.is_some() != immutable
            || b.access
                != if immutable {
                    Access::Read
                } else {
                    Access::ReadWrite
                }
        {
            return Err(format!("Invalid HC prefill buffer: {name}"));
        }
        Ok(())
    };
    for (program, operations) in &mut manifest.programs {
        let mut skip = BTreeSet::new();
        for (index, operation) in operations.iter().enumerate() {
            let Operation::Kernel { name } = operation else {
                continue;
            };
            if !fused.contains(name.as_str()) {
                continue;
            }
            let rows = name
                .split('/')
                .next()
                .and_then(|s| s.strip_prefix("prefill_m"))
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| matches!(n, 512 | 4096))
                .ok_or("Unsupported HC prefill fusion program")?;
            if (program != &format!("prefill_m{rows}") && program != "prefill") || index < 2 {
                return Err("Unsupported HC prefill fusion program".into());
            }
            let get = |index: usize| -> Result<&Kernel> {
                let Some(Operation::Kernel { name }) = operations.get(index) else {
                    return Err("Missing adjacent HC prefill operation".into());
                };
                contracts
                    .get(name.as_str())
                    .copied()
                    .ok_or("Missing HC prefill execution contract".into())
            };
            let norm = get(index - 2)?;
            let down = get(index - 1)?;
            let up = get(index)?;
            let mix = get(index + 1)?;
            let inject = get(index + 2)?;
            let count = |k: &Kernel| {
                k.args
                    .iter()
                    .filter(|a| matches!(a, crate::artifact::Argument::Buffer { .. }))
                    .count()
            };
            let (inv, normalized, residual, weight) = match (norm.symbol.as_str(), count(norm)) {
                ("hc_prefill_norm_kernel", 4) => (0, 1, 2, 3),
                ("hc_prefill_combine_norm_kernel", 7) => (2, 3, 4, 6),
                _ => return Err("Invalid HC prefill norm ABI".into()),
            };
            if down.symbol != "hc_prefill_projection_grouped_kernel"
                || inject.symbol != "hc_prefill_projection_kernel"
                || count(down) != 3
                || count(up) != 6
                || count(mix) != 3
                || count(inject) != 3
                || binding(down, 0)? != binding(norm, normalized)?
                || binding(up, 0)? != binding(down, 1)?
                || binding(up, 1)? != binding(norm, inv)?
                || binding(up, 3)? != binding(norm, residual)?
                || binding(up, 5)? != binding(norm, weight)?
                || binding(up, 2)? != binding(mix, 0)?
                || binding(inject, 0)? != binding(norm, normalized)?
            {
                return Err("HC prefill fusion producer/consumer mismatch".into());
            }
            check(
                binding(norm, normalized)?,
                Dtype::Bf16,
                rows * 4 * 2560,
                false,
            )?;
            check(binding(up, 0)?, Dtype::Bf16, rows * 320, false)?;
            check(binding(up, 1)?, Dtype::F32, rows * 4, false)?;
            check(binding(up, 2)?, Dtype::F16, rows * 2560, false)?;
            check(binding(up, 3)?, Dtype::F16, rows * 4 * 2560, false)?;
            check(binding(up, 4)?, Dtype::Bf16, 4 * 2560 * 320, true)?;
            check(binding(up, 5)?, Dtype::F32, 4 * 2560, true)?;
            for (name, shape) in [
                (binding(norm, normalized)?, &[rows, 4, 2560][..]),
                (binding(up, 0)?, &[rows, 320][..]),
                (binding(up, 1)?, &[rows, 4][..]),
                (binding(up, 4)?, &[10240, 320][..]),
                (binding(up, 5)?, &[4, 2560][..]),
            ] {
                if buffers[name].shape != shape {
                    return Err(format!("Invalid HC prefill shape: {name}"));
                }
            }
            skip.insert(index + 1);
            registered.insert(name.clone());
        }
        if !skip.is_empty() {
            *operations = std::mem::take(operations)
                .into_iter()
                .enumerate()
                .filter_map(|(index, operation)| (!skip.contains(&index)).then_some(operation))
                .collect();
        }
    }
    if registered != fused {
        return Err("Unregistered HC prefill fusion operation".into());
    }
    for alternative in kernels
        .iter()
        .filter(|k| k.name.ends_with("/aligned_rows") && k.symbol == "hc_prefill_up_mix_kernel")
    {
        let base = alternative.name.strip_suffix("/aligned_rows").unwrap();
        let canonical = contracts
            .get(base)
            .ok_or("Missing aligned HC canonical slot")?;
        let matching = alternative.args.len() == canonical.args.len()
            && alternative
                .args
                .iter()
                .zip(&canonical.args)
                .all(|(a, b)| match (a, b) {
                    (
                        crate::artifact::Argument::Buffer { name: a },
                        crate::artifact::Argument::Buffer { name: b },
                    ) => a == b,
                    (
                        crate::artifact::Argument::I32 { value: a },
                        crate::artifact::Argument::I32 { value: b },
                    ) => a == b,
                    _ => false,
                });
        if !registered.contains(base) || !matching {
            return Err("Aligned HC changes the canonical operand contract".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Manifest, Vec<Kernel>) {
        let definitions = [
            ("BF", "bf16", vec![512, 4, 2560], false),
            ("Act", "bf16", vec![512, 320], false),
            ("Inv", "f32", vec![512, 4], false),
            ("Mixed", "f16", vec![512, 2560], false),
            ("Residual", "f16", vec![512, 4, 2560], false),
            ("Up", "bf16", vec![10240, 320], true),
            ("Weight", "f32", vec![4, 2560], true),
        ];
        let buffers: Vec<_> = definitions
            .into_iter()
            .map(|(name, dtype, shape, immutable)| {
                let mut b = serde_json::json!({"name":name,"dtype":dtype,"shape":shape,
                    "layout":"native_contiguous","alignment":256,
                    "access":if immutable {"read"} else {"read_write"}});
                if immutable {
                    b["data"] = serde_json::json!({"tensor":name,"sha256":"0".repeat(64)});
                }
                b
            })
            .collect();
        let m = serde_json::from_value(serde_json::json!({
            "schema_version":1,"target":"sm_120","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":buffers,
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
            "logits":"Logits","position":"Position","weight_bytes":0,
            "weight_parameters":0,"weight_scope":"test","programs":{"prefill_m512":
            (0..5).map(|i| serde_json::json!({"kind":"kernel","name":format!("prefill_m512/layer0/k{i}")})).collect::<Vec<_>>()}
        })).unwrap();
        let definitions: [(&str, &[&str]); 5] = [
            (
                "hc_prefill_norm_kernel",
                &["Inv", "BF", "Residual", "Weight"],
            ),
            (
                "hc_prefill_projection_grouped_kernel",
                &["BF", "Act", "Down"],
            ),
            (
                "hc_prefill_up_mix_kernel",
                &["Act", "Inv", "Mixed", "Residual", "Up", "Weight"],
            ),
            ("main_kernel", &["Mixed", "OldNormed", "OldUp"]),
            ("hc_prefill_projection_kernel", &["BF", "Inject", "IW"]),
        ];
        let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
        let kernels = definitions.into_iter().enumerate().map(|(i,(symbol,args))|
            serde_json::from_value(serde_json::json!({"name":format!("prefill_m512/layer0/k{i}"),
                "symbol":symbol,"module":identity,"source":identity,"host_abi":identity,
                "grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
                "args":args.iter().map(|name| serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()})).unwrap()
        ).collect();
        (m, kernels)
    }

    #[test]
    fn fusion_removes_only_the_adjacent_mix() {
        let (mut m, kernels) = fixture();
        build(&mut m, &kernels).unwrap();
        let names: Vec<_> = m.programs["prefill_m512"]
            .iter()
            .map(|op| {
                let Operation::Kernel { name } = op else {
                    unreachable!()
                };
                name.as_str()
            })
            .collect();
        assert_eq!(
            names,
            [
                "prefill_m512/layer0/k0",
                "prefill_m512/layer0/k1",
                "prefill_m512/layer0/k2",
                "prefill_m512/layer0/k4"
            ]
        );
    }

    #[test]
    fn fusion_rejects_wrong_output_and_inverse_shape() {
        let (mut m, mut kernels) = fixture();
        kernels[3].args[0] = crate::artifact::Argument::Buffer {
            name: "wrong".into(),
        };
        assert!(build(&mut m, &kernels).unwrap_err().contains("mismatch"));
        let (mut m, kernels) = fixture();
        m.buffers
            .iter_mut()
            .find(|b| b.name == "Inv")
            .unwrap()
            .shape = vec![512, 8];
        assert!(build(&mut m, &kernels).unwrap_err().contains("shape"));
    }

    #[test]
    fn fusion_rejects_a_different_program() {
        let (mut m, kernels) = fixture();
        let ops = m.programs.remove("prefill_m512").unwrap();
        m.programs.insert("decode".into(), ops);
        assert!(build(&mut m, &kernels).unwrap_err().contains("program"));
    }

    #[test]
    fn dynamic_prefill_fuses_both_the_canonical_program_and_alias() {
        let (mut m, mut kernels) = fixture();
        let operations = m.programs.remove("prefill_m512").unwrap();
        let operations: Vec<_> = operations
            .into_iter()
            .map(|op| match op {
                Operation::Kernel { name } => Operation::Kernel {
                    name: name.replace("m512", "m4096"),
                },
                other => other,
            })
            .collect();
        m.programs
            .insert("prefill_m4096".into(), operations.clone());
        m.programs.insert("prefill".into(), operations);
        for b in &mut m.buffers {
            if b.data.is_none() && b.shape.first() == Some(&512) {
                b.shape[0] = 4096;
            }
        }
        for k in &mut kernels {
            k.name = k.name.replace("m512", "m4096");
            k.args.push(crate::artifact::Argument::I32 { value: 4096 });
        }
        build(&mut m, &kernels).unwrap();
        assert_eq!(m.programs["prefill"], m.programs["prefill_m4096"]);
        assert_eq!(m.programs["prefill"].len(), 4);
    }
}
