//! Exact single-token producer fusions retain the original rounding boundaries.
use crate::{
    artifact::{Access, Argument, Dtype, Kernel, Result},
    model::{Manifest, Operation},
};
use std::collections::{BTreeMap, BTreeSet};

fn binding(kernel: &Kernel, index: usize) -> Result<&str> {
    match kernel.args.get(index) {
        Some(Argument::Buffer { name }) => Ok(name),
        _ => Err("Decode fusion requires whole buffer arguments".into()),
    }
}
fn connect(a: &Kernel, ai: usize, b: &Kernel, bi: usize) -> Result<()> {
    if binding(a, ai)? != binding(b, bi)? {
        return Err("Decode fusion producer/consumer mismatch".into());
    }
    Ok(())
}

pub(super) fn build(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let mut fused: BTreeSet<_> = kernels
        .iter()
        .filter(|k| {
            matches!(
                k.symbol.as_str(),
                "rotate_dispatch_a8_kernel" | "rotate_split_a8_kernel"
            )
        })
        .map(|k| k.name.as_str())
        .collect();
    let mut shared: BTreeSet<_> = kernels
        .iter()
        .filter(|k| k.symbol == "exact_e8p_shared_input_kernel")
        .map(|k| k.name.as_str())
        .collect();
    let buffers: BTreeMap<_, _> = manifest
        .buffers
        .iter()
        .map(|b| (b.name.as_str(), b))
        .collect();
    let check = |k: &Kernel, i: usize, dtype: Dtype, bytes: usize, immutable: bool| -> Result<()> {
        let name = binding(k, i)?;
        let b = buffers.get(name).ok_or("Missing decode fusion buffer")?;
        if b.dtype != dtype
            || b.bytes()? != bytes
            || b.data.is_some() != immutable
            || b.access
                != if immutable {
                    Access::Read
                } else {
                    Access::ReadWrite
                }
        {
            return Err(format!("Invalid decode fusion buffer: {name}"));
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
            if program != "decode" {
                return Err("Activation fusion requires single-token decode".into());
            }
            let producer = contracts[name.as_str()];
            let get = |offset: usize| -> Result<&Kernel> {
                let Some(Operation::Kernel { name }) = operations.get(index + offset) else {
                    return Err("Missing adjacent activation fusion operation".into());
                };
                contracts
                    .get(name.as_str())
                    .copied()
                    .ok_or("Missing activation fusion contract".into())
            };
            let gu = producer.symbol == "rotate_dispatch_a8_kernel";
            let (split, project) = if gu {
                (get(2)?, get(3)?)
            } else {
                (get(1)?, get(2)?)
            };
            if split.symbol != "pack_exact_a8_kernel"
                || split.args.len() != 4
                || project.args.len() != 13
                || project.symbol
                    != if gu {
                        "exact_e8p_shared_input_kernel"
                    } else {
                        "exact_e8p_int4_kernel"
                    }
                || producer.args.len() != if gu { 12 } else { 7 }
                || producer.grid != if gu { [1, 1, 1] } else { [10, 1, 1] }
            {
                return Err("Unexpected activation fusion ABI/geometry".into());
            }
            let (hi, lo, sum, q, scale, signs, input) = if gu {
                let dispatch = get(1)?;
                if dispatch.symbol != "main_kernel"
                    || dispatch.args.len() != 8
                    || dispatch.grid != [10, 1, 1]
                {
                    return Err("Invalid single-token activation dispatch".into());
                }
                for (a, b) in [(0, 2), (2, 3), (4, 0), (5, 4), (6, 5), (7, 6), (9, 7)] {
                    connect(producer, a, dispatch, b)?;
                }
                connect(split, 0, dispatch, 1)?;
                for (i, size) in [(2, 40), (5, 40), (6, 2048), (9, 40)] {
                    check(producer, i, Dtype::I32, size, false)?;
                }
                check(producer, 0, Dtype::F16, 20, false)?;
                skip.insert(index + 1);
                shared.remove(project.name.as_str());
                (1, 3, 10, 4, 7, 8, 11)
            } else {
                connect(producer, 2, split, 0)?;
                (0, 1, 5, 2, 3, 4, 6)
            };
            for (a, b) in [(hi, 1), (lo, 2), (sum, 3)] {
                connect(producer, a, split, b)?;
            }
            for (a, b) in [(hi, 3), (lo, 4), (sum, 8), (scale, 0)] {
                connect(producer, a, project, b)?;
            }
            let width = if gu { 2560 } else { 640 };
            let rows = if gu { 1 } else { 10 };
            check(producer, hi, Dtype::U32, 10 * width / 2, false)?;
            check(producer, lo, Dtype::U32, 10 * width / 2, false)?;
            check(producer, sum, Dtype::I32, 40, false)?;
            check(producer, q, Dtype::I8, rows * width, false)?;
            check(producer, scale, Dtype::F16, rows * 2, false)?;
            check(producer, signs, Dtype::I8, width, true)?;
            check(
                producer,
                input,
                Dtype::F16,
                rows * width * if gu { 2 } else { 4 },
                false,
            )?;
            // Ten distinct top-k experts have one shared input on GU, while
            // each expert's SwiGLU output owns a separate row on down.
            let mut distinct = BTreeSet::new();
            for i in [hi, lo, sum, q, scale, input] {
                if !distinct.insert(binding(producer, i)?) {
                    return Err("Aliased activation fusion workspace".into());
                }
            }
            skip.insert(index + if gu { 2 } else { 1 });
            fused.remove(name.as_str());
        }
        if !skip.is_empty() {
            *operations = std::mem::take(operations)
                .into_iter()
                .enumerate()
                .filter_map(|(i, op)| (!skip.contains(&i)).then_some(op))
                .collect();
        }
    }
    if !fused.is_empty() || !shared.is_empty() {
        return Err("Unregistered activation fusion".into());
    }
    build_hc_finish(manifest, kernels)?;
    build_hc_combine_norm(manifest, kernels)?;
    build_conv_history(manifest, kernels)
}

/// The fused combine+norm kernel replaces the separate pair in single-token
/// decode; its host ABI order is Block, Inject, Normed, Output, Residual,
/// Weight. The removed norm keeps identical rounding boundaries.
fn build_hc_combine_norm(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let mut fused: BTreeSet<_> = kernels
        .iter()
        .filter(|k| k.symbol == "hc_combine_norm_kernel")
        .map(|k| k.name.as_str())
        .collect();
    let buffers: BTreeMap<_, _> = manifest
        .buffers
        .iter()
        .map(|b| (b.name.as_str(), b))
        .collect();
    let check = |k: &Kernel, i: usize, dtype: Dtype, bytes: usize, immutable: bool| -> Result<()> {
        let name = binding(k, i)?;
        let b = buffers.get(name).ok_or("Missing HC combine-norm buffer")?;
        if b.dtype != dtype
            || b.bytes()? != bytes
            || b.data.is_some() != immutable
            || b.access
                != if immutable {
                    Access::Read
                } else {
                    Access::ReadWrite
                }
        {
            return Err(format!("Invalid HC combine-norm buffer: {name}"));
        }
        Ok(())
    };
    for (program, operations) in &mut manifest.programs {
        if program != "decode" {
            continue;
        }
        let mut skip = BTreeSet::new();
        for index in 0..operations.len().saturating_sub(1) {
            let Operation::Kernel { name } = &operations[index] else {
                continue;
            };
            if !fused.contains(name.as_str()) {
                continue;
            }
            let Operation::Kernel { name: norm_name } = &operations[index + 1] else {
                return Err("HC combine-norm fusion requires an adjacent norm".into());
            };
            let fused_kernel = contracts[name.as_str()];
            let norm = contracts
                .get(norm_name.as_str())
                .copied()
                .ok_or("Missing HC combine-norm contract")?;
            // Host ABI order: Block, Inject, Normed, Output, Residual, Weight.
            if fused_kernel.args.len() != 6
                || fused_kernel.grid != [1, 4, 1]
                || fused_kernel.block != [256, 1, 1]
                || norm.symbol != "main_kernel"
                || norm.args.len() != 3
                || norm.grid != [1, 4, 1]
                || norm.block != [256, 1, 1]
            {
                return Err("Unexpected HC combine-norm ABI/geometry".into());
            }
            connect(fused_kernel, 2, norm, 0)?;
            connect(fused_kernel, 4, norm, 1)?;
            connect(fused_kernel, 5, norm, 2)?;
            check(fused_kernel, 0, Dtype::F16, 2560 * 2, false)?;
            check(fused_kernel, 1, Dtype::F16, 4 * 2, false)?;
            check(fused_kernel, 2, Dtype::F16, 4 * 2560 * 2, false)?;
            check(fused_kernel, 3, Dtype::F16, 4 * 2560 * 2, false)?;
            check(fused_kernel, 4, Dtype::F16, 4 * 2560 * 2, false)?;
            check(fused_kernel, 5, Dtype::F32, 4 * 2560 * 4, true)?;
            skip.insert(index + 1);
            fused.remove(name.as_str());
        }
        if !skip.is_empty() {
            *operations = std::mem::take(operations)
                .into_iter()
                .enumerate()
                .filter_map(|(i, op)| (!skip.contains(&i)).then_some(op))
                .collect();
        }
    }
    if !fused.is_empty() {
        return Err("Unregistered HC combine-norm fusion".into());
    }
    Ok(())
}

fn build_conv_history(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let buffers: BTreeMap<_, _> = manifest
        .buffers
        .iter()
        .map(|b| (b.name.as_str(), b))
        .collect();
    let mut fused: BTreeSet<_> = kernels
        .iter()
        .filter(|k| k.symbol == "gdn_conv_inplace_kernel")
        .map(|k| k.name.as_str())
        .collect();
    for (program, operations) in &mut manifest.programs {
        let mut skip = BTreeSet::new();
        for (index, op) in operations.iter().enumerate() {
            let Operation::Kernel { name } = op else {
                continue;
            };
            if !fused.contains(name.as_str()) {
                continue;
            }
            let k = contracts[name.as_str()];
            if program != "decode"
                || k.args.len() != 9
                || k.grid != [1, 80, 1]
                || k.block != [16, 1, 1]
            {
                return Err("Unexpected in-place decode convolution ABI".into());
            }
            let Some(Operation::Copy {
                source,
                destination,
                bytes,
            }) = operations.get(index + 1)
            else {
                return Err("Missing adjacent convolution history copy".into());
            };
            if destination != binding(k, 0)? || *bytes != 3 * 10240 * 2 || source == destination {
                return Err("Convolution history copy mismatch".into());
            }
            let check = |name: &str, dtype: Dtype, bytes: usize, immutable: bool| -> Result<()> {
                let b = buffers.get(name).ok_or("Missing convolution buffer")?;
                if b.dtype != dtype
                    || b.bytes()? != bytes
                    || b.data.is_some() != immutable
                    || b.access
                        != if immutable {
                            Access::Read
                        } else {
                            Access::ReadWrite
                        }
                {
                    return Err(format!("Invalid convolution buffer: {name}"));
                }
                Ok(())
            };
            check(source, Dtype::F16, 3 * 10240 * 2, false)?;
            let mut distinct = BTreeSet::new();
            for (i, dtype, size, immutable) in [
                (0, Dtype::F16, 3 * 10240 * 2, false),
                (1, Dtype::F16, 2048 * 2, false),
                (2, Dtype::F16, 2048 * 2, false),
                (3, Dtype::F16, 6144 * 2, false),
                (4, Dtype::F32, 10240 * 4 * 4, true),
                (5, Dtype::F16, 10240 * 2, false),
                (6, Dtype::I32, 4, false),
                (7, Dtype::I32, 4, false),
                (8, Dtype::I32, 4, false),
            ] {
                let name = binding(k, i)?;
                check(name, dtype, size, immutable)?;
                if !distinct.insert(name) {
                    return Err("Aliased convolution buffers".into());
                }
            }
            skip.insert(index + 1);
            fused.remove(name.as_str());
        }
        if !skip.is_empty() {
            *operations = std::mem::take(operations)
                .into_iter()
                .enumerate()
                .filter_map(|(i, op)| (!skip.contains(&i)).then_some(op))
                .collect();
        }
    }
    if !fused.is_empty() {
        return Err("Unregistered in-place convolution".into());
    }
    Ok(())
}

fn build_hc_finish(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let mut fused: BTreeSet<_> = kernels
        .iter()
        .filter(|k| k.symbol == "hc_up13_finish_kernel")
        .map(|k| k.name.as_str())
        .collect();
    let buffers: BTreeMap<_, _> = manifest
        .buffers
        .iter()
        .map(|b| (b.name.as_str(), b))
        .collect();
    let check = |k: &Kernel, i: usize, dtype: Dtype, bytes: usize, immutable: bool| -> Result<()> {
        let name = binding(k, i)?;
        let b = buffers.get(name).ok_or("Missing HC finish buffer")?;
        if b.dtype != dtype
            || b.bytes()? != bytes
            || b.data.is_some() != immutable
            || b.access
                != if immutable {
                    Access::Read
                } else {
                    Access::ReadWrite
                }
        {
            return Err(format!("Invalid HC finish buffer: {name}"));
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
            if !matches!(program.as_str(), "decode" | "head_m1") || index < 2 {
                return Err("HC finish fusion requires single-token target execution".into());
            }
            let get = |i: usize| -> Result<&Kernel> {
                let Some(Operation::Kernel { name }) = operations.get(i) else {
                    return Err("Missing adjacent HC finish operation".into());
                };
                contracts
                    .get(name.as_str())
                    .copied()
                    .ok_or("Missing HC finish contract".into())
            };
            let down = get(index - 2)?;
            let finish = get(index - 1)?;
            let up = contracts[name.as_str()];
            if up.args.len() != 5 && up.args.len() != 6
                || up.grid != [1, 80, 1]
                || up.block != [128, 1, 1]
                || !matches!(down.symbol.as_str(), "main_kernel" | "bf16_single_kernel")
                || down.args.len() != 3
                || down.block
                    != if down.symbol == "bf16_single_kernel" {
                        [64, 1, 1]
                    } else {
                        [128, 1, 1]
                    }
                || finish.symbol != "main_kernel"
                || finish.args.len() != 2
                || finish.block != [128, 1, 1]
            {
                return Err("Unexpected HC finish fusion ABI/geometry".into());
            }
            connect(up, 0, finish, 1)?;
            connect(up, 0, down, 1)?;
            let input = buffers
                .get(binding(up, 0)?)
                .ok_or("Missing HC partial buffer")?;
            let rank = input.bytes()? / (4 * 4);
            if !matches!(rank, 256 | 320)
                || finish.grid != [rank.div_ceil(128) as u32, 1, 1]
                || down.grid
                    != if down.symbol == "bf16_single_kernel" {
                        [rank.div_ceil(32) as u32, 4, 1]
                    } else {
                        [1, rank.div_ceil(64) as u32, 4]
                    }
            {
                return Err("Invalid HC finish rank".into());
            }
            let normed = 3;
            let out = 4;
            connect(up, normed, down, 0)?;
            check(up, 0, Dtype::F32, 4 * rank * 4, false)?;
            check(up, 1, Dtype::U32, 80 * (rank / 64) * 128 * 10 * 4, true)?;
            check(up, 2, Dtype::U32, 80 * (rank / 64) * 128 * 16 * 4, true)?;
            check(up, normed, Dtype::F16, 4 * 2560 * 2, false)?;
            check(up, out, Dtype::F16, 2560 * 2, false)?;
            check(finish, 0, Dtype::F16, rank * 2, false)?;
            let mut distinct = BTreeSet::new();
            for (k, i) in [(up, 0), (up, normed), (up, out), (finish, 0)] {
                if !distinct.insert(binding(k, i)?) {
                    return Err("Aliased HC finish workspace".into());
                }
            }
            if up.args.len() == 6 {
                let name = binding(up, 5)?;
                let b = buffers.get(name).ok_or("Missing HC escape buffer")?;
                let bytes = b.bytes()?;
                if !matches!(bytes, 8 | 16 | 24) {
                    return Err("Invalid HC escape count".into());
                }
                check(up, 5, Dtype::U64, bytes, true)?;
            }
            skip.insert(index - 1);
            fused.remove(name.as_str());
        }
        if !skip.is_empty() {
            *operations = std::mem::take(operations)
                .into_iter()
                .enumerate()
                .filter_map(|(i, op)| (!skip.contains(&i)).then_some(op))
                .collect();
        }
    }
    if !fused.is_empty() {
        return Err("Unregistered HC finish fusion".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn in_place_history_removes_only_the_validated_private_copy() {
        let (mut m, mut ks) = hc_fixture(320, false);
        let defs = [
            ("History", "f16", 30720, false),
            ("K", "f16", 2048, false),
            ("Q", "f16", 2048, false),
            ("V", "f16", 6144, false),
            ("W", "f32", 40960, true),
            ("X", "f16", 10240, false),
            ("Length", "i32", 1, false),
            ("Position", "i32", 1, false),
            ("PositionOut", "i32", 1, false),
            ("OldHistory", "f16", 30720, false),
        ];
        m.buffers = defs
            .into_iter()
            .map(|(name, dtype, n, immutable)| {
                let mut b = serde_json::json!({"name":name,"dtype":dtype,"shape":[n],
                "layout":"native_contiguous","alignment":256,
                "access":if immutable {"read"}else{"read_write"}});
                if immutable {
                    b["data"] = serde_json::json!({"tensor":name,"sha256":"0".repeat(64)});
                }
                serde_json::from_value(b).unwrap()
            })
            .collect();
        let mut k = ks.remove(0);
        k.name = "decode/conv".into();
        k.symbol = "gdn_conv_inplace_kernel".into();
        k.grid = [1, 80, 1];
        k.block = [16, 1, 1];
        k.args = defs[..9]
            .iter()
            .map(|(name, _, _, _)| Argument::Buffer {
                name: (*name).into(),
            })
            .collect();
        m.programs.insert(
            "decode".into(),
            vec![
                Operation::Kernel {
                    name: k.name.clone(),
                },
                Operation::Copy {
                    source: "OldHistory".into(),
                    destination: "History".into(),
                    bytes: 61440,
                },
            ],
        );
        let mut valid = m.clone();
        build_conv_history(&mut valid, std::slice::from_ref(&k)).unwrap();
        assert_eq!(valid.programs["decode"].len(), 1);
        let mut corrupt = m.clone();
        if let Operation::Copy { bytes, .. } = &mut corrupt.programs.get_mut("decode").unwrap()[1] {
            *bytes -= 1;
        }
        assert!(build_conv_history(&mut corrupt, std::slice::from_ref(&k)).is_err());
        let mut alias = k.clone();
        alias.args[8] = alias.args[7].clone();
        assert!(
            build_conv_history(&mut m.clone(), &[alias])
                .unwrap_err()
                .contains("Aliased")
        );
        let ops = m.programs.remove("decode").unwrap();
        m.programs.insert("verify_m4".into(), ops);
        assert!(build_conv_history(&mut m, &[k]).is_err());
    }
    fn hc_fixture(rank: usize, patches: bool) -> (Manifest, Vec<Kernel>) {
        let (mut m, _) = fixture(false);
        let definitions = [
            ("Partial", "f32", 4 * rank, false),
            ("Activation", "f16", rank, false),
            ("Normed", "f16", 10240, false),
            ("Output", "f16", 2560, false),
            ("Hi", "u32", 80 * (rank / 64) * 128 * 10, true),
            ("Lo", "u32", 80 * (rank / 64) * 128 * 16, true),
            ("Patches", "u64", 3, true),
        ];
        m.buffers = definitions
            .into_iter()
            .map(|(name, dtype, n, immutable)| {
                let mut b = serde_json::json!({"name":name,"dtype":dtype,"shape":[n],
                "layout":"native_contiguous","alignment":256,
                "access":if immutable {"read"} else {"read_write"}});
                if immutable {
                    b["data"] = serde_json::json!({"tensor":name,"sha256":"0".repeat(64)});
                }
                serde_json::from_value(b).unwrap()
            })
            .collect();
        let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
        let mut up = vec!["Partial", "Hi", "Lo", "Normed", "Output"];
        if patches {
            up.push("Patches");
        }
        let specs = [
            (
                "main_kernel",
                vec!["Normed", "Partial", "Weight"],
                [1, rank.div_ceil(64) as u32, 4],
            ),
            (
                "main_kernel",
                vec!["Activation", "Partial"],
                [rank.div_ceil(128) as u32, 1, 1],
            ),
            ("hc_up13_finish_kernel", up, [1, 80, 1]),
        ];
        let k: Vec<Kernel> = specs.into_iter().enumerate().map(|(i,(symbol,args,grid))|
            serde_json::from_value(serde_json::json!({"name":format!("decode/body/k{i}"),
                "symbol":symbol,"module":identity,"source":identity,"host_abi":identity,
                "grid":grid,"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
                "args":args.iter().map(|name|serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
            })).unwrap()).collect();
        m.programs.insert(
            "decode".into(),
            k.iter()
                .map(|k| Operation::Kernel {
                    name: k.name.clone(),
                })
                .collect(),
        );
        (m, k)
    }
    #[test]
    fn hc_finish_fusion_requires_exact_producer_geometry_and_workspace() {
        for rank in [256, 320] {
            for patches in [false, true] {
                let (mut m, k) = hc_fixture(rank, patches);
                build(&mut m, &k).unwrap();
                assert_eq!(m.programs["decode"].len(), 2);
                let (mut m, mut k) = hc_fixture(rank, patches);
                k[0].symbol = "bf16_single_kernel".into();
                k[0].grid = [rank.div_ceil(32) as u32, 4, 1];
                k[0].block = [64, 1, 1];
                build(&mut m, &k).unwrap();
                assert_eq!(m.programs["decode"].len(), 2);
                let (mut m, mut k) = hc_fixture(rank, patches);
                k[2].args[0] = Argument::Buffer {
                    name: "Output".into(),
                };
                assert!(build(&mut m, &k).is_err());
                let (mut m, k) = hc_fixture(rank, patches);
                m.buffers[0].shape[0] -= 1;
                assert!(build(&mut m, &k).is_err());
                let (mut m, k) = hc_fixture(rank, patches);
                let ops = m.programs.remove("decode").unwrap();
                m.programs.insert("verify_m4".into(), ops);
                assert!(build(&mut m, &k).is_err());
            }
        }
    }
    fn combine_norm_fixture() -> (Manifest, Vec<Kernel>) {
        let (mut m, _) = fixture(false);
        let definitions = [
            ("Block", "f16", 2560, false),
            ("Inject", "f16", 4, false),
            ("Normed", "f16", 10240, false),
            ("Output", "f16", 10240, false),
            ("Residual", "f16", 10240, false),
            ("Weight", "f32", 10240, true),
        ];
        m.buffers = definitions
            .into_iter()
            .map(|(name, dtype, n, immutable)| {
                let mut b = serde_json::json!({"name":name,"dtype":dtype,"shape":[n],
                "layout":"native_contiguous","alignment":256,
                "access":if immutable {"read"}else{"read_write"}});
                if immutable {
                    b["data"] = serde_json::json!({"tensor":name,"sha256":"0".repeat(64)});
                }
                serde_json::from_value(b).unwrap()
            })
            .collect();
        let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
        let specs = [
            (
                "hc_combine_norm_kernel",
                vec!["Block", "Inject", "Normed", "Output", "Residual", "Weight"],
                [1, 4, 1],
                [256, 1, 1],
            ),
            (
                "main_kernel",
                vec!["Normed", "Residual", "Weight"],
                [1, 4, 1],
                [256, 1, 1],
            ),
        ];
        let k: Vec<Kernel> = specs
            .into_iter()
            .enumerate()
            .map(|(i, (symbol, args, grid, block))| {
                serde_json::from_value(serde_json::json!({"name":format!("decode/body/k{i}"),
                    "symbol":symbol,"module":identity,"source":identity,"host_abi":identity,
                    "grid":grid,"block":block,"shared_memory_bytes":0,"cooperative":false,
                    "args":args.iter().map(|name|serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
                })).unwrap()
            })
            .collect();
        m.programs.insert(
            "decode".into(),
            k.iter()
                .map(|k| Operation::Kernel {
                    name: k.name.clone(),
                })
                .collect(),
        );
        (m, k)
    }
    #[test]
    fn combine_norm_removes_only_the_validated_adjacent_norm() {
        let (mut m, k) = combine_norm_fixture();
        build(&mut m, &k).unwrap();
        assert_eq!(m.programs["decode"].len(), 1);
        let (mut m, mut k) = combine_norm_fixture();
        k[1].args[2] = Argument::Buffer {
            name: "Output".into(),
        };
        assert!(build(&mut m, &k).is_err());
        let (mut m, mut k) = combine_norm_fixture();
        k[0].block = [128, 1, 1];
        assert!(build(&mut m, &k).is_err());
        let (mut m, k) = combine_norm_fixture();
        m.buffers[0].shape[0] -= 1;
        assert!(build(&mut m, &k).is_err());
        let (mut m, k) = combine_norm_fixture();
        let ops = m.programs.remove("decode").unwrap();
        m.programs.insert("verify_m4".into(), ops);
        assert!(build(&mut m, &k).is_err());
    }
    fn fixture(gu: bool) -> (Manifest, Vec<Kernel>) {
        let width = if gu { 2560 } else { 640 };
        let rows = if gu { 1 } else { 10 };
        let definitions = [
            ("High", "u32", 10 * width / 8, false),
            ("Low", "u32", 10 * width / 8, false),
            ("Sum", "i32", 10, false),
            ("Q", "i8", rows * width, false),
            ("S", "f16", rows, false),
            ("Signs", "i8", width, true),
            ("X", "f16", rows * width * if gu { 1 } else { 2 }, false),
            ("DS", "f16", 10, false),
            ("IDs", "i32", 10, false),
            ("RelativeSlot", "i32", 10, false),
            ("RowOffsets", "i32", 512, false),
            ("SlotMap", "i32", 10, false),
        ];
        let buffers: Vec<_> = definitions.into_iter().map(|(name,dtype,n,immutable)| {
            let mut b=serde_json::json!({"name":name,"dtype":dtype,"shape":[n],"layout":"native_contiguous","alignment":256,
                "access":if immutable {"read"}else{"read_write"}});
            if immutable {b["data"]=serde_json::json!({"tensor":name,"sha256":"0".repeat(64)});}
            b
        }).collect();
        let names: &[(&str, &[&str])] = if gu {
            &[
                (
                    "rotate_dispatch_a8_kernel",
                    &[
                        "DS",
                        "High",
                        "IDs",
                        "Low",
                        "Q",
                        "RelativeSlot",
                        "RowOffsets",
                        "S",
                        "Signs",
                        "SlotMap",
                        "Sum",
                        "X",
                    ],
                ),
                (
                    "main_kernel",
                    &[
                        "Q",
                        "D",
                        "DS",
                        "IDs",
                        "RelativeSlot",
                        "RowOffsets",
                        "S",
                        "SlotMap",
                    ],
                ),
                ("pack_exact_a8_kernel", &["D", "High", "Low", "Sum"]),
                (
                    "exact_e8p_shared_input_kernel",
                    &[
                        "S", "Book", "Counts", "High", "Low", "Offsets", "Out", "P", "Sum", "TC",
                        "TE", "TR", "WS",
                    ],
                ),
            ]
        } else {
            &[
                (
                    "rotate_split_a8_kernel",
                    &["High", "Low", "Q", "S", "Signs", "Sum", "X"],
                ),
                ("pack_exact_a8_kernel", &["Q", "High", "Low", "Sum"]),
                (
                    "exact_e8p_int4_kernel",
                    &[
                        "S", "Book", "Counts", "High", "Low", "Offsets", "Out", "P", "Sum", "TC",
                        "TE", "TR", "WS",
                    ],
                ),
            ]
        };
        let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
        let kernels: Vec<Kernel> = names.iter().enumerate().map(|(i,(symbol,args))| serde_json::from_value(serde_json::json!({
            "name":format!("decode/layer0/k{i}"),"symbol":symbol,"module":identity,"source":identity,"host_abi":identity,
            "grid":if gu && i==0 {[1,1,1]}else{[10,1,1]},"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
            "args":args.iter().map(|name|serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
        })).unwrap()).collect();
        let manifest=serde_json::from_value(serde_json::json!({"schema_version":1,"target":"sm_120","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":buffers,"reset_buffers":[],"input":"Input","token":"Token",
            "status":"Status","logits":"Logits","position":"Position","weight_bytes":0,"weight_parameters":0,"weight_scope":"test",
            "programs":{"decode":kernels.iter().map(|k|serde_json::json!({"kind":"kernel","name":k.name})).collect::<Vec<_>>()}
        })).unwrap();
        (manifest, kernels)
    }
    #[test]
    fn removes_dispatch_and_split_only_when_producer_contract_matches() {
        for gu in [false, true] {
            let (mut m, k) = fixture(gu);
            build(&mut m, &k).unwrap();
            assert_eq!(m.programs["decode"].len(), 2);
            assert!(
                matches!(&m.programs["decode"][1],Operation::Kernel{name} if name==&k.last().unwrap().name)
            );
        }
    }
    #[test]
    fn rejects_corrupt_bindings_aliases_and_short_buffers() {
        let (mut m, mut k) = fixture(true);
        k[0].args[3] = Argument::Buffer {
            name: "High".into(),
        };
        assert!(build(&mut m, &k).unwrap_err().contains("mismatch"));
        let (mut m, k) = fixture(false);
        m.buffers
            .iter_mut()
            .find(|b| b.name == "Low")
            .unwrap()
            .shape[0] -= 1;
        assert!(build(&mut m, &k).unwrap_err().contains("buffer"));
        let (mut m, mut k) = fixture(false);
        k[0].args[1] = Argument::BufferSlice {
            name: "Low".into(),
            offset: 0,
        };
        assert!(build(&mut m, &k).unwrap_err().contains("whole buffer"));
    }
    #[test]
    fn rejects_batched_fusion_and_unregistered_shared_projection() {
        let (mut m, k) = fixture(true);
        let ops = m.programs.remove("decode").unwrap();
        m.programs.insert("verify_m4".into(), ops);
        assert!(build(&mut m, &k).unwrap_err().contains("single-token"));
        let (mut m, mut k) = fixture(true);
        k[0].symbol = "main_kernel".into();
        assert!(build(&mut m, &k).unwrap_err().contains("Unregistered"));
        let (mut m, mut k) = fixture(false);
        k[2].args[0] = Argument::Buffer { name: "DS".into() };
        assert!(build(&mut m, &k).unwrap_err().contains("mismatch"));
    }
}
