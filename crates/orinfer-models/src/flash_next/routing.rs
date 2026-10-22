//! Single-token routing metadata fusion belongs to the Flash execution recipe.
use crate::{
    artifact::{Access, Argument, Dtype, Kernel, Result},
    model::{Manifest, Operation},
};
use std::collections::{BTreeMap, BTreeSet};

fn binding(kernel: &Kernel, index: usize) -> Result<&str> {
    match kernel.args.get(index) {
        Some(Argument::Buffer { name }) => Ok(name),
        _ => Err("Single-token routing requires whole buffer arguments".into()),
    }
}

pub(super) fn build(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let buffers: BTreeMap<_, _> = manifest
        .buffers
        .iter()
        .map(|b| (b.name.as_str(), b))
        .collect();
    let mut fused: BTreeSet<_> = kernels
        .iter()
        .filter(|k| k.symbol == "single_route_metadata_kernel_kernel")
        .map(|k| k.name.as_str())
        .collect();
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
                return Err("Single-token routing requires decode program".into());
            }
            let route = contracts[name.as_str()];
            let next = |offset: usize| -> Result<&Kernel> {
                let Some(Operation::Kernel { name }) = operations.get(index + offset) else {
                    return Err("Missing adjacent routing metadata operation".into());
                };
                contracts
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| "Missing routing metadata contract".into())
            };
            let prefix = next(1)?;
            let tiles = next(2)?;
            if route.args.len() != 8
                || prefix.args.len() != 4
                || tiles.args.len() != 4
                || prefix.symbol != "main_kernel"
                || tiles.symbol != "main_kernel"
            {
                return Err("Unexpected single-token routing ABI".into());
            }
            for (old, connections) in [
                (prefix, [(0, 0), (1, 2), (2, 4), (3, 6)]),
                (tiles, [(0, 0), (1, 5), (2, 6), (3, 7)]),
            ] {
                for (a, b) in connections {
                    if binding(old, a)? != binding(route, b)? {
                        return Err("Single-token routing producer/consumer mismatch".into());
                    }
                }
            }
            let mut distinct = BTreeSet::new();
            for (i, elements) in [
                (0, 512),
                (1, 10),
                (2, 512),
                (3, 10),
                (4, 1),
                (5, 11),
                (6, 512),
                (7, 11),
            ] {
                let name = binding(route, i)?;
                let b = buffers.get(name).ok_or("Missing routing buffer")?;
                if b.dtype != Dtype::I32
                    || b.bytes()? != elements * Dtype::I32.bytes()
                    || b.data.is_some()
                    || b.access != Access::ReadWrite
                    || !distinct.insert(name)
                {
                    return Err(format!("Invalid single-token routing buffer: {name}"));
                }
            }
            skip.extend([index + 1, index + 2]);
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
        return Err("Unregistered single-token routing fusion".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Manifest, Vec<Kernel>) {
        let names = [
            "Counts",
            "IDs",
            "Offsets",
            "Relative",
            "TileCount",
            "TileExpert",
            "TileOffsets",
            "TileRow",
        ];
        let sizes = [512, 10, 512, 10, 1, 11, 512, 11];
        let buffers: Vec<_> = names
            .iter()
            .zip(sizes)
            .map(|(name, n)| {
                serde_json::json!({
                    "name":name,"dtype":"i32","shape":[n],"layout":"native_contiguous",
                    "alignment":256,"access":"read_write"
                })
            })
            .collect();
        let manifest = serde_json::from_value(serde_json::json!({
            "schema_version":1,"target":"sm_120","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":buffers,
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
            "logits":"Logits","position":"Position","weight_bytes":0,
            "weight_parameters":0,"weight_scope":"test","programs":{"decode":
            (0..3).map(|i|serde_json::json!({"kind":"kernel","name":format!("decode/k{i}")})).collect::<Vec<_>>()}
        })).unwrap();
        let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
        let args = [
            names.to_vec(),
            vec!["Counts", "Offsets", "TileCount", "TileOffsets"],
            vec!["Counts", "TileExpert", "TileOffsets", "TileRow"],
        ];
        let kernels = args.into_iter().enumerate().map(|(i,args)| serde_json::from_value(
            serde_json::json!({"name":format!("decode/k{i}"),"symbol":if i==0 {"single_route_metadata_kernel_kernel"} else {"main_kernel"},
                "module":identity,"source":identity,"host_abi":identity,"grid":[1,1,1],
                "block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
                "args":args.iter().map(|name|serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()})
        ).unwrap()).collect();
        (manifest, kernels)
    }
    #[test]
    fn removes_only_the_two_routing_metadata_consumers() {
        let (mut m, k) = fixture();
        build(&mut m, &k).unwrap();
        assert_eq!(m.programs["decode"].len(), 1);
    }
    #[test]
    fn rejects_wrong_binding_and_aliased_input() {
        let (mut m, mut k) = fixture();
        k[2].args[2] = Argument::Buffer {
            name: "Relative".into(),
        };
        assert!(build(&mut m, &k).unwrap_err().contains("producer/consumer"));
        let (mut m, mut k) = fixture();
        k[0].args[1] = Argument::Buffer {
            name: "Relative".into(),
        };
        assert!(build(&mut m, &k).unwrap_err().contains("routing buffer"));
    }
    #[test]
    fn rejects_batched_program_and_missing_contract() {
        let (mut m, k) = fixture();
        let ops = m.programs.remove("decode").unwrap();
        m.programs.insert("verify_m4".into(), ops);
        assert!(build(&mut m, &k).unwrap_err().contains("decode program"));
        let (mut m, mut k) = fixture();
        k.pop();
        assert!(build(&mut m, &k).unwrap_err().contains("contract"));
    }
}
