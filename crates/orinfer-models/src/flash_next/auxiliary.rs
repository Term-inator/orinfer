//! Fuse complete auxiliary producer chains in every target/draft program alias.
use crate::{
    artifact::{Argument, Kernel, Result},
    model::{Manifest, Operation},
};
use std::collections::{BTreeMap, BTreeSet};

fn buffers(k: &Kernel) -> Vec<&Argument> {
    k.args
        .iter()
        .filter(|a| matches!(a, Argument::Buffer { .. } | Argument::BufferSlice { .. }))
        .collect()
}

fn connected(a: &[&Argument], ai: usize, b: &[&Argument], bi: usize) -> Result<()> {
    if a.get(ai).zip(b.get(bi)).is_none_or(|(a, b)| !match (a, b) {
        (Argument::Buffer { name: a }, Argument::Buffer { name: b }) => a == b,
        (
            Argument::BufferSlice {
                name: a,
                offset: ao,
            },
            Argument::BufferSlice {
                name: b,
                offset: bo,
            },
        ) => a == b && ao == bo,
        _ => false,
    }) {
        return Err("Auxiliary fusion producer/consumer mismatch".into());
    }
    Ok(())
}

pub(super) fn build(m: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let selected: BTreeSet<_> = kernels
        .iter()
        .filter(|k| {
            matches!(
                k.symbol.as_str(),
                "ple_norm_gate_kernel" | "qsa_select_merge_kernel"
            ) && !k.name.contains("/rows/")
        })
        .map(|k| k.name.as_str())
        .collect();
    let mut registered = BTreeSet::new();
    let partials: BTreeSet<_> = kernels
        .iter()
        .filter(|k| k.symbol == "qsa_select_partial_kernel" && !k.name.contains("/rows/"))
        .map(|k| k.name.as_str())
        .collect();
    let mut registered_partials = BTreeSet::new();
    for operations in m.programs.values_mut() {
        let mut skip = BTreeSet::new();
        for (i, op) in operations.iter().enumerate() {
            let Operation::Kernel { name } = op else {
                continue;
            };
            if !selected.contains(name.as_str()) {
                continue;
            }
            let fused = contracts[name.as_str()];
            let selection = fused.symbol == "qsa_select_merge_kernel";
            let count = if fused.symbol == "ple_norm_gate_kernel" {
                2
            } else {
                10
            };
            let start = i
                .checked_sub(count)
                .ok_or("Incomplete auxiliary fusion chain")?;
            let parent = name.rsplit_once('/').ok_or("Missing auxiliary section")?.0;
            let previous: Vec<_> = operations[start..i]
                .iter()
                .map(|op| {
                    let Operation::Kernel { name } = op else {
                        return Err("Interrupted auxiliary fusion chain".to_owned());
                    };
                    if name.rsplit_once('/').map(|p| p.0) != Some(parent) {
                        return Err("Auxiliary fusion crosses sections".to_owned());
                    }
                    contracts
                        .get(name.as_str())
                        .copied()
                        .ok_or("Missing auxiliary producer contract".to_owned())
                })
                .collect::<Result<_>>()?;
            let args = buffers(fused);
            if selection && previous[0].symbol != "qsa_select_partial_kernel" {
                return Err("Missing QSA local selection producer".into());
            }
            if selection {
                registered_partials.insert(previous[0].name.as_str());
            }
            if fused.symbol == "ple_norm_gate_kernel" {
                let key = buffers(previous[0]);
                let query = buffers(previous[1]);
                if args.len() != 6 || key.len() != 3 || query.len() != 3 {
                    return Err("Invalid fused PLE ABI".into());
                }
                // Sorted host ABI: Gated, Key, KeyWeight, Query, QueryWeight, Value.
                connected(&args, 1, &key, 1)?;
                connected(&args, 2, &key, 2)?;
                connected(&args, 3, &query, 1)?;
                connected(&args, 4, &query, 2)?;
            } else if args.len() == 1 {
                // Independent-row wrappers retain the same owner/column table.
                for old in previous {
                    let old = buffers(old);
                    if old.len() != 1 {
                        return Err("Invalid private QSA fusion ABI".into());
                    }
                    connected(&args, 0, &old, 0)?;
                }
            } else {
                if args.len() != 3 {
                    return Err("Invalid fused QSA ABI".into());
                }
                let counts = buffers(previous[8]);
                let offsets = buffers(previous[9]);
                if counts.len() != 4 || offsets.len() != 5 {
                    return Err("Invalid QSA selection producers".into());
                }
                let partial = buffers(previous[0]);
                if partial.len() != 3 {
                    return Err("Invalid local QSA selection ABI".into());
                }
                connected(&partial, 2, &args, 0)?;
                connected(&partial, 0, &args, 1)?;
                let logical = [partial[0], partial[1], args[2]];
                for (index, pair) in previous[..8].as_chunks::<2>().0.iter().enumerate() {
                    let hist = buffers(pair[0]);
                    let choose = buffers(pair[1]);
                    if index == 0 {
                        if choose.len() != 4 {
                            return Err("Invalid QSA threshold producer".into());
                        }
                        connected(&args, 0, &choose, 0)?;
                        connected(&logical, 0, &choose, 1)?;
                        connected(&choose, 2, &counts, 2)?;
                        continue;
                    }
                    if hist.len() != 4 || choose.len() != 4 {
                        return Err("Invalid QSA radix producer ABI".into());
                    }
                    for (a, b) in [(1, 0), (3, 1)] {
                        connected(&hist, a, &logical, b)?;
                    }
                    connected(&hist, 0, &choose, 0)?;
                    connected(&hist, 1, &choose, 1)?;
                    connected(&hist, 2, &choose, 2)?;
                    connected(&hist, 2, &counts, 2)?;
                }
                connected(&counts, 0, &offsets, 0)?;
                connected(&counts, 1, &logical, 0)?;
                connected(&counts, 3, &logical, 1)?;
                connected(&offsets, 3, &logical, 0)?;
                connected(&offsets, 4, &logical, 2)?;
            }
            if (start + usize::from(selection)..i).any(|j| !skip.insert(j)) {
                return Err("Overlapping auxiliary fusions".into());
            }
            registered.insert(fused.name.as_str());
        }
        *operations = std::mem::take(operations)
            .into_iter()
            .enumerate()
            .filter_map(|(i, op)| (!skip.contains(&i)).then_some(op))
            .collect();
    }
    if registered != selected || registered_partials != partials {
        return Err("Unregistered auxiliary fusion".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(name: &str, symbol: &str, args: &[&str]) -> Kernel {
        let asset = serde_json::json!({"file":"k","sha256":"0".repeat(64)});
        serde_json::from_value(serde_json::json!({"name":name,"symbol":symbol,
            "module":asset,"source":asset,"host_abi":asset,"grid":[1,1,1],
            "block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
            "args":args.iter().map(|name| serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
        })).unwrap()
    }

    fn manifest(names: &[String]) -> Manifest {
        let ops: Vec<_> = names
            .iter()
            .map(|name| serde_json::json!({"kind":"kernel","name":name}))
            .collect();
        serde_json::from_value(serde_json::json!({"schema_version":1,"target":"sm_120","model":"test",
            "chunk_tokens":4096,"max_context":262144,"vocab":16,"toolchain":{},"buffers":[],
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status","logits":"Logits",
            "position":"Position","weight_bytes":0,"weight_parameters":0,"weight_scope":"test",
            "programs":{"main":ops,"alias":ops}
        })).unwrap()
    }

    #[test]
    fn ple_fuses_every_alias_and_requires_both_norm_producers() {
        let mut kernels = vec![
            kernel("layer/kn", "main_kernel", &["NK", "Key", "KW"]),
            kernel("layer/qn", "main_kernel", &["NQ", "Query", "QW"]),
            kernel(
                "layer/gate",
                "ple_norm_gate_kernel",
                &["Gated", "Key", "KW", "Query", "QW", "Value"],
            ),
        ];
        let names: Vec<_> = kernels.iter().map(|k| k.name.clone()).collect();
        let mut m = manifest(&names);
        build(&mut m, &kernels).unwrap();
        assert!(m.programs.values().all(|ops| ops.len() == 1));
        kernels[2].args[4] = Argument::Buffer {
            name: "OtherWeight".into(),
        };
        assert!(
            build(&mut manifest(&names), &kernels)
                .unwrap_err()
                .contains("producer/consumer")
        );
    }

    #[test]
    fn private_qsa_requires_one_owner_table_and_a_complete_section() {
        let mut kernels: Vec<_> = (0..11)
            .map(|i| {
                kernel(
                    &format!("private/k{i}"),
                    if i == 10 {
                        "qsa_select_merge_kernel"
                    } else if i == 0 {
                        "qsa_select_partial_kernel"
                    } else {
                        "main_kernel"
                    },
                    &["Owners"],
                )
            })
            .collect();
        let names: Vec<_> = kernels.iter().map(|k| k.name.clone()).collect();
        let mut m = manifest(&names);
        build(&mut m, &kernels).unwrap();
        assert!(m.programs.values().all(|ops| ops.len() == 2));
        let mut orphan = kernels.clone();
        orphan.push(kernel(
            "orphan/partial",
            "qsa_select_partial_kernel",
            &["Owners"],
        ));
        assert!(build(&mut manifest(&names), &orphan).is_err());
        kernels[3].args[0] = Argument::Buffer {
            name: "OtherOwner".into(),
        };
        assert!(build(&mut manifest(&names), &kernels).is_err());
        assert!(build(&mut manifest(&names[1..]), &kernels).is_err());
        let mut m = manifest(&names);
        m.programs.get_mut("main").unwrap().insert(
            5,
            Operation::Copy {
                source: "A".into(),
                destination: "B".into(),
                bytes: 4,
            },
        );
        assert!(build(&mut m, &kernels).is_err());
    }

    #[test]
    fn local_qsa_keeps_both_stages_and_checks_the_bounded_union() {
        let mut kernels = vec![kernel(
            "qsa/k0",
            "qsa_select_partial_kernel",
            &["Position", "Scores", "Candidates"],
        )];
        for i in 1..8 {
            kernels.push(kernel(
                &format!("qsa/k{i}"),
                "main_kernel",
                if i % 2 == 0 {
                    &["Candidates", "Position", "Prefix", "Scores"]
                } else {
                    &["Candidates", "Position", "Prefix", "Remaining"]
                },
            ));
        }
        kernels.push(kernel(
            "qsa/k8",
            "main_kernel",
            &["Counts", "Position", "Prefix", "Scores"],
        ));
        kernels.push(kernel(
            "qsa/k9",
            "main_kernel",
            &["Counts", "Greater", "Offsets", "Position", "Selected"],
        ));
        kernels.push(kernel(
            "qsa/k10",
            "qsa_select_merge_kernel",
            &["Candidates", "Position", "Selected"],
        ));
        let names: Vec<_> = kernels.iter().map(|k| k.name.clone()).collect();
        let mut m = manifest(&names);
        build(&mut m, &kernels).unwrap();
        assert!(m.programs.values().all(|ops| ops.len() == 2));
        kernels[10].args[0] = Argument::Buffer {
            name: "OtherUnion".into(),
        };
        assert!(build(&mut manifest(&names), &kernels).is_err());
    }
}
