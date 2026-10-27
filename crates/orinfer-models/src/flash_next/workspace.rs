//! QSA partials have contiguous rows and serial per-sequence lifetimes.
use crate::{
    artifact::{Argument, Dtype, Kernel, Result},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::operators::workspace::SequenceRows;
use std::collections::BTreeMap;

pub(super) fn configure(m: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let find = |name: &str| {
        kernels
            .iter()
            .find(|k| k.name == name)
            .ok_or_else(|| format!("Missing QSA workspace binding {name}"))
    };
    let producer = find("prefill_m4096/layer3/k27")?;
    let consumer = find("prefill_m4096/layer3/k28")?;
    if producer.args.len() != 11
        || consumer.args.len() != 6
        || producer.grid != [4096, 2, 8]
        || consumer.grid != [4096, 24, 1]
    {
        return Err("QSA workspace primitive geometry differs".into());
    }
    for (write, read, stride) in [
        (0, 0, 24 * 8 * 4),
        (3, 2, 24 * 8 * 4),
        (4, 3, 24 * 8 * 256 * 4),
    ] {
        let Argument::Buffer { name } = &producer.args[write] else {
            return Err("QSA partial must bind a whole buffer".into());
        };
        if !matches!(&consumer.args[read], Argument::Buffer { name: n } if n == name) {
            return Err("QSA producer and consumer partials differ".into());
        }
        let b = m
            .buffers
            .iter()
            .find(|b| &b.name == name)
            .ok_or("Missing QSA partial")?;
        if b.dtype != Dtype::F32 || b.bytes()? != 4096 * stride {
            return Err("QSA partial row geometry differs".into());
        }
        let mut rows = SequenceRows {
            bytes_per_row: stride,
            kernels: BTreeMap::new(),
        };
        for k in kernels {
            for (reference, argument, index) in [(producer, write, 10), (consumer, read, 5)] {
                if k.source.sha256 == reference.source.sha256
                    && k.host_abi.sha256 == reference.host_abi.sha256
                    && k.symbol == reference.symbol
                    && matches!(k.args.get(argument), Some(Argument::Buffer { name: n }) if n == name)
                {
                    if k.args.len() != reference.args.len() {
                        return Err("QSA partial ABI differs".into());
                    }
                    if let Some(dynamic) = m.dynamic_batch_kernels.get(&k.name)
                        && !dynamic.arguments.iter().any(|a| {
                            a.index == index
                                && matches!(
                                    a.value,
                                    orinfer_model_sdk::operators::dynamic::RowExpression::Rows
                                )
                        })
                    {
                        return Err(
                            "QSA workspace rows must equal the sequence segment length".into()
                        );
                    }
                    rows.kernels.insert(k.name.clone(), index);
                }
            }
        }
        // These programs execute one request segment at a time. Concatenated
        // packet scratch cannot use the largest-request admission bound.
        for (program, operations) in &m.programs {
            if operations.iter().any(
                |op| matches!(op, Operation::Kernel { name } if rows.kernels.contains_key(name)),
            ) && !m
                .prefill_plans
                .iter()
                .any(|p| p.prefill_program == *program || p.stages.contains(program))
                && !(program.starts_with("causal_rows/") && program.ends_with("/private"))
            {
                return Err("QSA workspace is used outside a serial sequence segment".into());
            }
        }
        if m.sequence_workspaces.insert(name.clone(), rows).is_some() {
            return Err("Repeated QSA workspace contract".into());
        }
    }
    Ok(())
}
