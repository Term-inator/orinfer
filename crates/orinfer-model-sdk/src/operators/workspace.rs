//! Shared row-major scratch used serially by individual sequence segments.
//! Admission bounds its rows by the largest request context; each launch maps
//! only its actual segment length. This contract excludes concatenated packets.
use crate::{
    artifact::{Access, Argument, Kernel, Result},
    execution::Invocation,
    model::{Manifest, Operation},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceRows {
    pub bytes_per_row: usize,
    /// Kernel name to the signed scalar argument carrying the real row count.
    pub kernels: BTreeMap<String, usize>,
}
impl SequenceRows {
    pub fn bytes(&self, rows: usize, capacity: usize) -> Result<usize> {
        if rows == 0 || self.bytes_per_row == 0 {
            return Err("Invalid workspace rows".into());
        }
        let bytes = rows
            .checked_mul(self.bytes_per_row)
            .ok_or("Workspace rows overflow")?;
        if bytes > capacity {
            return Err("Workspace rows exceed buffer".into());
        }
        Ok(bytes)
    }
    pub fn invocation_bytes(
        &self,
        name: &str,
        kernel: &Kernel,
        invocation: &Invocation,
        capacity: usize,
    ) -> Result<Option<usize>> {
        let Some(&index) = self.kernels.get(&kernel.name) else {
            return Ok(None);
        };
        if invocation
            .views
            .get(name)
            .is_some_and(|v| v.buffer != name || v.offset != 0)
        {
            return Err("Sequence workspace must retain its whole shared binding".into());
        }
        let rows = invocation
            .launch
            .as_ref()
            .and_then(|l| {
                l.arguments
                    .iter()
                    .rev()
                    .find(|(i, _)| *i == index)
                    .map(|(_, n)| *n)
            })
            .or_else(|| match kernel.args.get(index) {
                Some(Argument::I32 { value }) => Some(*value),
                _ => None,
            })
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("Invalid workspace row argument")?;
        self.bytes(rows, capacity).map(Some)
    }
}

pub fn validate(m: &Manifest) -> Result<()> {
    for (name, rows) in &m.sequence_workspaces {
        let b = m
            .buffers
            .iter()
            .find(|b| &b.name == name)
            .ok_or("Unknown sequence workspace")?;
        let capacity = b.bytes()?;
        if rows.bytes_per_row == 0
            || !capacity.is_multiple_of(rows.bytes_per_row)
            || rows.kernels.is_empty()
            || b.data.is_some()
            || b.access != Access::ReadWrite
            || m.reset_buffers.contains(name)
            || m.kv_cache.as_ref().is_some_and(|kv| {
                kv.buffers.contains_key(name) || kv.prefill_workspace.contains_key(name)
            })
        {
            return Err("Invalid sequence workspace storage".into());
        }
        let mut uses = 0;
        for k in &m.kernels {
            let bound = k
                .args
                .iter()
                .any(|a| matches!(a, Argument::Buffer { name: n } if n == name));
            let sliced = k
                .args
                .iter()
                .any(|a| matches!(a, Argument::BufferSlice { name: n, .. } if n == name));
            if sliced || bound != rows.kernels.contains_key(&k.name) {
                return Err("Incomplete sequence workspace kernel contract".into());
            }
            if bound {
                uses += 1;
                let index = rows.kernels[&k.name];
                let Some(Argument::I32 { value }) = k.args.get(index) else {
                    return Err("Workspace rows require an i32 argument".into());
                };
                rows.bytes(
                    usize::try_from(*value).map_err(|_| "Negative workspace rows")?,
                    capacity,
                )?;
            }
        }
        if uses != rows.kernels.len()
            || m.programs.values().flatten().any(|op| match op {
                Operation::Copy {
                    source,
                    destination,
                    ..
                } => source == name || destination == name,
                Operation::Zero { destination, .. } => destination == name,
                _ => false,
            })
            || m.prefill_plans
                .iter()
                .any(|p| p.live_buffers.contains(name))
        {
            return Err("Sequence workspace has an unbounded use".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Manifest {
        let identity = serde_json::json!({"file":"test", "sha256":"0".repeat(64)});
        serde_json::from_value(serde_json::json!({
            "schema_version":2,"target":"sm_120","model":"test","chunk_tokens":4,"max_context":4,"vocab":1,
            "toolchain":{},"buffers":[{"name":"Partial","dtype":"u8","shape":[64],"layout":"row_major","alignment":256,"access":"read_write"}],
            "kernels":[{"name":"k","module":identity,"source":identity,"host_abi":identity,
                "symbol":"test","grid":[4,1,1],"block":[32,1,1],"shared_memory_bytes":0,"cooperative":false,
                "args":[{"kind":"buffer","name":"Partial"},{"kind":"i32","value":4}]}],
            "programs":{"p":[{"kind":"kernel","name":"k"}]},"reset_buffers":[],
            "input":"","token":"","status":"","logits":"","position":"",
            "weight_bytes":0,"weight_parameters":1,"weight_scope":"test",
            "sequence_workspaces":{"Partial":{"bytes_per_row":16,"kernels":{"k":1}}}
        })).unwrap()
    }
    #[test]
    fn all_workspace_uses_must_have_a_bounded_row_contract() {
        let m = fixture();
        validate(&m).unwrap();
        let mut unclassified = m.clone();
        let mut extra = m.kernels[0].clone();
        extra.name = "unclassified".into();
        unclassified.kernels.push(extra);
        assert!(validate(&unclassified).is_err());
        let mut pointer = m.clone();
        pointer
            .sequence_workspaces
            .get_mut("Partial")
            .unwrap()
            .kernels
            .insert("k".into(), 0);
        assert!(validate(&pointer).is_err());
        let mut slice = m.clone();
        slice.kernels[0].args[0] = Argument::BufferSlice {
            name: "Partial".into(),
            offset: 0,
        };
        assert!(validate(&slice).is_err());
        let mut oversized = m.clone();
        oversized.kernels[0].args[1] = Argument::I32 { value: 5 };
        assert!(validate(&oversized).is_err());
        let mut copy = m.clone();
        copy.programs.get_mut("p").unwrap().push(Operation::Copy {
            source: "Partial".into(),
            destination: "Partial".into(),
            bytes: 16,
        });
        assert!(validate(&copy).is_err());
        let mut reset = m.clone();
        reset.reset_buffers.push("Partial".into());
        assert!(validate(&reset).is_err());
    }
    #[test]
    fn actual_launch_rows_and_whole_shared_addresses_bound_mapping() {
        let m = fixture();
        let rows = &m.sequence_workspaces["Partial"];
        let k = &m.kernels[0];
        let mut call = Invocation {
            operation: Operation::Kernel { name: "k".into() },
            sequence: Some(7),
            launch: None,
            views: BTreeMap::new(),
        };
        assert_eq!(
            rows.invocation_bytes("Partial", k, &call, 64).unwrap(),
            Some(64)
        );
        for n in [1, 3, 4] {
            call.launch = Some(crate::operators::dynamic::RowLaunch {
                grid: [n as u32, 1, 1],
                arguments: vec![(1, n)],
            });
            assert_eq!(
                rows.invocation_bytes("Partial", k, &call, 64).unwrap(),
                Some(n as usize * 16)
            );
        }
        for n in [0, -1, 5] {
            call.launch.as_mut().unwrap().arguments[0].1 = n;
            assert!(rows.invocation_bytes("Partial", k, &call, 64).is_err());
        }
        // Launch argument overrides are applied in order by the executor.
        call.launch.as_mut().unwrap().arguments = vec![(1, 1), (1, 5)];
        assert!(rows.invocation_bytes("Partial", k, &call, 64).is_err());
        call.launch.as_mut().unwrap().arguments = vec![(1, 1)];
        call.launch.as_mut().unwrap().arguments[0].1 = 1;
        for (buffer, offset) in [("Partial", 16), ("Other", 0)] {
            call.views.insert(
                "Partial".into(),
                crate::execution::BufferView {
                    buffer: buffer.into(),
                    offset,
                },
            );
            assert!(rows.invocation_bytes("Partial", k, &call, 64).is_err());
        }
        let mut overflow = rows.clone();
        overflow.bytes_per_row = usize::MAX;
        assert!(overflow.bytes(2, usize::MAX).is_err());
    }
}
