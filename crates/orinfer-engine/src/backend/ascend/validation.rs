//! Small, bounded native checks. This does not load a model or reset a device.
use super::{
    inventory,
    operators::{GroupedInputs, GroupedShape, GroupedW4A8},
    runtime::{Limits, Session},
};
use crate::artifact::Result;
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
pub struct Report {
    device: u32,
    soc: Option<String>,
    hbm_free_before: usize,
    hbm_free_after: usize,
    transfer_roundtrip_verified: bool,
    changed_input_reuse_verified: bool,
    changed_input_graph_verified: bool,
    cases: Vec<Case>,
    scope: &'static str,
}
#[derive(Serialize)]
struct Case {
    rows: usize,
    experts: usize,
    input: usize,
    output: usize,
    workspace_bytes: usize,
    max_abs_error: f32,
    prepare_ms: f64,
    reused_execute_wall_ms: f64,
    graph_replay_wall_ms: f64,
}

pub fn run(device: u32) -> Result<Report> {
    let session = Session::new(
        device,
        Limits {
            device_bytes: 512 << 20,
            ..Limits::default()
        },
    )?;
    let soc = inventory(&session.context.driver)?.soc;
    let (hbm_free_before, _) = session.memory_info()?;
    if hbm_free_before < 256 << 20 {
        return Err("Native validation needs at least 256 MiB free HBM".into());
    }
    let buffer = session.allocate(1024)?;
    let mut source: Vec<u8> = (0..1024).map(|i| i as u8).collect();
    let expected = source.clone();
    let upload = session.upload(&buffer, 0, &source)?;
    source.fill(0);
    session.wait_on_compute(&upload)?.wait()?;
    if session.download(&buffer, 0, 1024)? != expected {
        return Err("Ascend async transfer roundtrip failed".into());
    }
    drop((upload, buffer));
    let mut workspace = None;
    let cases = vec![
        matmul(&session, 1, &[1], &mut workspace)?,
        matmul(&session, 16, &[8, 16], &mut workspace)?,
        matmul(&session, 3, &[0, 1, 3], &mut workspace)?,
        matmul(
            &session,
            96,
            &(1..=48).map(|i| i * 2).collect::<Vec<_>>(),
            &mut workspace,
        )?,
        matmul_k(
            &session,
            96,
            &(1..=48).map(|i| i * 2).collect::<Vec<_>>(),
            &mut workspace,
            5120,
        )?,
    ];
    drop(workspace);
    session.synchronize()?;
    if session.allocated_bytes() != 0 || session.pinned_bytes() != 0 {
        return Err("Ascend validation left live allocations".into());
    }
    let (hbm_free_after, _) = session.memory_info()?;
    Ok(Report {
        device,
        soc,
        hbm_free_before,
        hbm_free_after,
        transfer_roundtrip_verified: true,
        changed_input_reuse_verified: true,
        changed_input_graph_verified: true,
        cases,
        scope: "Bounded native ACL transfers, grouped W4A8 and changed-input Graph replay; no model or distributed inference",
    })
}

fn matmul(
    session: &Session,
    m: usize,
    ends: &[i64],
    workspace: &mut Option<super::runtime::Buffer>,
) -> Result<Case> {
    matmul_k(session, m, ends, workspace, 256)
}
fn matmul_k(
    session: &Session,
    m: usize,
    ends: &[i64],
    workspace: &mut Option<super::runtime::Buffer>,
    k: usize,
) -> Result<Case> {
    let (n, e) = (64usize, ends.len());
    let weights: Vec<i8> = (0..e * k * n)
        .map(|i| {
            let row = i / n % k;
            let column = i % n;
            [0, 1, -1, 2, -2, 3, -3, 0][(row + column + i / (k * n)) % 8]
                + if k == 256 {
                    (i / (k * n) % 4 + 1) as i8
                } else {
                    0
                }
        })
        .collect();
    let packed: Vec<u8> = weights
        .as_chunks::<2>()
        .0
        .iter()
        .map(|w| (w[0] as u8 & 15) | ((w[1] as u8 & 15) << 4))
        .collect();
    let scales: Vec<f32> = (0..e * n).map(|i| [0.125, 0.25, 0.5][i % 3]).collect();
    let scale_data: Vec<u8> = scales
        .iter()
        .flat_map(|v| u64::from(v.to_bits()).to_le_bytes())
        .collect();
    let token_scales: Vec<f32> = (0..m).map(|i| [0.5, 1.0, 2.0][i % 3]).collect();
    let assist: Vec<u8> = (0..e * n)
        .flat_map(|i| {
            let (expert, column) = (i / n, i % n);
            let sum: i32 = (0..k)
                .map(|row| i32::from(weights[(expert * k + row) * n + column]))
                .sum();
            (8.0 * sum as f32 * scales[i]).to_le_bytes()
        })
        .collect();
    let a = session.allocate(m * k)?;
    let w = session.allocate(packed.len())?;
    let ws = session.allocate(scale_data.len())?;
    let bias = session.allocate(assist.len())?;
    let xs = session.allocate(m * 4)?;
    let output_storage = session.allocate(m * n * 2 * (e + 1))?;
    let y = output_storage.slice(0, m * n * 2)?;
    let output_guard = output_storage.slice(m * n * 2, m * n * 2 * e)?;
    for (buffer, data) in [
        (&w, packed.as_slice()),
        (&ws, &scale_data),
        (&bias, &assist),
    ] {
        session.upload(buffer, 0, data)?.wait()?;
    }
    let xscale: Vec<u8> = token_scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    session.upload(&xs, 0, &xscale)?.wait()?;
    session.fill(&a, 0)?.wait()?;
    let start = Instant::now();
    let plan = GroupedW4A8::prepare_with_workspace(
        session,
        GroupedShape {
            rows: m,
            input: k,
            output: n,
            experts: e,
        },
        ends,
        GroupedInputs {
            activation: &a,
            weight: &w,
            weight_scale: &ws,
            assist: &bias,
            token_scale: &xs,
            output: &y,
        },
        workspace.as_ref(),
    )?;
    if workspace.is_none() {
        *workspace = plan.workspace();
    }
    let prepare_ms = start.elapsed().as_secs_f64() * 1000.;
    plan.enqueue(session)?.wait()?;
    let graph = plan.capture(session)?;
    let mut max_abs_error = 0f32;
    let mut reused_execute_wall_ms = 0.;
    let mut graph_replay_wall_ms = 0.;
    for pass in 0..4 {
        let active_ends = if pass % 2 == 0 {
            ends.to_vec()
        } else {
            vec![m as i64; e]
        };
        plan.update_groups(session, &active_ends)?;
        let input: Vec<i8> = (0..m * k)
            .map(|i| ((i * 3 + i / k + pass * 7) % 17) as i8 - 8)
            .collect();
        let input_data: Vec<u8> = input.iter().map(|v| *v as u8).collect();
        let upload = session.upload(&a, 0, &input_data)?;
        session.wait_on_compute(&upload)?.wait()?;
        session.fill(&y, 0xa5)?.wait()?;
        session.fill(&output_guard, 0xa7)?.wait()?;
        let start = Instant::now();
        if pass < 2 {
            plan.enqueue(session)?.wait()?;
            reused_execute_wall_ms = start.elapsed().as_secs_f64() * 1000.;
        } else {
            graph.enqueue(session)?.wait()?;
            graph_replay_wall_ms = start.elapsed().as_secs_f64() * 1000.;
        }
        let result = session.download(&y, 0, m * n * 2)?;
        if session
            .download(&output_guard, 0, output_guard.bytes())?
            .iter()
            .any(|&byte| byte != 0xa7)
        {
            return Err(format!(
                "Grouped W4A8 wrote outside its output: M={m} E={e} pass={pass}"
            ));
        }
        for row in 0..m {
            let expert = active_ends
                .iter()
                .position(|&end| row < (end as usize))
                .ok_or("Missing validation expert")?;
            for column in 0..n {
                let dot: i32 = (0..k)
                    .map(|i| {
                        i32::from(input[row * k + i])
                            * i32::from(weights[(expert * k + i) * n + column])
                    })
                    .sum();
                let expected = half::bf16::from_f32(
                    dot as f32 * scales[expert * n + column] * token_scales[row],
                )
                .to_f32();
                let offset = (row * n + column) * 2;
                let actual =
                    half::bf16::from_bits(u16::from_le_bytes([result[offset], result[offset + 1]]))
                        .to_f32();
                let error = (expected - actual).abs();
                if !actual.is_finite() || error > 0.02 {
                    return Err(format!(
                        "Native W4A8 mismatch M={m} E={e} pass={pass} row={row} column={column}: expected {expected}, got {actual}"
                    ));
                }
                max_abs_error = max_abs_error.max(error);
            }
        }
    }
    Ok(Case {
        rows: m,
        experts: e,
        input: k,
        output: n,
        workspace_bytes: plan.workspace_bytes(),
        max_abs_error,
        prepare_ms,
        reused_execute_wall_ms,
        graph_replay_wall_ms,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "Requires exclusive GPU experiment lock and ORINFER_ASCEND_TEST_DEVICE"]
    fn native_w4a8_and_changed_input_graph_replay() {
        let device = std::env::var("ORINFER_ASCEND_TEST_DEVICE")
            .expect("select an explicit Ascend test device")
            .parse()
            .unwrap();
        let report = super::run(device).unwrap();
        assert!(report.transfer_roundtrip_verified);
        assert!(report.changed_input_reuse_verified);
        assert!(report.changed_input_graph_verified);
        assert_eq!(report.cases.len(), 5);
        println!("{}", serde_json::to_string(&report).unwrap());
    }
}
