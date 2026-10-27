//! Native multi-device collective checks with a coordinated preflight gate.
use super::{Communicator, Counts, Element, RootInfo};
use crate::{
    artifact::Result,
    backend::ascend::runtime::{Limits, Session},
};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    sync::mpsc,
    time::{Duration, Instant},
};

#[derive(Serialize)]
pub struct Report {
    ranks: Vec<Rank>,
    scope: &'static str,
}
#[derive(Serialize)]
struct Rank {
    rank: u32,
    device: u32,
    all_reduce_verified: bool,
    all_to_all_v_verified: bool,
    initialize_ms: f64,
}

pub fn run(devices: &[u32]) -> Result<Report> {
    if !(2..=8).contains(&devices.len())
        || devices.iter().copied().collect::<BTreeSet<_>>().len() != devices.len()
    {
        return Err("Select 2..8 distinct Ascend devices for HCCL validation".into());
    }
    let bootstrap = Session::new(devices[0], Limits::default())?;
    let root = RootInfo::new(&bootstrap, devices.len() as u32)?;
    let ranks = std::thread::scope(|scope| {
        let (ready_send, ready_receive) = mpsc::channel();
        let mut workers = vec![];
        for (rank, &device) in devices.iter().enumerate() {
            let (start, gate) = mpsc::channel();
            let ready = ready_send.clone();
            let root = root.clone();
            let worker = scope.spawn(move || {
                let prepared = Session::new(device, Limits::default()).and_then(|session| {
                    if session.memory_info()?.0 < 512 << 20 {
                        return Err("HCCL validation needs 512 MiB free HBM per device".into());
                    }
                    Ok(session)
                });
                ready
                    .send(prepared.as_ref().map(|_| ()).map_err(Clone::clone))
                    .map_err(|e| e.to_string())?;
                let session = prepared?;
                if !gate.recv().map_err(|e| e.to_string())? {
                    return Err("HCCL preflight failed on a peer".into());
                }
                validate_rank(&session, &root, rank as u32, device)
            });
            workers.push((worker, start));
        }
        drop(ready_send);
        let mut preflight = Ok(());
        for _ in devices {
            match ready_receive.recv_timeout(Duration::from_secs(30)) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    if preflight.is_ok() {
                        preflight = Err(error);
                    }
                }
                Err(error) => {
                    preflight = Err(format!("HCCL preflight: {error}"));
                    break;
                }
            }
        }
        for (_, start) in &workers {
            let _ = start.send(preflight.is_ok());
        }
        let mut results = vec![];
        for (worker, _) in workers {
            results.push(
                worker
                    .join()
                    .map_err(|_| "HCCL validation worker panicked".to_string())
                    .and_then(|r| r),
            );
        }
        preflight?;
        results.into_iter().collect::<Result<Vec<_>>>()
    })?;
    Ok(Report {
        ranks,
        scope: "Native HCCL AllReduce and variable-count AlltoAllV across device contexts; no model execution",
    })
}
fn count(source: u32, destination: u32) -> u64 {
    if (source + 2 * destination).is_multiple_of(4) {
        0
    } else {
        u64::from((source + destination) % 3 + 1)
    }
}
fn values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}
fn validate_rank(session: &Session, root: &RootInfo, rank: u32, device: u32) -> Result<Rank> {
    let start = Instant::now();
    let comm = Communicator::join(session, root, rank)?;
    let initialize_ms = start.elapsed().as_secs_f64() * 1000.;
    let a = session.allocate(128)?;
    let b = session.allocate(128)?;
    let input: Vec<u8> = (0..32)
        .flat_map(|_| (rank as f32 + 1.).to_le_bytes())
        .collect();
    let upload = session.upload(&a, 0, &input)?;
    session.wait_on_compute(&upload)?;
    comm.all_reduce_sum(session, &a, &b, 32, Element::Float32)?
        .wait()?;
    let expected = (root.ranks() * (root.ranks() + 1) / 2) as f32;
    if values(&session.download(&b, 0, 128)?)
        .iter()
        .any(|&v| v != expected)
    {
        return Err(format!("HCCL AllReduce mismatch on rank {rank}"));
    }
    let sends: Vec<u64> = (0..root.ranks()).map(|peer| count(rank, peer)).collect();
    let receives: Vec<u64> = (0..root.ranks()).map(|peer| count(peer, rank)).collect();
    let send = Counts::new(sends)?;
    let receive = Counts::new(receives)?;
    let input: Vec<f32> = (0..root.ranks())
        .flat_map(|peer| {
            (0..count(rank, peer)).map(move |i| (rank * 10000 + peer * 100 + i as u32) as f32)
        })
        .collect();
    let expected: Vec<f32> = (0..root.ranks())
        .flat_map(|peer| {
            (0..count(peer, rank)).map(move |i| (peer * 10000 + rank * 100 + i as u32) as f32)
        })
        .collect();
    let a = session.allocate((send.elements() as usize * 4).max(4))?;
    let b = session.allocate((receive.elements() as usize * 4).max(4))?;
    let input: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    if !input.is_empty() {
        session.wait_on_compute(&session.upload(&a, 0, &input)?)?;
    }
    comm.all_to_all_v(session, &a, &b, send, receive, Element::Float32)?
        .wait()?;
    if !expected.is_empty() && values(&session.download(&b, 0, expected.len() * 4)?) != expected {
        return Err(format!("HCCL AlltoAllV mismatch on rank {rank}"));
    }
    comm.check_async_error()?;
    Ok(Rank {
        rank,
        device,
        all_reduce_verified: true,
        all_to_all_v_verified: true,
        initialize_ms,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "Requires exclusive GPU experiment lock and ORINFER_ASCEND_TEST_DEVICES"]
    fn native_multi_device_collectives() {
        let devices = std::env::var("ORINFER_ASCEND_TEST_DEVICES")
            .expect("select explicit Ascend devices")
            .split(',')
            .map(|d| d.parse().unwrap())
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string(&super::run(&devices).unwrap()).unwrap()
        );
    }
}
