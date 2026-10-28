//! Upload independently owned encoder rows before capturing or replaying a packet.
use crate::{artifact::Result, backend::Executor};
use orinfer_model_sdk::encoder_rows::{Packet, Spec};

pub(crate) fn upload(e: &Executor, spec: &Spec, packet: &Packet) -> Result<()> {
    let encode = |rows: &[[u32; 2]]| -> Vec<u8> {
        rows.iter()
            .flat_map(|r| r.iter().flat_map(|v| v.to_le_bytes()))
            .collect()
    };
    e.upload_bytes(&spec.ranges, &encode(&packet.ranges))?;
    e.upload_bytes(&spec.segments, &encode(&packet.segments))?;
    e.upload_ids(&spec.count, &[packet.segments.len() as u32])
}
