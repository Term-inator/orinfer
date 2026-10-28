//! Decode explicitly typed result buffers without device dependencies.
use crate::artifact::Dtype;
pub(crate) fn floats(raw: &[u8], dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        Dtype::F16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half_to_float(u16::from_le_bytes(*b)))
            .collect(),
        Dtype::Bf16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect(),
        _ => unreachable!("validated floating point output"),
    }
}
pub(super) fn half_to_float(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}
