//! Complete independent sequences sharing projections and bounded row workspace.
use crate::artifact::{Access, Dtype, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub ranges: String,
    pub segments: String,
    pub count: String,
    pub max_sequences: usize,
    pub alignment: usize,
}
impl Spec {
    pub fn validate(&self, m: &crate::model::Manifest) -> Result<()> {
        if self.alignment != 32
            || !(1..=128).contains(&self.max_sequences)
            || !m.chunk_tokens.is_multiple_of(self.alignment)
            || [&self.ranges, &self.segments, &self.count]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != 3
        {
            return Err("Invalid independent encoder row geometry".into());
        }
        for (name, shape) in [
            (&self.ranges, vec![m.chunk_tokens, 2]),
            (&self.segments, vec![self.max_sequences, 2]),
            (&self.count, vec![1]),
        ] {
            let b = m
                .buffers
                .iter()
                .find(|b| b.name == *name)
                .ok_or("Missing independent encoder row buffer")?;
            if b.dtype != Dtype::I32
                || b.shape != shape
                || b.data.is_some()
                || b.access == Access::Read
            {
                return Err(format!("Invalid independent encoder row buffer {name}"));
            }
        }
        Ok(())
    }

    pub fn packets(&self, lengths: &[usize], capacity: usize) -> Result<Vec<Packet>> {
        if self.alignment != 32
            || !(1..=128).contains(&self.max_sequences)
            || !(32..=32768).contains(&capacity)
            || !capacity.is_multiple_of(self.alignment)
            || lengths.iter().any(|&n| n == 0 || n > capacity)
        {
            return Err("Invalid independent encoder input lengths".into());
        }
        let mut result = Vec::new();
        let mut packet = Packet::default();
        for (index, &n) in lengths.iter().enumerate() {
            let padded = n.div_ceil(self.alignment) * self.alignment;
            if packet.rows + padded > capacity || packet.segments.len() == self.max_sequences {
                result.push(packet);
                packet = Packet {
                    first: index,
                    ..Default::default()
                };
            }
            let start = packet.rows;
            let end = start + n;
            packet.segments.push([start as u32, end as u32]);
            packet
                .ranges
                .extend(std::iter::repeat_n([start as u32, end as u32], padded));
            packet.rows += padded;
        }
        if packet.rows > 0 {
            result.push(packet);
        }
        Ok(result)
    }
}

#[derive(Default, Debug)]
pub struct Packet {
    pub first: usize,
    pub rows: usize,
    pub segments: Vec<[u32; 2]>,
    pub ranges: Vec<[u32; 2]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_boundaries_survive_padding_and_packet_splits() {
        let s = Spec {
            ranges: "r".into(),
            segments: "s".into(),
            count: "c".into(),
            max_sequences: 2,
            alignment: 32,
        };
        let packets = s.packets(&[1, 33, 65, 31, 128], 128).unwrap();
        assert_eq!(packets.len(), 3);
        assert_eq!(packets[0].segments, [[0, 1], [32, 65]]);
        assert_eq!(packets[0].ranges[31], [0, 1]);
        assert_eq!(packets[0].ranges[95], [32, 65]);
        assert_eq!(packets[1].first, 2);
        assert_eq!(packets[1].segments, [[0, 65], [96, 127]]);
        assert_eq!(packets[2].first, 4);
        for lengths in [&[0][..], &[129][..], &[usize::MAX][..]] {
            assert!(s.packets(lengths, 128).is_err());
        }
        assert!(s.packets(&[32], usize::MAX - 31).is_err());
    }
}
