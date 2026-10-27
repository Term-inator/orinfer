use super::{Step, Worker, get, upload};
use crate::{artifact::Result, backend::ascend::runtime::Buffer};
use safetensors::Dtype;
use serde::Deserialize;
use std::{fs, path::Path, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Coordinates {
    pub base_cos: String,
    pub base_sin: String,
    pub long_cos: String,
    pub long_sin: String,
    pub source_cos: String,
    pub source_sin: String,
    pub slots: String,
    pub c2_slots: String,
    pub swa_indices: String,
    pub c1_indices: String,
    pub c2_indices: String,
    pub c1_lengths: String,
    pub c2_lengths: String,
    pub ring: String,
    pub premix: String,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Best {
    pub value: f32,
    pub token: u32,
}

/// Per-row-count execution accounting for one request segment. All seconds
/// are wall time on this rank's worker thread; counts are per forward.
#[derive(Default)]
pub(super) struct StepStats {
    forwards: usize,
    coordinate_s: f64,
    hash_s: f64,
    body_s: f64,
    head_s: f64,
    download_s: f64,
    graph: (usize, f64),
    reduce: (usize, f64),
    w4: (usize, f64),
    route: (usize, f64),
    route_dl: (usize, f64),
    route_cpu: (usize, f64),
    route_up: (usize, f64),
    route_grp: (usize, f64),
    lookup: (usize, f64),
    embed: (usize, f64),
}

fn ints(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
impl Worker {
    fn put(&self, name: &str, data: &[u8]) -> Result<()> {
        upload(&self.session, &get(&self.buffers, name)?, data)
    }
    fn coordinate(&self, names: &Coordinates, base: usize, rows: usize) -> Result<()> {
        if rows == 0 || base.checked_add(rows).is_none_or(|end| end > self.capacity) {
            return Err("Packet exceeds model context capacity".into());
        }
        for (asset, target, compressed) in [
            ("base_cos", &names.base_cos, false),
            ("base_sin", &names.base_sin, false),
            ("long_cos", &names.long_cos, false),
            ("long_sin", &names.long_sin, false),
            ("long_cos", &names.source_cos, true),
            ("long_sin", &names.source_sin, true),
        ] {
            let table = self.rope.get(asset).ok_or("RoPE asset missing")?;
            let mut data = Vec::with_capacity(rows * 128);
            for position in base..base + rows {
                let row = if compressed {
                    position / 2 * 2
                } else {
                    position
                };
                data.extend_from_slice(
                    table
                        .get(row * 128..row * 128 + 128)
                        .ok_or("RoPE position outside table")?,
                );
            }
            self.put(target, &data)?;
        }
        let slots = (base..base + rows).map(|p| p as i32).collect::<Vec<_>>();
        let c2_slots = (base..base + rows)
            .map(|p| if p % 2 == 1 { (p / 2) as i32 } else { -1 })
            .collect::<Vec<_>>();
        self.put(&names.slots, &ints(&slots))?;
        self.put(&names.c2_slots, &ints(&c2_slots))?;
        let mut swa = vec![-1; rows * 128];
        let mut metadata = vec![0; rows * 8];
        let mut mix = vec![0f32; rows * 8];
        for row in 0..rows {
            let position = base + row;
            for (lane, previous) in (position.saturating_sub(127)..=position).enumerate() {
                swa[row * 128 + lane] = previous as i32;
            }
            metadata[row * 8..row * 8 + 6].copy_from_slice(&[
                position as i32,
                1,
                if row == 0 { -1 } else { (row - 1) as i32 },
                1,
                (base + rows - 1) as i32,
                1,
            ]);
            mix[row * 8] = 1.0;
        }
        self.put(&names.swa_indices, &ints(&swa))?;
        self.put(&names.ring, &ints(&metadata))?;
        self.put(
            &names.premix,
            &mix.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        for (ratio, indices, lengths) in [
            (1, &names.c1_indices, &names.c1_lengths),
            (2, &names.c2_indices, &names.c2_lengths),
        ] {
            let capacity = self.capacity / ratio;
            let mut ids = vec![-1; rows * capacity];
            let mut lens = vec![0; rows];
            for row in 0..rows {
                let count = (base + row + 1) / ratio;
                lens[row] = count as i32;
                for slot in 0..count {
                    ids[row * capacity + slot] = slot as i32;
                }
            }
            self.put(indices, &ints(&ids))?;
            self.put(lengths, &ints(&lens))?;
        }
        Ok(())
    }
    pub(super) fn forward(
        &mut self,
        history: &[u32],
        tokens: &[u32],
        logits: bool,
        trace: Option<&Path>,
    ) -> Result<Option<Best>> {
        let rows = tokens.len();
        let execution = self
            .packets
            .remove(&rows)
            .ok_or_else(|| format!("No A2 packet for {rows} rows"))?;
        let result = (|| {
            let t = Instant::now();
            self.coordinate(&execution.packet.coordinates, history.len(), rows)?;
            let coordinate = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut hashes = vec![];
            let mut valid = vec![];
            self.hasher
                .hash_into(history, tokens, &mut hashes, &mut valid)?;
            let hash = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for step in &execution.body {
                self.step(step, tokens, &hashes, &valid, history.len(), trace)?;
            }
            let body = t.elapsed().as_secs_f64();
            if !logits {
                self.session.synchronize()?;
                let stats = self.steps.entry(rows).or_default();
                stats.forwards += 1;
                stats.coordinate_s += coordinate;
                stats.hash_s += hash;
                stats.body_s += body;
                return Ok(None);
            }
            let t = Instant::now();
            for step in &execution.head {
                self.step(step, tokens, &hashes, &valid, history.len(), trace)?;
            }
            let head = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let raw_value =
                self.session
                    .download(&get(&self.buffers, &execution.packet.best_value)?, 0, 4)?;
            let raw_index =
                self.session
                    .download(&get(&self.buffers, &execution.packet.best_index)?, 0, 4)?;
            let download = t.elapsed().as_secs_f64();
            let value = f32::from_le_bytes(raw_value.try_into().unwrap());
            let local = i32::from_le_bytes(raw_index.try_into().unwrap());
            if !value.is_finite() || !(0..16160).contains(&local) {
                return Err("Nonfinite/invalid native model output".into());
            }
            let stats = self.steps.entry(rows).or_default();
            stats.forwards += 1;
            stats.coordinate_s += coordinate;
            stats.hash_s += hash;
            stats.body_s += body;
            stats.head_s += head;
            stats.download_s += download;
            Ok(Some(Best {
                value,
                token: execution.packet.vocab_first + local as u32,
            }))
        })();
        self.packets.insert(rows, execution);
        result
    }
    /// Prints and clears per-row-count step accounting for this rank.
    pub(super) fn report_steps(&mut self) {
        for (rows, s) in std::mem::take(&mut self.steps) {
            if s.forwards == 0 {
                continue;
            }
            let n = s.forwards as f64;
            let ms = |v: f64| v * 1000.0 / n;
            let each = |c: (usize, f64)| (c.0 as f64 / n, c.1 * 1000.0 / n);
            let (gn, gt) = each(s.graph);
            let (rn, rt) = each(s.reduce);
            let (wn, wt) = each(s.w4);
            let (un, ut) = each(s.route);
            let (ln, lt) = each(s.lookup);
            eprintln!(
                "A2 rank {} rows={rows} x{} avg: coordinate={:.2}ms hash={:.3}ms body={:.2}ms head={:.2}ms dl={:.2}ms | graph {:.0}x{:.3}ms reduce {:.0}x{:.3}ms w4 {:.0}x{:.3}ms route {:.0}x{:.2}ms(dl={:.2} cpu={:.3} up={:.2} grp={:.2}) lookup {:.0}x{:.2}ms embed={:.3}ms",
                self.rank,
                s.forwards,
                ms(s.coordinate_s),
                ms(s.hash_s),
                ms(s.body_s),
                ms(s.head_s),
                ms(s.download_s),
                gn,
                gt,
                rn,
                rt,
                wn,
                wt,
                un,
                ut,
                ms(s.route_dl.1),
                ms(s.route_cpu.1),
                ms(s.route_up.1),
                ms(s.route_grp.1),
                ln,
                lt,
                ms(s.embed.1),
            );
        }
    }
    fn step(
        &mut self,
        step: &Step,
        tokens: &[u32],
        hashes: &[i64],
        valid: &[bool],
        position: usize,
        trace: Option<&Path>,
    ) -> Result<()> {
        // Synchronized profiling isolates device execution per step type by
        // fencing after every submission; it changes timing, not values.
        if std::env::var_os("ORINFER_A2_SYNC_PROFILE").is_some() {
            let t = Instant::now();
            let submission = match step {
                Step::Graph(program) => program.enqueue(&self.session).map(|_| ()),
                Step::W4(name) => self
                    .w4
                    .get(name)
                    .ok_or("W4 binding missing".to_string())
                    .and_then(|plan| plan.enqueue(&self.session).map(|_| ())),
                _ => Ok(()),
            };
            let result = submission.and_then(|()| self.session.synchronize());
            let elapsed = t.elapsed().as_secs_f64();
            let stats = self.steps.entry(tokens.len()).or_default();
            match step {
                Step::Graph(_) => {
                    stats.graph.0 += 1;
                    stats.graph.1 += elapsed;
                }
                Step::W4(_) => {
                    stats.w4.0 += 1;
                    stats.w4.1 += elapsed;
                }
                _ => {}
            }
            return result;
        }
        let t = Instant::now();
        let result = self.step_inner(step, tokens, hashes, valid, position, trace);
        let elapsed = t.elapsed().as_secs_f64();
        let rows = tokens.len();
        let stats = self.steps.entry(rows).or_default();
        match step {
            Step::Graph(_) => {
                stats.graph.0 += 1;
                stats.graph.1 += elapsed;
            }
            Step::W4(_) => {}
            Step::Route { .. } => {
                stats.route.0 += 1;
                stats.route.1 += elapsed;
            }
            Step::Lookup { .. } => {
                stats.lookup.0 += 1;
                stats.lookup.1 += elapsed;
            }
            Step::Embed(_) => {
                stats.embed.0 += 1;
                stats.embed.1 += elapsed;
            }
            Step::Snapshot { .. } => {}
        }
        result
    }
    fn step_inner(
        &mut self,
        step: &Step,
        tokens: &[u32],
        hashes: &[i64],
        valid: &[bool],
        position: usize,
        trace: Option<&Path>,
    ) -> Result<()> {
        match step {
            Step::Graph(program) => {
                program.enqueue(&self.session)?;
            }
            Step::W4(name) => {
                self.w4
                    .get(name)
                    .ok_or("W4 binding missing")?
                    .enqueue(&self.session)?;
            }
            Step::Route {
                indices,
                gather,
                restore,
                groups,
                rows,
            } => {
                let t = Instant::now();
                let ids = self.session.download(indices, 0, indices.bytes())?;
                let dl = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let ids = ids
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|x| i32::from_le_bytes(*x))
                    .collect::<Vec<_>>();
                if ids.len() != rows * 8
                    || (0..*rows).any(|row| {
                        ids[row * 8..row * 8 + 6]
                            .iter()
                            .any(|id| !(0..384).contains(id))
                    })
                {
                    return Err(format!(
                        "Invalid expert routes for rank {} in {groups:?}",
                        self.rank
                    ));
                }

                let mut selected = vec![-1; rows * 6];
                let mut map = vec![-1; rows * 8];
                let mut ends = vec![0i64; 48];
                let mut count = 0;
                for (local, end) in ends.iter_mut().enumerate() {
                    let expert = (self.rank as usize * 48 + local) as i32;
                    for row in 0..*rows {
                        for route in 0..6 {
                            if ids[row * 8 + route] == expert {
                                selected[count] = row as i32;
                                map[row * 8 + route] = count as i32;
                                count += 1;
                            }
                        }
                    }
                    *end = count as i64;
                }
                let cpu = t.elapsed().as_secs_f64();
                // Unreferenced zero rows pad the fixed expert packet. Their
                // final bank is never restored into a real token output.
                ends[47] = (rows * 6) as i64;
                let t = Instant::now();
                upload(&self.session, gather, &ints(&selected))?;
                upload(&self.session, restore, &ints(&map))?;
                let up = t.elapsed().as_secs_f64();
                let t = Instant::now();
                for name in groups {
                    self.w4
                        .get(name)
                        .ok_or("Routed W4 binding missing")?
                        .update_groups(&self.session, &ends)?;
                }
                let grp = t.elapsed().as_secs_f64();
                let stats = self.steps.entry(*rows).or_default();
                stats.route_dl.0 += 1;
                stats.route_dl.1 += dl;
                stats.route_cpu.0 += 1;
                stats.route_cpu.1 += cpu;
                stats.route_up.0 += 1;
                stats.route_up.1 += up;
                stats.route_grp.0 += 1;
                stats.route_grp.1 += grp;
            }
            Step::Embed(destination) => {
                let embedding = self.checkpoint.read("embed.weight")?;
                if embedding.dtype != Dtype::BF16 || embedding.shape != [129280, 5120] {
                    return Err("Embedding checkpoint layout differs".into());
                }
                let mut values = Vec::with_capacity(tokens.len() * 10240);
                for &token in tokens {
                    let start = token as usize * 10240;
                    values.extend_from_slice(
                        embedding
                            .data
                            .get(start..start + 10240)
                            .ok_or("Embedding token out of range")?,
                    );
                }
                upload(&self.session, destination, &values)?;
            }
            Step::Lookup {
                layer,
                values,
                scales,
                mask,
            } => self.lookup(*layer, tokens.len(), hashes, valid, values, scales, mask)?,
            Step::Snapshot { name, buffer } => {
                if let Some(base) = trace.filter(|_| self.rank == 0) {
                    fs::create_dir_all(base).map_err(|e| e.to_string())?;
                    let data = self.session.download(buffer, 0, buffer.bytes())?;
                    fs::write(base.join(format!("p{position}-{name}.bin")), data)
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn lookup(
        &self,
        layer: usize,
        rows: usize,
        hashes: &[i64],
        valid: &[bool],
        values: &Buffer,
        scales: &Buffer,
        mask: &Buffer,
    ) -> Result<()> {
        let layer_index = self
            .hasher
            .layout()
            .layers
            .iter()
            .position(|l| l.layer == layer)
            .ok_or("Engram layer missing")?;
        let width = self.hasher.values_per_token();
        let mut ids = Vec::with_capacity(rows * 24);
        for (row, &active) in valid.iter().enumerate() {
            for lane in 0..24 {
                ids.push(if active {
                    hashes[row * width + layer_index * 24 + lane] as u64
                } else {
                    0
                });
            }
        }
        let table = self.tables.get(&layer).ok_or("Engram table missing")?;
        let mut packed = vec![];
        table.gather_into(&ids, &mut packed, rows * 24 * 288)?;
        let mut q = Vec::with_capacity(rows * 24 * 256);
        let mut scale = Vec::with_capacity(rows * 24 * 32);
        for data in packed.as_chunks::<288>().0.iter() {
            q.extend_from_slice(&data[..256]);
            scale.extend_from_slice(&data[256..]);
        }
        upload(&self.session, values, &q)?;
        upload(&self.session, scales, &scale)?;
        upload(
            &self.session,
            mask,
            &valid.iter().map(|&v| u8::from(v)).collect::<Vec<_>>(),
        )?;
        Ok(())
    }
}
