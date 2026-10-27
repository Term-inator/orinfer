//! Original-checkpoint single-sequence A2 execution and latency measurements.
mod benchmark;
mod packet;
mod weights;

use super::{
    hccl::{Communicator, Element, RootInfo},
    kernels::{Kernel, Program},
    operators::{GroupedInputs, GroupedShape, GroupedW4A8},
    runtime::{Buffer, Limits, Session},
};
use crate::{
    artifact::{FileIdentity, Result, read_identity, resolve_file},
    host_rows::{Int8Rows, Shard},
    weights::checkpoint::Checkpoint,
};
use orinfer_model_sdk::conditional_memory::{HashLayout, Hasher};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub use benchmark::{Report, run};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    schema_version: u32,
    config: FileIdentity,
    index: FileIdentity,
    engram: FileIdentity,
    capacity: usize,
    layers: usize,
    ranks: Vec<Rank>,
    rope: BTreeMap<String, FileIdentity>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Rank {
    rank: u32,
    buffers: Vec<Definition>,
    packets: Vec<Packet>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Definition {
    name: String,
    bytes: usize,
    view: Option<View>,
    weight: Option<weights::Weight>,
    data: Option<FileIdentity>,
    prepared_weight: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct View {
    parent: String,
    offset: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    rows: usize,
    coordinates: packet::Coordinates,
    body: Vec<Instruction>,
    head: Vec<Instruction>,
    reset: Vec<String>,
    best_value: String,
    best_index: String,
    vocab_first: u32,
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Instruction {
    Kernel {
        manifest: String,
        buffers: Vec<String>,
    },
    Reduce {
        source: String,
        destination: String,
        elements: u64,
        fp32: bool,
    },
    W4 {
        name: String,
        rows: usize,
        input: usize,
        output: usize,
        experts: usize,
        buffers: [String; 6],
        workspace: String,
    },
    Route {
        indices: String,
        gather: String,
        restore: String,
        groups: Vec<String>,
        rows: usize,
    },
    Embed {
        destination: String,
    },
    Lookup {
        layer: usize,
        values: String,
        scales: String,
        mask: String,
    },
    Snapshot {
        name: String,
        buffer: String,
    },
}
enum Step {
    Graph(super::graph::Graph),
    Reduce {
        source: Buffer,
        destination: Buffer,
        elements: u64,
        element: Element,
    },
    W4(String),
    Route {
        indices: Buffer,
        gather: Buffer,
        restore: Buffer,
        groups: Vec<String>,
        rows: usize,
    },
    Embed(Buffer),
    Lookup {
        layer: usize,
        values: Buffer,
        scales: Buffer,
        mask: Buffer,
    },
    Snapshot {
        name: String,
        buffer: Buffer,
    },
}
struct Execution {
    packet: Packet,
    body: Vec<Step>,
    head: Vec<Step>,
}
struct Worker {
    session: Session,
    comm: Communicator,
    buffers: BTreeMap<String, Buffer>,
    w4: BTreeMap<String, GroupedW4A8>,
    packets: BTreeMap<usize, Execution>,
    checkpoint: Checkpoint,
    hasher: Hasher,
    tables: BTreeMap<usize, Int8Rows>,
    rope: BTreeMap<String, Vec<u8>>,
    capacity: usize,
    rank: u32,
}

fn upload(session: &Session, buffer: &Buffer, data: &[u8]) -> Result<()> {
    if data.len() != buffer.bytes() {
        return Err("Model upload byte extent differs".into());
    }
    for (chunk, bytes) in data.chunks(8 << 20).enumerate() {
        session.upload(buffer, chunk * (8 << 20), bytes)?.wait()?;
    }
    Ok(())
}
fn get(buffers: &BTreeMap<String, Buffer>, name: &str) -> Result<Buffer> {
    buffers
        .get(name)
        .cloned()
        .ok_or_else(|| format!("Model buffer is not defined: {name}"))
}
impl Worker {
    fn load(
        session: Session,
        root: &RootInfo,
        rank: Rank,
        model: &Model,
        model_dir: &Path,
        package: &Path,
        layout: HashLayout,
    ) -> Result<Self> {
        let comm = Communicator::join(&session, root, rank.rank)?;
        let mut checkpoint = Checkpoint::open(model_dir, &model.index.file)?;
        let mut buffers = BTreeMap::new();
        let needed = rank
            .buffers
            .iter()
            .filter(|b| b.view.is_none())
            .try_fold(0usize, |n, b| {
                n.checked_add(b.bytes).ok_or("Model byte estimate overflow")
            })?;
        if needed > session.memory_info()?.0.saturating_sub(1 << 30) {
            return Err(format!(
                "Rank {} needs {needed} bytes plus 1 GiB headroom",
                rank.rank
            ));
        }
        for definition in rank.buffers {
            if buffers.contains_key(&definition.name) || definition.bytes == 0 {
                return Err("Duplicate/empty model buffer".into());
            }
            let buffer = if let Some(view) = definition.view {
                if definition.weight.is_some()
                    || definition.data.is_some()
                    || definition.prepared_weight.is_some()
                {
                    return Err("Model view cannot own weight data".into());
                }
                get(&buffers, &view.parent)?.slice(view.offset, definition.bytes)?
            } else {
                let buffer = session.allocate(definition.bytes)?;
                let data = if let Some(file) = definition.prepared_weight {
                    let path = resolve_file(package, &file)?;
                    let data = std::fs::read(path).map_err(|e| e.to_string())?;
                    if data.len() != definition.bytes {
                        return Err("Prepared weight byte extent differs".into());
                    }
                    Some(data)
                } else if let Some(weight) = definition.weight {
                    Some(weights::load(&mut checkpoint, &weight)?)
                } else if let Some(identity) = definition.data {
                    Some(read_identity(package, &identity)?)
                } else {
                    None
                };
                if let Some(data) = data {
                    upload(&session, &buffer, &data)?;
                } else {
                    session.fill(&buffer, 0)?.wait()?;
                }
                buffer
            };
            buffers.insert(definition.name, buffer);
        }
        let mut tables = BTreeMap::new();
        for layer in &layout.layers {
            let prefix = format!("layers.{}.engram.embed", layer.layer);
            tables.insert(
                layer.layer,
                Int8Rows::open(
                    &checkpoint.path(&format!("{prefix}.weight"))?,
                    &format!("{prefix}.weight"),
                    &checkpoint.path(&format!("{prefix}.scale"))?,
                    &format!("{prefix}.scale"),
                    Shard {
                        first: 0,
                        rows: layer.rows,
                    },
                )?,
            );
        }
        let rope = model
            .rope
            .iter()
            .map(|(name, id)| Ok((name.clone(), read_identity(package, id)?)))
            .collect::<Result<_>>()?;
        let mut worker = Self {
            session,
            comm,
            buffers,
            w4: BTreeMap::new(),
            packets: BTreeMap::new(),
            checkpoint,
            hasher: Hasher::new(layout)?,
            tables,
            rope,
            capacity: model.capacity,
            rank: rank.rank,
        };
        for packet in rank.packets {
            let body = worker.compile(&packet.body, package)?;
            let head = worker.compile(&packet.head, package)?;
            if worker
                .packets
                .insert(packet.rows, Execution { packet, body, head })
                .is_some()
            {
                return Err("Duplicate packet row count".into());
            }
        }
        Ok(worker)
    }
    fn compile(&mut self, instructions: &[Instruction], base: &Path) -> Result<Vec<Step>> {
        let mut steps = vec![];
        let mut pending = vec![];
        for instruction in instructions {
            if let Instruction::Kernel { manifest, buffers } = instruction {
                let operands = buffers
                    .iter()
                    .map(|n| get(&self.buffers, n))
                    .collect::<Result<Vec<_>>>()?;
                pending.push(Kernel::load(
                    &self.session,
                    &resolve_file(base, manifest)?,
                    &operands,
                )?);
                continue;
            }
            if !pending.is_empty() {
                let program = Program::new(&self.session, &pending.iter().collect::<Vec<_>>())?;
                program.enqueue(&self.session)?.wait()?;
                steps.push(Step::Graph(program.capture(&self.session)?));
                pending.clear();
            }
            steps.push(match instruction {
                Instruction::Reduce {
                    source,
                    destination,
                    elements,
                    fp32,
                } => Step::Reduce {
                    source: get(&self.buffers, source)?,
                    destination: get(&self.buffers, destination)?,
                    elements: *elements,
                    element: if *fp32 {
                        Element::Float32
                    } else {
                        Element::Bfloat16
                    },
                },
                Instruction::W4 {
                    name,
                    rows,
                    input,
                    output,
                    experts,
                    buffers,
                    workspace,
                } => {
                    let b = buffers
                        .iter()
                        .map(|n| get(&self.buffers, n))
                        .collect::<Result<Vec<_>>>()?;
                    let mut ends = vec![0; *experts];
                    *ends.last_mut().ok_or("Empty W4 expert bank")? = *rows as i64;
                    let plan = GroupedW4A8::prepare_with_workspace(
                        &self.session,
                        GroupedShape {
                            rows: *rows,
                            input: *input,
                            output: *output,
                            experts: *experts,
                        },
                        &ends,
                        GroupedInputs {
                            activation: &b[0],
                            weight: &b[1],
                            weight_scale: &b[2],
                            assist: &b[3],
                            token_scale: &b[4],
                            output: &b[5],
                        },
                        Some(&get(&self.buffers, workspace)?),
                    )?;
                    if self.w4.insert(name.clone(), plan).is_some() {
                        return Err("Duplicate W4 binding".into());
                    }
                    Step::W4(name.clone())
                }
                Instruction::Route {
                    indices,
                    gather,
                    restore,
                    groups,
                    rows,
                } => Step::Route {
                    indices: get(&self.buffers, indices)?,
                    gather: get(&self.buffers, gather)?,
                    restore: get(&self.buffers, restore)?,
                    groups: groups.clone(),
                    rows: *rows,
                },
                Instruction::Embed { destination } => Step::Embed(get(&self.buffers, destination)?),
                Instruction::Lookup {
                    layer,
                    values,
                    scales,
                    mask,
                } => Step::Lookup {
                    layer: *layer,
                    values: get(&self.buffers, values)?,
                    scales: get(&self.buffers, scales)?,
                    mask: get(&self.buffers, mask)?,
                },
                Instruction::Snapshot { name, buffer } => Step::Snapshot {
                    name: name.clone(),
                    buffer: get(&self.buffers, buffer)?,
                },
                Instruction::Kernel { .. } => unreachable!(),
            });
        }
        if !pending.is_empty() {
            let program = Program::new(&self.session, &pending.iter().collect::<Vec<_>>())?;
            program.enqueue(&self.session)?.wait()?;
            steps.push(Step::Graph(program.capture(&self.session)?));
        }
        Ok(steps)
    }
    fn reset(&self) -> Result<()> {
        for execution in self.packets.values() {
            for name in &execution.packet.reset {
                self.session.fill(&get(&self.buffers, name)?, 0)?.wait()?;
            }
        }
        Ok(())
    }
}

fn load_spec(path: &Path, model_dir: &Path) -> Result<(Model, PathBuf, HashLayout)> {
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    let base = path.parent().ok_or("Model plan has no parent")?.to_owned();
    let model: Model = crate::model::read(&path)?;
    if model.schema_version != 1
        || model.layers != 40
        || model.ranks.len() != 8
        || !(128..=16384).contains(&model.capacity)
        || !model.capacity.is_multiple_of(32)
    {
        return Err("Expected complete 40-layer TP8 A2 plan with capacity <=16384".into());
    }
    read_identity(model_dir, &model.config)?;
    read_identity(model_dir, &model.index)?;
    let layout: HashLayout =
        serde_json::from_slice(&read_identity(&base, &model.engram)?).map_err(|e| e.to_string())?;
    if layout.config_sha256 != model.config.sha256 {
        return Err("Engram/config identity differs".into());
    }
    if crate::artifact::sha256(
        &std::fs::read(model_dir.join("tokenizer.json")).map_err(|e| e.to_string())?,
    ) != layout.tokenizer_sha256
    {
        return Err("Engram/tokenizer identity differs".into());
    }
    preflight(&model, &base)?;
    Ok((model, base, layout))
}

/// Reject incomplete or mismatched packages before allocating resident weights.
fn preflight(model: &Model, base: &Path) -> Result<()> {
    let mut kernels = BTreeMap::<String, super::kernels::Spec>::new();
    for (rank_index, rank) in model.ranks.iter().enumerate() {
        if rank.rank as usize != rank_index {
            return Err("Rank order differs from TP topology".into());
        }
        let mut buffers = BTreeMap::new();
        for definition in &rank.buffers {
            if definition.bytes == 0 || buffers.contains_key(&definition.name) {
                return Err("Invalid/duplicate model buffer".into());
            }
            if let Some(view) = &definition.view {
                let bytes = *buffers
                    .get(&view.parent)
                    .ok_or("Model view precedes its allocation")?;
                if view
                    .offset
                    .checked_add(definition.bytes)
                    .is_none_or(|end| end > bytes)
                    || definition.weight.is_some()
                    || definition.data.is_some()
                    || definition.prepared_weight.is_some()
                {
                    return Err("Invalid model tensor view".into());
                }
            }
            buffers.insert(definition.name.clone(), definition.bytes);
            if let Some(file) = &definition.prepared_weight {
                let path = resolve_file(base, file)?;
                let metadata = std::fs::metadata(path).map_err(|e| e.to_string())?;
                if !metadata.is_file()
                    || metadata.len() != definition.bytes as u64
                    || definition.weight.is_some()
                    || definition.data.is_some()
                {
                    return Err("Prepared weight metadata differs".into());
                }
            }
        }
        if rank.packets.len() != 2 {
            return Err("Expected single-row decode and 16-row prefill packets".into());
        }
        for packet in &rank.packets {
            if ![1, 16].contains(&packet.rows) {
                return Err("Unsupported packet row count".into());
            }
            let layers = packet
                .body
                .iter()
                .filter_map(|step| {
                    if let Instruction::Snapshot { name, .. } = step {
                        Some(name.clone())
                    } else {
                        None
                    }
                })
                .collect::<std::collections::BTreeSet<_>>();
            if layers != (0..40).map(|i| format!("layer{i:02}")).collect() {
                return Err("A2 execution packet does not cover all 40 layers".into());
            }
            for instruction in packet.body.iter().chain(&packet.head) {
                if let Instruction::Kernel {
                    manifest,
                    buffers: bindings,
                } = instruction
                {
                    if !kernels.contains_key(manifest) {
                        let path = resolve_file(base, manifest)?;
                        let spec: super::kernels::Spec = crate::model::read(&path)?;
                        spec.validate()?;
                        spec.target.require_host(std::env::consts::ARCH)?;
                        read_identity(
                            path.parent().ok_or("Kernel path has no parent")?,
                            &spec.library,
                        )?;
                        kernels.insert(manifest.clone(), spec);
                    }
                    let spec = &kernels[manifest];
                    if bindings.len() != spec.operands.len() {
                        return Err("Packet/AOT operand count differs".into());
                    }
                    for (binding, operand) in bindings.iter().zip(&spec.operands) {
                        if buffers.get(binding).copied() != Some(operand.bytes()?) {
                            return Err(format!("{binding}: packet/AOT byte extent differs"));
                        }
                    }
                }
                if let Instruction::W4 {
                    rows,
                    input,
                    output,
                    experts,
                    buffers: bindings,
                    workspace,
                    ..
                } = instruction
                {
                    if *experts != 48
                        || *rows != packet.rows * 6
                        || !input.is_multiple_of(256)
                        || !output.is_multiple_of(32)
                    {
                        return Err("W4 model geometry differs from the routed packet".into());
                    }
                    let extents = [
                        rows * input,
                        experts * input * output / 2,
                        experts * output * 8,
                        experts * output * 4,
                        rows * 4,
                        rows * output * 2,
                    ];
                    if bindings
                        .iter()
                        .zip(extents)
                        .any(|(name, bytes)| buffers.get(name).copied() != Some(bytes))
                        || buffers
                            .get(workspace)
                            .is_none_or(|&bytes| bytes < 100663808)
                    {
                        return Err("W4 packet operand/workspace extent differs".into());
                    }
                }
            }
        }
    }
    Ok(())
}
