use super::*;
use half::slice::HalfFloatSliceExt;
impl ModelRuntime {
    pub(super) fn prepare_audio_stem(
        &self,
        input: &crate::audio::AudioInput,
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        self.prepare_audio_stems(&[(0, input)], cancelled)?;
        self.execution.sync()
    }
    fn prepare_audio_stems(
        &self,
        inputs: &[(usize, &crate::audio::AudioInput)],
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        let a = self
            .manifest
            .audio
            .as_ref()
            .ok_or("Model has no audio adapter")?;
        let chunks: Vec<_> = inputs
            .iter()
            .flat_map(|&(base, input)| {
                (0..input.frames.div_ceil(100)).map(move |chunk| (base, input, chunk))
            })
            .collect();
        for group in chunks.chunks(a.stem_chunks) {
            if cancelled() {
                return Err("Request cancelled during audio encoding".into());
            }
            let mut raw = vec![half::f16::ZERO; group.len() * 128 * 100];
            for (local, &(_, input, chunk)) in group.iter().enumerate() {
                let padded = input.frames.div_ceil(100) * 100;
                for mel in 0..128 {
                    let source = mel * padded + chunk * 100;
                    let destination = (local * 128 + mel) * 100;
                    raw[destination..destination + 100]
                        .convert_from_f32_slice(&input.mel[source..source + 100]);
                }
            }
            // SAFETY: f16 is transparent 16-bit storage; the byte view stays
            // within its live allocation. Jetson stores it little-endian.
            let bytes =
                unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<u8>(), raw.len() * 2) };
            self.upload_bytes(&a.mel, bytes)?;
            self.execution.submit_program_rows(
                "audio_stem",
                group.len(),
                ExecutionPhase::Vision,
            )?;
            for (local, &(base, input, chunk)) in group.iter().enumerate() {
                let features = (input.frames / 100) * 13 + (input.frames % 100).div_ceil(8);
                let live = (features - chunk * 13).min(13);
                self.execution.copy_range_async(
                    &a.stem_output,
                    local * 13 * 1024 * 2,
                    &a.encoder_input,
                    (base + chunk * 13) * 1024 * 2,
                    live * 1024 * 2,
                )?;
            }
        }
        Ok(())
    }
    pub(super) fn encode_audio_cohort(
        &mut self,
        inputs: &[(usize, &crate::audio::AudioInput)],
    ) -> Result<()> {
        let a = self
            .manifest
            .audio
            .clone()
            .ok_or("Model has no audio adapter")?;
        let sources: Vec<_> = inputs.iter().map(|(_, input)| *input).collect();
        let layout = audio_layout(&sources, a.window_size, a.max_features)?;
        self.prepare_audio_stems(
            &layout
                .spans
                .iter()
                .zip(&sources)
                .map(|(&(base, _), input)| (base, *input))
                .collect::<Vec<_>>(),
            &|| false,
        )?;
        self.upload_ids(&a.length, &layout.lengths)?;
        self.launch_rows("audio_encoder", layout.rows, ExecutionPhase::Vision)?;
        for (&(slot, _), &(base, features)) in inputs.iter().zip(&layout.spans) {
            self.execution.copy_to_sequence_async(
                &a.output,
                base * a.hidden * 2,
                slot,
                &a.features,
                features * a.hidden * 2,
            )?;
        }
        self.execution.sync()
    }
    pub(super) fn prepare_audio_capacity(
        &self,
        tokens: &[u32],
        inputs: &[crate::audio::AudioInput],
        capacity: usize,
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        self.prepare_audio_inputs(tokens, inputs, capacity, true, cancelled)
    }
    pub(super) fn prepare_audio_metadata(
        &self,
        tokens: &[u32],
        inputs: &[crate::audio::AudioInput],
        capacity: usize,
    ) -> Result<()> {
        self.prepare_audio_inputs(tokens, inputs, capacity, false, &|| false)
    }
    fn prepare_audio_inputs(
        &self,
        tokens: &[u32],
        inputs: &[crate::audio::AudioInput],
        capacity: usize,
        encode: bool,
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        let Some(a) = &self.manifest.audio else {
            return if inputs.is_empty() {
                Ok(())
            } else {
                Err("Model has no audio adapter".into())
            };
        };
        if inputs.len() > 1 {
            return Err("ASR accepts one recording per request".into());
        }
        let mut index = vec![-1_i32; capacity];
        let mut cursor = 0;
        let mut feature_offset = 0;
        for input in inputs {
            let features = input.features()?;
            let start = tokens
                .iter()
                .skip(cursor)
                .position(|&t| t == a.token_id)
                .map(|p| p + cursor)
                .ok_or("Missing ASR audio placeholder")?;
            let end = start
                .checked_add(features)
                .filter(|&n| n <= tokens.len())
                .ok_or("Audio span exceeds prompt")?;
            if tokens[start..end].iter().any(|&t| t != a.token_id)
                || feature_offset + features > a.max_features
            {
                return Err("Invalid audio placeholder span".into());
            }
            for (i, slot) in index[start..end].iter_mut().enumerate() {
                *slot = (feature_offset + i) as i32;
            }
            if encode {
                self.prepare_audio_stem(input, cancelled)?;
                let layout = audio_layout(&[input], a.window_size, a.max_features)?;
                self.upload_ids(&a.length, &layout.lengths)?;
                self.launch_rows("audio_encoder", features, ExecutionPhase::Vision)?;
                self.execution.copy_range(
                    &a.output,
                    &a.features,
                    feature_offset * a.hidden * 2,
                    features * a.hidden * 2,
                )?;
            }
            feature_offset += features;
            cursor = end;
        }
        if tokens.iter().filter(|&&t| t == a.token_id).count() != feature_offset {
            return Err("Unbound audio placeholder".into());
        }
        self.upload_bytes(
            &a.feature_index,
            &index
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let positions: Vec<u32> = (0..capacity as u32).flat_map(|i| [i, i, i]).collect();
        self.upload_ids(&a.positions, &positions)?;
        Ok(())
    }
}

struct AudioLayout {
    rows: usize,
    spans: Vec<(usize, usize)>,
    lengths: Vec<u32>,
}
fn audio_layout(
    inputs: &[&crate::audio::AudioInput],
    window: usize,
    capacity: usize,
) -> Result<AudioLayout> {
    if inputs.is_empty() || (window == 0 && inputs.len() > 1) {
        return Err("Invalid audio cohort".into());
    }
    let mut layout = AudioLayout {
        rows: 0,
        spans: vec![],
        lengths: vec![],
    };
    for input in inputs {
        let features = input.features()?;
        let base = if window == 0 {
            0
        } else {
            layout.rows.div_ceil(window) * window
        };
        layout.rows = base
            .checked_add(features)
            .filter(|&n| n <= capacity)
            .ok_or("Audio cohort exceeds workspace".to_string())?;
        layout.spans.push((base, features));
        if window == 0 {
            layout.lengths.push(features as u32);
        } else {
            layout.lengths.extend(
                (0..features)
                    .step_by(window)
                    .map(|first| (features - first).min(window) as u32),
            );
        }
    }
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_windows_keep_recording_tails_separate() {
        let first = crate::audio::AudioInput {
            frames: 801,
            mel: vec![0.; 128 * 900],
        };
        let second = crate::audio::AudioInput {
            frames: 101,
            mel: vec![0.; 128 * 200],
        };
        let layout = audio_layout(&[&first, &second], 104, 312).unwrap();
        assert_eq!(layout.spans, [(0, 105), (208, 14)]);
        assert_eq!(layout.lengths, [104, 1, 14]);
        assert_eq!(layout.rows, 222);
        assert!(audio_layout(&[&first, &second], 104, 221).is_err());
        assert!(audio_layout(&[&first, &second], 0, 312).is_err());
    }
    #[derive(serde::Deserialize)]
    struct Case {
        input: crate::audio::AudioInput,
        reference: std::path::PathBuf,
        output: Option<std::path::PathBuf>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        model: std::path::PathBuf,
        cases: Vec<Case>,
    }
    #[test]
    #[ignore = "Requires ASR weights, FP16 references and exclusive GPU lock"]
    fn encoder_matches_fp16_and_replays_changed_audio() {
        let f: Fixture = crate::model::read(&std::path::PathBuf::from(
            std::env::var("ORINFER_AUDIO_FIXTURE").unwrap(),
        ))
        .unwrap();
        assert!(f.cases.len() >= 2);
        let model = ModelRuntime::load_with_options(
            &f.model,
            LoadOptions {
                cuda_graph: crate::execution::GraphMode::Full,
                ..Default::default()
            },
        )
        .unwrap();
        let a = model.manifest.audio.as_ref().unwrap();
        let mut originals = Vec::new();
        for replay in 0..2 {
            for (i, case) in f.cases.iter().enumerate() {
                let rows = case.input.features().unwrap();
                let tokens = vec![a.token_id; rows];
                model
                    .prepare_audio_capacity(
                        &tokens,
                        std::slice::from_ref(&case.input),
                        rows,
                        &|| false,
                    )
                    .unwrap();
                let raw = model
                    .execution
                    .download_bytes(&a.features, rows * a.hidden * 2)
                    .unwrap();
                let actual = floats(&raw, crate::artifact::Dtype::F16);
                if replay == 0
                    && let Some(path) = &case.output
                {
                    std::fs::write(
                        path,
                        actual
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                }
                let reference: Vec<_> = std::fs::read(&case.reference)
                    .unwrap()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                assert_eq!(reference.len(), actual.len());
                assert!(actual.iter().all(|v| v.is_finite()));
                let error: f64 = actual
                    .iter()
                    .zip(&reference)
                    .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                    .sum();
                let energy: f64 = reference.iter().map(|&v| (v as f64).powi(2)).sum();
                let relative_rmse = (error / energy.max(1e-30)).sqrt();
                println!(
                    "audio case={i} replay={replay} rows={rows} relative_rmse={relative_rmse}"
                );
                assert!(relative_rmse < 0.02, "audio FP16 encoder mismatch");
                if replay == 0 {
                    originals.push(raw);
                } else {
                    assert_eq!(raw, originals[i], "graph replay changed audio features");
                }
            }
        }
    }
}
