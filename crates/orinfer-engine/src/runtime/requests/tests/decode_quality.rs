//! Stream fixed-history distributions so packages can be checked one at a time.
use super::*;
use std::io::{BufReader, BufWriter, Read, Write};

#[test]
#[ignore = "Requires real multimodal requests and the exclusive GPU experiment lock"]
fn validate_decode_owner_order() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(fixture.cases.len() >= 2);
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: crate::execution::GraphMode::DecodeOnly,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let options = scheduler::Options::default();
    let mut expected = None;
    for reverse in [false, true] {
        let mut requests: Vec<_> = fixture
            .cases
            .iter()
            .map(|case| model.start_request(input(case, 24), &|| false).unwrap())
            .collect();
        for request in &mut requests {
            prefill_reference(&mut model, request);
        }
        let before = state(&mut model, &mut requests[0]);
        let mut cancelled = model
            .start_request(input(&fixture.cases[1], 8), &|| false)
            .unwrap();
        model.finish_request(&mut cancelled, false).unwrap();
        assert_eq!(before, state(&mut model, &mut requests[0]));
        let mut reused = model
            .start_request(input(&fixture.cases[1], 8), &|| false)
            .unwrap();
        assert_eq!(before, state(&mut model, &mut requests[0]));
        model.finish_request(&mut reused, false).unwrap();
        let mut rounds = vec![];
        for step in 0..8 {
            let mut active: Vec<_> = requests.iter_mut().collect();
            if reverse {
                active.reverse();
            }
            let steps = model.advance_requests(&mut active, &options).unwrap();
            assert_eq!(steps.len(), fixture.cases.len());
            drop(active);
            let mut snapshots = vec![];
            for request in &mut requests {
                let token = *request.history.last().unwrap();
                if step != 7 {
                    snapshots.push((token, BTreeMap::new()));
                    continue;
                }
                let end = request.input.len() + request.generated - 1;
                let hashes: BTreeMap<String, String> = model
                    .with_request(request, |m, _| {
                        m.prefix_ranges(end)?
                            .into_iter()
                            .map(|(name, range)| {
                                let bytes = m
                                    .execution
                                    .download_bytes(&name, range.offset + range.bytes)?;
                                Ok((name, crate::artifact::sha256(&bytes[range.offset..])))
                            })
                            .collect()
                    })
                    .unwrap();
                snapshots.push((token, hashes));
            }
            rounds.push(snapshots);
        }
        if let Some(reference) = &expected {
            for (step, (actual, baseline)) in rounds.iter().zip(reference).enumerate() {
                assert_eq!(
                    actual, baseline,
                    "Owner order changed committed state at step {step}"
                );
            }
        } else {
            expected = Some(rounds);
        }
        for request in &mut requests {
            model.finish_request(request, false).unwrap();
        }
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"owner_order_state_equal":true,
            "cancel_reuse_state_equal":true,"steps":8,"batch":fixture.cases.len(),
            "scope":"Same live batch and per-owner histories, reversed request order. Committed KV, GDN, convolution and position ranges must match byte for byte; MTP off."}))
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires official token scoring cases, a real package and the exclusive GPU lock"]
fn score_batched_decode_histories() {
    let fixture: serde_json::Value = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let spec: crate::runtime::scoring::ScoreRequests =
        crate::model::read(&PathBuf::from(fixture["requests"].as_str().unwrap())).unwrap();
    let count = fixture["batch"].as_u64().unwrap() as usize;
    assert!((1..=128).contains(&count) && spec.cases.len() <= count);
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::GraphMode::DecodeOnly,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    spec.validate(model.manifest.vocab, model.manifest.max_context)
        .unwrap();
    model.manifest.mtp = None;
    let steps = spec.cases.iter().map(|c| c.target_ids.len()).max().unwrap();
    let mut requests: Vec<_> = spec
        .cases
        .iter()
        .cycle()
        .take(count)
        .map(|case| {
            model
                .start_request(
                    GenerationInput {
                        audios: case.audios.clone(),
                        input_tokens: case.prompt_ids.clone(),
                        images: case.images.clone(),
                        max_new_tokens: steps + 2,
                        sampling: crate::sampling::Options {
                            temperature: 0.,
                            seed: spec.seed,
                            repetition_penalty: 1.,
                            ..Default::default()
                        },
                        prefix_hints: vec![],
                    },
                    &|| false,
                )
                .unwrap()
        })
        .collect();
    if fixture["joint_prefill"].as_bool() == Some(true) {
        let selected: Vec<_> = requests
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.input.len()))
            .collect();
        assert!(
            selected.iter().map(|(_, n)| n).sum::<usize>()
                <= model.manifest.batch_layout.as_ref().unwrap().segment_tokens
        );
        let mut active: Vec<_> = requests.iter_mut().collect();
        model.prepare_audio_cohorts(&mut active).unwrap();
        model
            .execute_batch_selected(&mut active, &selected, &mut vec![], &mut || {})
            .unwrap();
    } else {
        for request in &mut requests {
            prefill_reference(&mut model, request);
        }
    }
    let mut probes = vec![];
    for position in 0..steps {
        for (lane, case) in spec.cases.iter().enumerate() {
            if position >= case.target_ids.len() {
                continue;
            }
            let distribution = flash_batch::distribution(&mut model, &mut requests[lane]);
            assert!(distribution.iter().all(|x| x.is_finite()));
            let mut ids: Vec<_> = (0..distribution.len()).collect();
            ids.select_nth_unstable_by(2, |&a, &b| {
                distribution[b].total_cmp(&distribution[a]).then(a.cmp(&b))
            });
            ids[..3].sort_by(|&a, &b| distribution[b].total_cmp(&distribution[a]).then(a.cmp(&b)));
            let target = case.target_ids[position];
            let mut queries = case.query_ids.get(position).cloned().unwrap_or_default();
            queries.push(target);
            let queried: BTreeMap<_, _> = queries
                .into_iter()
                .map(|id| (id, distribution[id as usize]))
                .collect();
            let history: Vec<_> = case
                .prompt_ids
                .iter()
                .chain(&case.target_ids[..position])
                .copied()
                .collect();
            probes.push(json!({
                "case_id":case.id,"execution_mode":"decode","position":position,"seed":spec.seed,
                "context_sha256":crate::runtime::scoring::scoring_context_hash(&history, &case.images, &case.audios).unwrap(),
                "reference_token_id":target,"reference_logprob":distribution[target as usize],
                "top3":ids[..3].iter().map(|&id|json!({"token_id":id,"logprob":distribution[id]})).collect::<Vec<_>>(),
                "queried_logprobs":queried
            }));
        }
        if position + 1 < steps {
            for (lane, request) in requests.iter_mut().enumerate() {
                let case = &spec.cases[lane % spec.cases.len()];
                let token = case.target_ids[position.min(case.target_ids.len() - 1)];
                *request.history.last_mut().unwrap() = token;
                model
                    .with_request(request, |m, _| m.upload_ids(&m.manifest.token, &[token]))
                    .unwrap();
            }
            let outputs = model
                .advance_requests(
                    &mut requests.iter_mut().collect::<Vec<_>>(),
                    &scheduler::Options::default(),
                )
                .unwrap();
            assert_eq!(outputs.iter().map(|s| s.tokens.len()).sum::<usize>(), count);
        }
    }
    for request in &mut requests {
        model.finish_request(request, false).unwrap();
    }
    std::fs::write(
        fixture["output"].as_str().unwrap(),
        serde_json::to_vec_pretty(&json!({
            "model":fixture["model"],"manifest_sha256":model.stats.manifest_sha256,
            "seed":spec.seed,"batch":count,"probes":probes,
            "scope":"Teacher-forced histories through native batched decode, MTP off. Repeated lanes maintain batch occupancy; this is a quality probe, not a throughput benchmark."
        }))
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires tokenized requests, a real package and the exclusive GPU experiment lock"]
fn compare_decode_package_quality() {
    let fixture: serde_json::Value = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = PathBuf::from(fixture["output"].as_str().unwrap());
    let reference = fixture["reference"].as_str().map(PathBuf::from);
    let cases: Vec<GenerationInput> = serde_json::from_value(fixture["cases"].clone()).unwrap();
    let count = fixture["batch"].as_u64().unwrap() as usize;
    let steps = fixture["steps"].as_u64().unwrap() as usize;
    let lanes: Vec<usize> = serde_json::from_value(fixture["sample_lanes"].clone()).unwrap();
    assert!((1..=128).contains(&count) && (8..=128).contains(&steps));
    assert!(!lanes.is_empty() && lanes.iter().all(|&i| i < count));
    assert!(cases.len() >= count, "Distinct prompts are required");
    let original: Option<serde_json::Value> =
        reference.as_ref().map(|p| crate::model::read(p).unwrap());
    let reference_tokens: Option<Vec<Vec<u32>>> = original
        .as_ref()
        .map(|r| serde_json::from_value(r["tokens"].clone()).unwrap());
    if let Some(r) = &original {
        assert_eq!(r["batch"], count);
        assert_eq!(r["steps"], steps);
        assert_eq!(r["sample_lanes"], json!(lanes));
        assert_eq!(r["cases"], fixture["cases"]);
    }
    let score_path = output.with_extension("f32");
    let mut writer = reference
        .is_none()
        .then(|| BufWriter::new(std::fs::File::create(&score_path).unwrap()));
    let mut reader = reference
        .as_ref()
        .map(|p| BufReader::new(std::fs::File::open(p.with_extension("f32")).unwrap()));
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            cuda_graph: crate::execution::GraphMode::DecodeOnly,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..LoadOptions::default()
        },
    )
    .unwrap();
    model.manifest.mtp = None;
    let options = scheduler::Options::default();
    let mut requests: Vec<_> = cases[..count]
        .iter()
        .map(|source| {
            model
                .start_request(input(source, steps + 2), &|| false)
                .unwrap()
        })
        .collect();
    for request in &mut requests {
        prefill_reference(&mut model, request);
    }
    let mut tokens = Vec::with_capacity(steps);
    let mut max_kl = 0.0_f64;
    let mut nll_sum = 0.0;
    let mut top3 = 0;
    let mut observations = 0;
    let mut deviations = vec![];
    let mut raw = vec![0u8; model.manifest.vocab * 4];
    for step in 0..steps {
        let selected: Vec<u32> = requests
            .iter()
            .map(|r| *r.history.last().unwrap())
            .collect();
        let forced = reference_tokens.as_ref().map_or(&selected, |t| &t[step]);
        for &lane in &lanes {
            let actual = flash_batch::distribution(&mut model, &mut requests[lane]);
            if let Some(writer) = &mut writer {
                let (destinations, remainder) = raw.as_chunks_mut::<4>();
                assert!(remainder.is_empty());
                for (destination, &value) in destinations.iter_mut().zip(&actual) {
                    destination.copy_from_slice(&(value as f32).to_le_bytes());
                }
                writer.write_all(&raw).unwrap();
            } else {
                reader.as_mut().unwrap().read_exact(&mut raw).unwrap();
                let (bytes, remainder) = raw.as_chunks::<4>();
                assert!(remainder.is_empty());
                let expected: Vec<_> = bytes
                    .iter()
                    .map(|b| f32::from_le_bytes(*b) as f64)
                    .collect();
                let kl: f64 = expected
                    .iter()
                    .zip(&actual)
                    .map(|(b, a)| b.exp() * (b - a))
                    .sum();
                assert!(kl.is_finite(), "step {step}, lane {lane}: KL {kl}");
                max_kl = max_kl.max(kl);
                let token = forced[lane] as usize;
                let delta = expected[token] - actual[token];
                nll_sum += delta;
                if kl > 0.001 {
                    deviations.push(
                        json!({"step":step,"lane":lane,"kl":kl,"selected_nll_delta":delta,
                        "reference_selected_probability":expected[token].exp(),
                        "candidate_selected_probability":actual[token].exp()}),
                    );
                }
                top3 += usize::from(actual.iter().filter(|&&p| p > actual[token]).count() < 3);
                observations += 1;
            }
        }
        tokens.push(forced.to_vec());
        if step + 1 < steps {
            for (request, &token) in requests.iter_mut().zip(forced) {
                *request.history.last_mut().unwrap() = token;
                model
                    .with_request(request, |m, _| m.upload_ids(&m.manifest.token, &[token]))
                    .unwrap();
            }
            let result = model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
            assert_eq!(result.iter().map(|s| s.tokens.len()).sum::<usize>(), count);
        }
    }
    for request in &mut requests {
        model.finish_request(request, false).unwrap();
    }
    if let Some(writer) = &mut writer {
        writer.flush().unwrap();
        writer.get_ref().sync_all().unwrap();
    }
    if let Some(reader) = &mut reader {
        assert_eq!(reader.read(&mut [0u8; 1]).unwrap(), 0);
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&json!({
        "model":fixture["model"],"fingerprint":model.stats.manifest_sha256,
        "reference":reference,"batch":count,"steps":steps,"sample_lanes":lanes,
        "cases":fixture["cases"],"tokens":tokens,"seed":20261002,
        "max_kl":max_kl,"observations":observations,"deviations":deviations,
        "mean_selected_nll_delta":nll_sum/(observations.max(1) as f64),
        "selected_in_candidate_top3":top3 as f64/observations.max(1) as f64,
        "scope":"Fixed-history real native decode; models loaded separately, MTP off. Distributions sampled across distinct requests. Not a performance benchmark."
    })).unwrap()).unwrap();
    if reference.is_some() {
        eprintln!(
            "package quality: max KL {max_kl}, mean NLL delta {}, candidate top3 {}",
            nll_sum / observations as f64,
            top3 as f64 / observations as f64
        );
        assert!(
            max_kl < 0.01
                && nll_sum / (observations as f64) < 0.01
                && top3 as f64 / observations as f64 >= 0.99
        );
    }
}
