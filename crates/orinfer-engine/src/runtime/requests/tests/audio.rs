//! Packed audio must preserve each owner's features across order and reuse.
use super::*;

#[test]
#[ignore = "Real admission/reuse/reclamation state; requires GPU lock"]
fn admission_selection_survives_reclamation_and_cancellation() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(fixture.cases.len() >= 2);
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options::default();
    let mut survivor = model
        .start_request(input(&fixture.cases[0], 8), &|| false)
        .unwrap();
    prefill_reference(&mut model, &mut survivor);
    let expected = state(&mut model, &mut survivor);
    let mut other = model
        .start_request(input(&fixture.cases[1], 8), &|| false)
        .unwrap();
    model.finish_request(&mut other, false).unwrap();
    let source = input(&fixture.cases[1], 16);
    assert!(model.can_admit_request(&source, &options).unwrap());
    assert!(model.can_admit_request(&source, &options).unwrap());
    model.execution.reclaim_idle_state().unwrap();
    let mut reused = model.start_request(source, &|| false).unwrap();
    assert_ne!(reused.slot, survivor.slot);
    assert_eq!(expected, state(&mut model, &mut survivor));
    model.finish_request(&mut reused, false).unwrap();
    let mut requests = vec![];
    for i in 0..8 {
        let source = input(&fixture.cases[i % 2], 16);
        assert!(model.can_admit_request(&source, &options).unwrap());
        requests.push(model.start_request(source, &|| false).unwrap());
    }
    let mut slots = BTreeSet::from([survivor.slot]);
    assert!(requests.iter().all(|r| slots.insert(r.slot)));
    for request in &mut requests {
        model.finish_request(request, false).unwrap();
    }
    assert_eq!(expected, state(&mut model, &mut survivor));
    model.finish_request(&mut survivor, false).unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert_eq!(model.execution.resident_kv_bytes(), 0);
}

#[test]
#[ignore = "Real joint ASR prefill distributions and owner isolation; requires GPU lock"]
fn joint_prefill_preserves_distributions_and_owner_order() {
    let raw: serde_json::Value = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let fixture: Fixture = serde_json::from_value(raw.clone()).unwrap();
    let reference_path = raw["distribution_reference"].as_str().map(PathBuf::from);
    let write_reference = raw["write_distribution_reference"].as_bool() == Some(true);
    let reference = reference_path
        .as_ref()
        .filter(|_| !write_reference)
        .map(|path| {
            std::fs::read(path)
                .unwrap()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect::<Vec<_>>()
        });
    assert!(fixture.cases.len() >= 2);
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let steps = 8;
    let options = scheduler::Options::default();
    let mut expected = vec![];
    let mut targets = vec![];
    for case in &fixture.cases {
        let mut request = model
            .start_request(input(case, steps + 2), &|| false)
            .unwrap();
        prefill_reference(&mut model, &mut request);
        let mut probabilities = vec![];
        let mut tokens = vec![];
        for step in 0..steps {
            let distribution = flash_batch::distribution(&mut model, &mut request);
            let token = distribution
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            tokens.push(token);
            probabilities.push(distribution);
            if step + 1 < steps {
                *request.history.last_mut().unwrap() = token;
                model
                    .with_request(&mut request, |m, _| {
                        m.upload_ids(&m.manifest.token, &[token])
                    })
                    .unwrap();
                model
                    .advance_requests(&mut [&mut request], &options)
                    .unwrap();
            }
        }
        expected.push(probabilities);
        targets.push(tokens);
        model.finish_request(&mut request, false).unwrap();
    }
    let mut rows = vec![];
    if let Some(path) = &reference_path {
        let token_path = path.with_extension("tokens.json");
        if write_reference {
            std::fs::write(token_path, serde_json::to_vec(&targets).unwrap()).unwrap();
        } else {
            let tokens: Vec<Vec<u32>> = crate::model::read(&token_path).unwrap();
            assert_eq!(
                tokens, targets,
                "Comparison packages require identical forced histories"
            );
            assert_eq!(
                reference.as_ref().unwrap().len(),
                fixture.cases.len() * steps * model.manifest.vocab
            );
        }
    }
    let mut previous: Option<Vec<Vec<Vec<f64>>>> = None;
    for reverse in [false, true, false] {
        let mut requests: Vec<_> = fixture
            .cases
            .iter()
            .map(|case| {
                model
                    .start_request(input(case, steps + 2), &|| false)
                    .unwrap()
            })
            .collect();
        let mut active: Vec<_> = requests.iter_mut().collect();
        if reverse {
            active.reverse();
        }
        model.prepare_audio_cohorts(&mut active).unwrap();
        let selected: Vec<_> = active
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.input.len()))
            .collect();
        assert!(selected.iter().map(|(_, n)| n).sum::<usize>() <= 4096);
        let mut outputs = vec![];
        model
            .execute_batch_selected(&mut active, &selected, &mut outputs, &mut || {})
            .unwrap();
        drop(active);
        let mut actuals: Vec<Vec<Vec<f64>>> = vec![vec![]; requests.len()];
        for step in 0..steps {
            for (i, request) in requests.iter_mut().enumerate() {
                assert!(!request.prefilling);
                let actual = flash_batch::distribution(&mut model, request);
                assert!(actual.iter().all(|v| v.is_finite()));
                let baseline = if let Some(values) = &reference {
                    let first = (i * steps + step) * model.manifest.vocab;
                    &values[first..first + model.manifest.vocab]
                } else {
                    &expected[i][step]
                };
                let kl: f64 = baseline
                    .iter()
                    .zip(&actual)
                    .map(|(p, q)| p.exp() * (p - q))
                    .sum();
                // Compare complete distributions under identical histories;
                // the selected token alone is only a diagnostic.
                if let Some(previous) = &previous {
                    assert_eq!(
                        actual, previous[i][step],
                        "Owner order changed a fixed-history distribution"
                    );
                }
                let token = targets[i][step];
                rows.push(json!({"case":i,"reverse":reverse,"step":step,
                    "kl_from_reference":kl,"target_nll_delta":baseline[token as usize]-actual[token as usize]}));
                actuals[i].push(actual);
                *request.history.last_mut().unwrap() = token;
                model
                    .with_request(request, |m, _| m.upload_ids(&m.manifest.token, &[token]))
                    .unwrap();
            }
            if step + 1 < steps {
                let mut active: Vec<_> = requests.iter_mut().collect();
                if reverse {
                    active.reverse();
                }
                model.advance_requests(&mut active, &options).unwrap();
            }
        }
        if write_reference && previous.is_none() {
            use std::io::Write;
            let mut writer = std::io::BufWriter::new(
                std::fs::File::create(reference_path.as_ref().unwrap()).unwrap(),
            );
            for value in actuals.iter().flatten().flatten() {
                writer.write_all(&value.to_le_bytes()).unwrap();
            }
            writer.flush().unwrap();
        }
        previous = Some(actuals);
        for request in &mut requests {
            model.finish_request(request, false).unwrap();
        }
    }
    if reference.is_some() {
        let delta = rows
            .iter()
            .map(|r| r["target_nll_delta"].as_f64().unwrap())
            .sum::<f64>()
            / rows.len() as f64;
        assert!(
            delta < 0.02,
            "Mean fixed-history target NLL regression: {delta}"
        );
    }
    std::fs::write(fixture.output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires ASR recordings, window lengths and exclusive GPU lock"]
fn packed_audio_matches_individual_recordings() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(fixture.cases.len() >= 2);
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: crate::execution::GraphMode::Full,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let a = model.manifest.audio.clone().unwrap();
    assert_eq!(a.window_size, 104);
    let mut expected = vec![];
    for case in &fixture.cases {
        let mut request = model.start_request(input(case, 2), &|| false).unwrap();
        model.prepare_audio_cohorts(&mut [&mut request]).unwrap();
        let bytes = case.audios[0].features().unwrap() * a.hidden * 2;
        expected.push(
            model
                .with_request(&mut request, |m, _| {
                    m.execution.download_bytes(&a.features, bytes)
                })
                .unwrap(),
        );
        model.finish_request(&mut request, false).unwrap();
    }
    let distinct_sizes = expected.iter().enumerate().all(|(i, bytes)| {
        expected[i + 1..]
            .iter()
            .all(|other| bytes.len().abs_diff(other.len()) >= 65536)
    });
    let mut slots = vec![0; fixture.cases.len()];
    for (round, (reverse_admission, reverse)) in
        [(false, false), (true, false), (false, true), (true, true)]
            .into_iter()
            .enumerate()
    {
        let mut cases: Vec<_> = (0..fixture.cases.len()).collect();
        if reverse_admission {
            cases.reverse();
        }
        let mut requests: Vec<_> = cases
            .iter()
            .map(|&i| {
                model
                    .start_request(input(&fixture.cases[i], 2), &|| false)
                    .unwrap()
            })
            .collect();
        for (&case, request) in cases.iter().zip(&requests) {
            if round == 0 {
                slots[case] = request.slot;
            } else if distinct_sizes {
                assert_eq!(
                    request.slot, slots[case],
                    "Audio cohort resized a fitting idle arena"
                );
            }
        }
        let mut order: Vec<_> = requests.iter_mut().collect();
        if reverse {
            order.reverse();
        }
        if round == 3 {
            model.prepare_selected_audio(&mut order, &[(0, 1)]).unwrap();
            assert!(order[0].pending_audio.is_empty());
            assert!(order[1..].iter().all(|r| !r.pending_audio.is_empty()));
            let remaining: Vec<_> = (1..order.len()).map(|i| (i, 1)).collect();
            model
                .prepare_selected_audio(&mut order, &remaining)
                .unwrap();
            assert!(order.iter().all(|r| r.pending_audio.is_empty()));
        } else {
            model.prepare_audio_cohorts(&mut order).unwrap();
        }
        drop(order);
        for (i, request) in requests.iter_mut().enumerate() {
            let case = cases[i];
            let actual = model
                .with_request(request, |m, _| {
                    m.execution
                        .download_bytes(&a.features, expected[case].len())
                })
                .unwrap();
            let actual = floats(&actual, crate::artifact::Dtype::F16);
            let reference = floats(&expected[case], crate::artifact::Dtype::F16);
            assert!(actual.iter().all(|v| v.is_finite()));
            let error: f64 = actual
                .iter()
                .zip(&reference)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum();
            let energy: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
            let relative_rmse = (error / energy.max(1e-30)).sqrt();
            println!("packed audio case={i} reverse={reverse} relative_rmse={relative_rmse}");
            assert!(relative_rmse < 0.002);
            model.finish_request(request, false).unwrap();
        }
    }
}
