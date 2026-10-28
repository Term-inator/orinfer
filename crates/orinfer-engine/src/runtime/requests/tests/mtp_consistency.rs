//! Full generation plus ordinary replay of the exact committed MTP history.
use super::*;

fn target_state(model: &mut ModelRuntime, request: &mut RequestState) -> BTreeMap<String, Vec<u8>> {
    model
        .with_request(request, |m, r| {
            let deferred = m
                .manifest
                .gdn_execution
                .as_ref()
                .is_some_and(|g| g.deferred_capacity().is_some());
            if deferred {
                // A base plus pending rank-one updates is the same logical
                // state as its materialized matrix. Canonicalize both paths
                // before comparing, and prove every pending count is drained.
                let materialize: Vec<_> = m
                    .manifest
                    .kernels
                    .iter()
                    .filter(|k| k.name.ends_with("/private/recurrent_materialize"))
                    .map(|k| crate::execution::Invocation {
                        operation: crate::model::Operation::Kernel {
                            name: k.name.clone(),
                        },
                        sequence: Some(r.slot),
                        launch: None,
                        views: BTreeMap::new(),
                    })
                    .collect();
                assert!(
                    !materialize.is_empty(),
                    "Deferred state has no materialization kernels"
                );
                m.execution
                    .execute_batch(vec![], &materialize, false, false)?;
                for buffer in m
                    .manifest
                    .buffers
                    .iter()
                    .filter(|b| b.name.ends_with("_GdnLogCount"))
                {
                    assert!(
                        m.read_controls(&buffer.name, buffer.bytes()? / 4)?
                            .iter()
                            .all(|&n| n == 0)
                    );
                }
            }
            let mut draft_state = std::collections::BTreeSet::new();
            if let Some(spec) = &m.manifest.mtp {
                if let Some(kv) = &m.manifest.kv_cache {
                    for growth in kv.growth.values().filter(|g| g.position == spec.position) {
                        draft_state.extend(growth.buffers.iter().cloned());
                    }
                }
                if let Some(program) = &spec.draft_snapshot_program {
                    for operation in &m.manifest.programs[program] {
                        if let crate::model::Operation::Copy { source, .. } = operation {
                            draft_state.insert(source.clone());
                        }
                    }
                }
            }
            m.prefix_ranges(r.input.len() + r.generated - 1)?
                .into_iter()
                // Draft state has a different cursor during lazy catch-up.
                // Compare live target state and the hidden row needed by the next bridge.
                .filter(|(name, _)| {
                    ((name.starts_with('L') || name.starts_with("State_"))
                        && m.manifest.reset_buffers.contains(name)
                        && !draft_state.contains(name)
                        && (!deferred || !name.contains("_GdnLog")))
                        || name == &m.manifest.position
                        || m.manifest.mtp.as_ref().and_then(|s| s.hidden_ring.as_ref())
                            == Some(name)
                })
                .map(|(name, range)| {
                    let raw = m
                        .execution
                        .download_bytes(&name, range.offset + range.bytes)?;
                    Ok((name, raw[range.offset..].to_vec()))
                })
                .collect()
        })
        .unwrap()
}

#[test]
#[ignore = "Requires real MTP model and exclusive GPU experiment lock"]
fn validate_mtp_same_history_consistency() {
    #[derive(Deserialize)]
    struct Probe {
        #[serde(flatten)]
        fixture: Fixture,
        widths: Vec<usize>,
        output_tokens: usize,
        #[serde(default)]
        allow_differences: bool,
    }
    let probe: Probe = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let fixture = probe.fixture;
    assert!(!fixture.output.exists());
    assert!(probe.output_tokens >= 4);
    assert!(probe.widths.iter().all(|w| (1..=8).contains(w)));
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(probe.widths.iter().max().unwrap() - 1),
            resident_experts: fixture
                .resident_experts
                .map_or(Default::default(), crate::execution::ResidentExperts::Count),
            host_weight_cache_bytes: 16usize << 30,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(model.manifest.mtp.is_some());
    let options = scheduler::Options {
        max_active: fixture.batches.iter().copied().max().unwrap(),
        memory_reserve_bytes: 256 << 20,
        ..Default::default()
    };
    let mut evidence = vec![];
    for (case_index, source) in fixture.cases.iter().enumerate() {
        assert_eq!(source.sampling.temperature, 0.);
        let mut ordinary_output: Option<Vec<u32>> = None;
        for &count in &fixture.batches {
            for &width in &probe.widths {
                eprintln!("consistency case={case_index} batch={count} width={width}: generation");
                model.dynamic_speculation.fixed_width = Some(width);
                let mut requests: Vec<_> = (0..count)
                    .map(|_| {
                        model
                            .start_request(input(source, probe.output_tokens), &|| false)
                            .unwrap()
                    })
                    .collect();
                for request in &mut requests {
                    prefill_reference(&mut model, request);
                    assert_eq!(request.generated, 1);
                }
                let initial = target_state(&mut model, &mut requests[0]);
                let mut packets = vec![];
                while requests.iter().any(|r| !r.is_finished()) {
                    let steps = model
                        .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                        .unwrap();
                    assert!(!steps.is_empty(), "Generation made no progress");
                    packets.push(
                        steps
                            .iter()
                            .map(|s| json!({"lane":s.request,"tokens":s.tokens}))
                            .collect::<Vec<_>>(),
                    );
                    assert!(requests.iter().all(|r| r.failure().is_none()));
                }
                let outputs: Vec<Vec<u32>> = requests
                    .iter()
                    .map(|r| r.history[r.input.len()..].to_vec())
                    .collect();
                assert!(outputs.iter().all(|v| v.len() == probe.output_tokens));
                let mtp: Vec<_> = requests.iter().map(|r| r.mtp.clone()).collect();
                if width > 1 {
                    assert!(mtp.iter().all(|s| s.rounds > 0), "MTP did not execute");
                } else {
                    assert!(mtp.iter().all(|s| s.rounds == 0));
                }
                let actual_state = target_state(&mut model, &mut requests[0]);
                let mut different_lanes = vec![];
                for (lane, request) in requests.iter_mut().enumerate().skip(1) {
                    let lane_state = target_state(&mut model, request);
                    let changed: Vec<_> = actual_state
                        .iter()
                        .filter(|(name, raw)| lane_state.get(*name) != Some(*raw))
                        .map(|(name, _)| name.clone())
                        .collect();
                    if !changed.is_empty() {
                        different_lanes.push(json!({"lane":lane,"buffers":changed}));
                    }
                }
                for request in &mut requests {
                    model.finish_request(request, false).unwrap();
                }
                if count == 1 && width == 1 {
                    ordinary_output = Some(outputs[0].clone());
                }
                let baseline = ordinary_output.as_ref().expect("Run batch=1 width=1 first");
                let first_free_divergence =
                    outputs[0].iter().zip(baseline).position(|(a, b)| a != b);

                // This is the ordinary production packet, consuming precisely
                // the candidate's committed history, even after a free-run fork.
                eprintln!(
                    "consistency case={case_index} batch={count} width={width}: same-history replay"
                );
                model.dynamic_speculation.fixed_width = Some(1);
                let mut replay = model
                    .start_request(input(source, probe.output_tokens + 4), &|| false)
                    .unwrap();
                prefill_reference(&mut model, &mut replay);
                assert!(
                    target_state(&mut model, &mut replay) == initial,
                    "Cold prefill state changed before decode comparison"
                );
                assert_eq!(*replay.history.last().unwrap(), outputs[0][0]);
                let mut different_choices = vec![];
                for index in 1..probe.output_tokens {
                    let forced = outputs[0][index - 1];
                    *replay.history.last_mut().unwrap() = forced;
                    model
                        .with_request(&mut replay, |m, _| {
                            m.upload_ids(&m.manifest.token, &[forced])
                        })
                        .unwrap();
                    let mut steps = vec![];
                    model
                        .execute_batch_selected(
                            &mut [&mut replay],
                            &[(0, 1)],
                            &mut steps,
                            &mut || {},
                        )
                        .unwrap();
                    assert_eq!(steps.len(), 1);
                    assert_eq!(steps[0].tokens.len(), 1);
                    let chosen = steps[0].tokens[0];
                    if chosen != outputs[0][index] {
                        let scores = flash_batch::distribution(&mut model, &mut replay);
                        different_choices.push(json!({"output_index":index,
                            "candidate":outputs[0][index],"ordinary":chosen,
                            "ordinary_logp_candidate":scores[outputs[0][index] as usize],
                            "ordinary_logp_ordinary":scores[chosen as usize]}));
                    }
                }
                let reference_state = target_state(&mut model, &mut replay);
                let changed: Vec<_> = actual_state
                    .iter()
                    .filter(|(name, raw)| reference_state.get(*name) != Some(*raw))
                    .map(|(name, raw)| {
                        let reference = &reference_state[name];
                        json!({"buffer":name,"bytes":raw.len(),
                            "different_bytes":raw.iter().zip(reference).filter(|(a,b)| a!=b).count(),
                            "candidate_sha256":crate::artifact::sha256(raw),
                            "ordinary_sha256":crate::artifact::sha256(reference)})
                    })
                    .collect();
                model.finish_request(&mut replay, false).unwrap();
                let row = json!({"case":case_index,"prompt_tokens":source.input_tokens.len(),
                    "batch":count,"verification_width":width,"drafts":width-1,
                    "cuda_graph":fixture.cuda_graph,"outputs":outputs,"ordinary_output":baseline,
                    "all_lanes_equal":outputs.iter().all(|v| v==&outputs[0]),
                    "different_lane_states":different_lanes,
                    "first_free_divergence":first_free_divergence,
                    "same_history_different_choices":different_choices,
                    "same_history_changed_target_state":changed,"mtp":mtp,"packets":packets});
                eprintln!(
                    "consistency case={case_index} batch={count} width={width}: free_divergence={first_free_divergence:?}, same_history_choices={}, changed_target_buffers={}",
                    different_choices.len(),
                    changed.len()
                );
                evidence.push(row);
                std::fs::write(&fixture.output, serde_json::to_vec_pretty(&json!({
                    "status":"running","model":fixture.model,"seed":20261002,
                    "manifest_sha256":model.stats.manifest_sha256,"rows":evidence,
                    "scope":"Real continuous-worker greedy generation, followed by B=1 ordinary packet replay of exactly the committed token history. Fixed serial prompt chunking, prefix cache disabled. Target live KV/scales, materialized GDN, convolution, position and the latest target hidden row compared byte for byte, including each owner. Deferred log counts must be zero; inactive log payloads and draft cursor/state excluded. Differences are recorded, not hidden behind a tolerance."
                })).unwrap()).unwrap();
            }
        }
    }
    let mut report: serde_json::Value = crate::model::read(&fixture.output).unwrap();
    let consistent = report["rows"].as_array().unwrap().iter().all(|row| {
        row["first_free_divergence"].is_null()
            && row["all_lanes_equal"] == true
            && row["different_lane_states"].as_array().unwrap().is_empty()
            && row["same_history_different_choices"]
                .as_array()
                .unwrap()
                .is_empty()
            && row["same_history_changed_target_state"]
                .as_array()
                .unwrap()
                .is_empty()
    });
    report["status"] = json!(if consistent { "passed" } else { "inconsistent" });
    report["consistent"] = json!(consistent);
    std::fs::write(&fixture.output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    assert!(
        consistent || probe.allow_differences,
        "Ordinary/MTP output or committed target state differs; see {}",
        fixture.output.display()
    );
}
