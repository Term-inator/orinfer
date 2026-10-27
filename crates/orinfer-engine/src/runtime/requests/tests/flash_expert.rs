//! Full-model causal partitions exercise experts with more than one routed tile.
use super::*;

#[test]
#[ignore = "Requires the native Flash package and exclusive GPU experiment lock"]
fn validate_flash_expert_route_consistency() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            resident_experts: crate::execution::ResidentExperts::Count(8),
            host_weight_cache_bytes: 16usize << 30,
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let capture = |model: &mut ModelRuntime, source: &GenerationInput, chunks: &[usize]| {
        assert_eq!(chunks.iter().sum::<usize>(), source.input_tokens.len());
        let mut request = model.start_request(input(source, 1), &|| false).unwrap();
        for &chunk in chunks {
            model
                .with_request(&mut request, |m, r| m.advance_prefill(r, chunk, false))
                .unwrap();
        }
        let scores = flash_batch::distribution(model, &mut request);
        let state = model
            .with_request(&mut request, |m, _| {
                m.prefix_ranges(source.input_tokens.len())?
                    .into_iter()
                    .map(|(name, range)| {
                        let raw = m
                            .execution
                            .download_bytes(&name, range.offset + range.bytes)?;
                        Ok((name, crate::artifact::sha256(&raw[range.offset..])))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()
            })
            .unwrap();
        model.finish_request(&mut request, false).unwrap();
        (scores, state)
    };
    let source = &fixture.cases[0];
    assert_eq!(source.input_tokens.len(), 31);
    let reference = capture(&mut model, source, &[31]);
    let mut evidence = vec![];
    for chunks in [vec![30, 1], vec![17, 14], vec![1; 31], vec![31]] {
        let actual = capture(&mut model, source, &chunks);
        assert_eq!(
            actual.0, reference.0,
            "Logits changed with chunks {chunks:?}"
        );
        assert_eq!(
            actual.1, reference.1,
            "Committed state changed with chunks {chunks:?}"
        );
        evidence
            .push(json!({"chunks":chunks,"logits_equal":true,"all_committed_state_equal":true}));
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&evidence).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires the native Flash package and exclusive GPU experiment lock"]
fn validate_flash_compact_expert_owners() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            resident_experts: crate::execution::ResidentExperts::Count(8),
            // Force real host eviction and CPU/GPU coverage transitions.
            host_weight_cache_bytes: 256usize << 20,
            cuda_graph: crate::execution::CudaGraphMode::Off,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let first = input(&fixture.cases[0], 8);
    let mut second = input(&fixture.cases[0], 8);
    second.input_tokens.push(32);
    let sources = [first, second];
    let options = scheduler::Options::default();
    let capture = |model: &mut ModelRuntime, request: &mut RequestState| {
        let scores = flash_batch::distribution(model, request);
        let committed = model
            .with_request(request, |m, _| {
                let length = m.execution.read_control(&m.manifest.position)? as usize;
                m.prefix_ranges(length)?
                    .into_iter()
                    .map(|(name, range)| {
                        let raw = m
                            .execution
                            .download_bytes(&name, range.offset + range.bytes)?;
                        Ok((name, crate::artifact::sha256(&raw[range.offset..])))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()
            })
            .unwrap();
        (scores, committed)
    };
    let mut reference = vec![];
    let regions = std::mem::take(&mut *model.execution.parallel_regions.borrow_mut());
    for source in &sources {
        let mut request = model.start_request(input(source, 8), &|| false).unwrap();
        prefill_reference(&mut model, &mut request);
        let mut steps = vec![];
        for _ in 0..3 {
            model
                .advance_requests(&mut [&mut request], &options)
                .unwrap();
            steps.push(capture(&mut model, &mut request));
        }
        model.finish_request(&mut request, false).unwrap();
        reference.push(steps);
    }
    assert_eq!(
        model
            .execution
            .batch_statistics
            .get()
            .expert_parallel_branches,
        0
    );
    *model.execution.parallel_regions.borrow_mut() = regions;
    model
        .execution
        .set_graph_mode_for_test(fixture.cuda_graph.parse().unwrap());
    for order in [[0, 1], [1, 0]] {
        // A cancelled owner must release its arena before the reversed admission.
        let mut cancelled = model
            .start_request(input(&sources[0], 8), &|| false)
            .unwrap();
        prefill_reference(&mut model, &mut cancelled);
        model.finish_request(&mut cancelled, false).unwrap();
        let mut requests = vec![];
        for &owner in &order {
            let mut request = model
                .start_request(input(&sources[owner], 8), &|| false)
                .unwrap();
            prefill_reference(&mut model, &mut request);
            requests.push(request);
        }
        for step in [0, 1] {
            let mut output = vec![];
            model
                .execute_selected(
                    &mut requests.iter_mut().collect::<Vec<_>>(),
                    &[(0, 1), (1, 1)],
                    &mut output,
                )
                .unwrap();
            assert_eq!(output.len(), 2);
            for (lane, &owner) in order.iter().enumerate() {
                let actual = capture(&mut model, &mut requests[lane]);
                assert_eq!(
                    actual, reference[owner][step],
                    "owner {owner} lane {lane} step {step}"
                );
            }
        }
        model.finish_request(&mut requests[1], false).unwrap();
        model
            .advance_requests(&mut [&mut requests[0]], &options)
            .unwrap();
        assert_eq!(
            capture(&mut model, &mut requests[0]),
            reference[order[0]][2]
        );
        model.finish_request(&mut requests[0], false).unwrap();
    }
    let branches = model
        .execution
        .batch_statistics
        .get()
        .expert_parallel_branches;
    assert!(
        branches > 0,
        "Finite expert validation must execute independent branches"
    );
    let stats = model.execution.batch_statistics.get();
    if fixture.cuda_graph != "off" {
        assert!(
            stats.shared_graph_captures > 0,
            "No finite GPU pieces captured"
        );
        assert!(stats.shared_graph_hits > 0, "No finite GPU pieces replayed");
        assert_eq!(stats.dynamic_graph_captures, 0);
        assert!(model.execution.graphs.borrow().is_empty());
    }
    let host = model.expert_cache_statistics().unwrap().host_cache;
    assert!(host.evictions > 0 && host.gpu_priority_evictions > 0);
    assert_eq!(host.pending_bytes, 0);
    assert!(host.resident_bytes <= host.capacity_bytes);
    std::fs::write(fixture.output,serde_json::to_vec_pretty(&json!({"owners":2,"reverse_admission":true,"cancelled_arena_reuse":true,"batch_shrink":true,"logits_and_committed_state_exact":true,"serial_reference":true,"host_cache_pressure":true,"gpu_priority_evictions":host.gpu_priority_evictions,"expert_parallel_branches":branches})).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires the native Flash package and exclusive GPU experiment lock"]
fn validate_flash_expert_parallel_read_failure() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            resident_experts: crate::execution::ResidentExperts::Count(1),
            host_weight_cache_bytes: 0,
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let mut request = model
        .start_request(input(&fixture.cases[0], 8), &|| false)
        .unwrap();
    prefill_reference(&mut model, &mut request);
    if fixture.cuda_graph != "off" {
        for _ in 0..3 {
            model
                .advance_requests(&mut [&mut request], &scheduler::Options::default())
                .unwrap();
        }
        let stats = model.execution.batch_statistics.get();
        assert!(stats.shared_graph_captures > 0);
        assert!(stats.shared_graph_hits > 0);
    } else {
        assert!(model.execution.session.parallel_handles.borrow().is_none());
    }
    let directory = tempfile::tempdir().unwrap();
    model.execution.fail_expert_reads_for_test(directory.path());
    let error = model
        .advance_requests(&mut [&mut request], &scheduler::Options::default())
        .unwrap_err();
    assert!(error.contains("missing-expert"), "{error}");
    let (side, _, _) = (*model.execution.session.parallel_handles.borrow())
        .expect("Independent branch was not submitted");
    // SAFETY: The live model owns this stream; failed execution must drain it
    // before finish_request can release the owner's arena.
    unsafe {
        crate::cuda::check(
            crate::cuda::sys::cuStreamQuery(side),
            "failed branch drained",
        )
        .unwrap();
    }
    model.finish_request(&mut request, false).unwrap();
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(
            &json!({"missing_read_failed":true,"side_stream_drained":true,"owner_released":true}),
        )
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires real Flash weights and exclusive GPU experiment lock"]
fn validate_flash_sequence_workspace_growth() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            resident_experts: crate::execution::ResidentExperts::Count(8),
            host_weight_cache_bytes: 256usize << 20,
            cuda_graph: crate::execution::CudaGraphMode::Full,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let (name, stride) = model
        .manifest
        .sequence_workspaces
        .iter()
        .max_by_key(|(_, r)| r.bytes_per_row)
        .map(|(n, r)| (n.clone(), r.bytes_per_row))
        .expect("QSA row contract");
    let address = model.execution.pointers[&name];
    let capacity = model.execution.sizes[&name];
    assert_eq!(
        model.execution.session.workspace_buffers.borrow()[&name].mapped,
        0
    );
    let mut sources = vec![];
    for (length, first) in [(31, 32), (1057, 33), (31, 34), (1057, 35)] {
        let mut source = input(&fixture.cases[0], 1);
        source.input_tokens = source
            .input_tokens
            .iter()
            .copied()
            .cycle()
            .take(length)
            .collect();
        source.input_tokens[0] = first;
        sources.push(source);
    }
    let capture = |model: &mut ModelRuntime, source: &GenerationInput| {
        let mut request = model.start_request(input(source, 1), &|| false).unwrap();
        model
            .with_request(&mut request, |m, r| {
                m.advance_prefill(r, source.input_tokens.len(), false)
            })
            .unwrap();
        let scores = flash_batch::distribution(model, &mut request);
        let state = model
            .with_request(&mut request, |m, _| {
                m.prefix_ranges(source.input_tokens.len())?
                    .into_iter()
                    .map(|(name, range)| {
                        let raw = m
                            .execution
                            .download_bytes(&name, range.offset + range.bytes)?;
                        Ok((name, crate::artifact::sha256(&raw[range.offset..])))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()
            })
            .unwrap();
        model.finish_request(&mut request, false).unwrap();
        (scores, state)
    };
    let mut reference = vec![];
    let mut evidence = vec![];
    let mut peak = 0;
    for source in &sources {
        reference.push(capture(&mut model, source));
        let buffers = model.execution.session.workspace_buffers.borrow();
        let r = &buffers[&name];
        peak = peak.max(
            r.required_extent(source.input_tokens.len() * stride)
                .unwrap(),
        );
        // required_extent consumes rows when stride is one for ordinary workspace.
        assert_eq!(r.mapped, peak);
        assert!(r.mapped < capacity);
        assert_eq!(model.execution.pointers[&name], address);
        evidence.push(json!({"rows":source.input_tokens.len(),"mapped_bytes":r.mapped,"capacity_bytes":capacity}));
    }
    model.execution.ensure_workspace(&name, capacity).unwrap();
    for (source, expected) in sources.iter().zip(reference) {
        let actual = capture(&mut model, source);
        assert_eq!(
            actual.0, expected.0,
            "Full logits changed after full mapping"
        );
        assert_eq!(
            actual.1, expected.1,
            "Request state changed after full mapping"
        );
    }
    let statistics = model.execution.batch_statistics.get();
    assert!(statistics.shared_graph_hits > 0 && statistics.shared_graph_captures > 0);
    std::fs::write(fixture.output,serde_json::to_vec_pretty(&json!({"rows":evidence,"full_logits_equal":true,"all_state_equal":true,"stable_address":true,"shared_graph_hits":statistics.shared_graph_hits,"shared_graph_captures":statistics.shared_graph_captures})).unwrap()).unwrap();
}
