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
            cuda_graph: crate::execution::CudaGraphMode::Off,
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
            host_weight_cache_bytes: 16usize << 30,
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
    std::fs::write(fixture.output,serde_json::to_vec_pretty(&json!({"owners":2,"reverse_admission":true,"cancelled_arena_reuse":true,"batch_shrink":true,"logits_and_committed_state_exact":true})).unwrap()).unwrap();
}
