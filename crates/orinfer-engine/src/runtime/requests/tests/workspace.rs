//! First-touch workspace mapping must preserve graphs and request state.
use super::*;

fn run(model: &mut ModelRuntime, source: &GenerationInput) -> (Vec<u32>, BTreeMap<String, String>) {
    let mut request = model.start_request(input(source, 4), &|| false).unwrap();
    let mut tokens = Vec::new();
    while !request.is_finished() {
        for output in model
            .advance_requests(&mut [&mut request], &scheduler::Options::default())
            .unwrap()
        {
            tokens.extend(output.tokens);
        }
    }
    let state = single_decode::state(model, &mut request);
    model.finish_request(&mut request, false).unwrap();
    (tokens, state)
}

#[test]
#[ignore = "Requires real media model and the exclusive GPU experiment lock"]
fn validate_lazy_workspace_requests() {
    let f: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!f.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &f.model,
        LoadOptions {
            cuda_graph: f.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let addresses: BTreeMap<_, _> = model
        .execution
        .session
        .workspace_buffers
        .borrow()
        .iter()
        .map(|(n, b)| {
            assert_eq!(b.mapped, 0);
            (n.clone(), b.address)
        })
        .collect();
    assert!(!addresses.is_empty() && model.stats.load_stages.deferred_workspace_bytes > 0);
    let text = f
        .cases
        .iter()
        .find(|c| c.images.is_empty() && c.audios.is_empty())
        .expect("Text case");
    let media: Vec<_> = f
        .cases
        .iter()
        .filter(|c| !c.images.is_empty() || !c.audios.is_empty())
        .collect();
    assert!(
        !media.is_empty(),
        "Include an image or audio first-use case"
    );
    let media_input = model
        .manifest
        .vision
        .as_ref()
        .map(|v| v.pixels.clone())
        .or_else(|| model.manifest.audio.as_ref().map(|a| a.mel.clone()))
        .expect("Media adapter");
    let before = run(&mut model, text);
    assert_eq!(
        model.execution.session.workspace_buffers.borrow()[&media_input].mapped,
        0
    );
    assert!(
        model
            .execution
            .ensure_workspace(&media_input, model.execution.sizes[&media_input] + 1)
            .is_err()
    );
    assert_eq!(
        model.execution.session.workspace_buffers.borrow()[&media_input].mapped,
        0
    );
    let deferred_media = model.manifest.audio.is_some();
    if deferred_media {
        // Audio metadata is admitted before encoding; later text admission
        // must retain that recording's still-unmapped shared working memory.
        let source = input(media[0], 4);
        let context = model.validate_generation(&source).unwrap();
        let text_budget = model.pending_workspace(context, false, &[]).unwrap();
        let mut pending_media = model.start_request(source, &|| false).unwrap();
        let media_bytes = model
            .execution
            .pending_workspace_bytes(&["audio_stem", "audio_encoder"])
            .unwrap();
        assert!(media_bytes > 0);
        assert_eq!(
            model.pending_workspace(context, false, &[]).unwrap(),
            text_budget + media_bytes
        );
        let mut later_text = model.start_request(input(text, 4), &|| false).unwrap();
        assert_eq!(
            model.pending_workspace(context, false, &[]).unwrap(),
            text_budget + media_bytes
        );
        model.finish_request(&mut later_text, false).unwrap();
        model.finish_request(&mut pending_media, false).unwrap();
    }
    for source in media {
        run(&mut model, source);
    }
    assert!(model.execution.session.workspace_buffers.borrow()[&media_input].mapped > 0);
    let after = run(&mut model, text);
    assert_eq!(
        before, after,
        "Media first use changed later text tokens/state"
    );
    for (name, buffer) in model.execution.session.workspace_buffers.borrow().iter() {
        assert_eq!(
            buffer.address, addresses[name],
            "Lazy mapping changed graph VA"
        );
    }
    let stats = model.execution.batch_statistics();
    assert!(stats.workspace_mappings > 0 && stats.workspace_mapped_bytes > 0);
    std::fs::write(
        f.output,
        serde_json::to_vec_pretty(&json!({
            "deferred_bytes":model.stats.load_stages.deferred_workspace_bytes,
            "workspace":stats,"text_tokens_exact":true,"text_states_exact":true,
            "media_first_use":true,"stable_addresses":true,"bounds_failure_recovered":true,
            "pending_media_admission_budget":deferred_media,
        }))
        .unwrap(),
    )
    .unwrap();
}
