//! Fixed-work text inference through real request state and dynamic packets.
use super::*;

#[derive(Deserialize)]
struct Workload {
    name: String,
    prompts: Vec<Vec<u32>>,
    output_tokens: usize,
    #[serde(default)]
    concurrent_prefill: bool,
}

#[derive(Deserialize)]
struct Benchmark {
    model: PathBuf,
    output: PathBuf,
    cases: Vec<Workload>,
    repetitions: usize,
    cuda_graph: String,
    #[serde(default)]
    resident_experts: Option<usize>,
}

#[test]
#[ignore = "Requires a real model, frozen requests and exclusive GPU experiment lock"]
fn benchmark_fixed_requests() {
    let spec: Benchmark = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!spec.output.exists() && spec.repetitions > 0);
    assert!(spec.cases.iter().all(|case| {
        (1..=128).contains(&case.prompts.len())
            && case.output_tokens > 0
            && (!case.concurrent_prefill || case.output_tokens == 1)
    }));
    let mut model = ModelRuntime::load_with_options(
        &spec.model,
        LoadOptions {
            cuda_graph: spec.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            resident_experts: spec.resident_experts.map_or(
                crate::execution::ResidentExperts::Full,
                crate::execution::ResidentExperts::Count,
            ),
            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options {
        policy: scheduler::Policy::Throughput,
        memory_reserve_bytes: 0,
        ..Default::default()
    };
    let mut records = Vec::new();
    for case in &spec.cases {
        let mut expected = None;
        for repetition in 0..=spec.repetitions {
            let started = Instant::now();
            let mut requests: Vec<_> = case
                .prompts
                .iter()
                .map(|prompt| {
                    model
                        .start_request(
                            GenerationInput {
                                audios: vec![],
                                input_tokens: prompt.clone(),
                                images: vec![],
                                max_new_tokens: case.output_tokens,
                                sampling: crate::sampling::Options {
                                    temperature: 0.,
                                    seed: 20261002,
                                    ..Default::default()
                                },
                                prefix_hints: vec![],
                            },
                            &|| false,
                        )
                        .unwrap()
                })
                .collect();
            let mut tokens = vec![Vec::new(); requests.len()];
            // Decode cohorts hold serial first outputs at a barrier. Separate
            // prefill-only cases advance the waiting owners together.
            if case.concurrent_prefill {
                while tokens.iter().any(Vec::is_empty) {
                    let waiting: Vec<_> = tokens.iter().map(Vec::is_empty).collect();
                    let owners: Vec<_> = (0..requests.len()).filter(|&i| waiting[i]).collect();
                    let mut refs: Vec<_> = requests
                        .iter_mut()
                        .enumerate()
                        .filter_map(|(i, request)| waiting[i].then_some(request))
                        .collect();
                    for step in model.advance_requests(&mut refs, &options).unwrap() {
                        let owner = owners[step.request];
                        assert!(tokens[owner].is_empty() && step.tokens.len() == 1);
                        tokens[owner].extend(step.tokens);
                    }
                    assert!(requests.iter().all(|request| request.failure().is_none()));
                }
            } else {
                for (request, output) in requests.iter_mut().zip(&mut tokens) {
                    while output.is_empty() {
                        for step in model.advance_requests(&mut [request], &options).unwrap() {
                            output.extend(step.tokens);
                        }
                    }
                }
            }
            for (request, output) in requests.iter().zip(&tokens) {
                assert_eq!(output.len(), 1);
                assert_eq!(request.prefix.statistics.cached_tokens, 0);
            }
            let first_s = started.elapsed().as_secs_f64();
            let decode_start = Instant::now();
            let mut last_s = first_s;
            for _ in 1..case.output_tokens {
                let mut refs: Vec<_> = requests.iter_mut().collect();
                let steps = model.advance_requests(&mut refs, &options).unwrap();
                last_s = started.elapsed().as_secs_f64();
                let mut seen = vec![false; requests.len()];
                for step in steps {
                    assert!(!seen[step.request] && step.tokens.len() == 1);
                    seen[step.request] = true;
                    tokens[step.request].extend(step.tokens);
                }
                assert!(seen.iter().all(|&value| value));
            }
            let decode_s = decode_start.elapsed().as_secs_f64();
            let wall_s = started.elapsed().as_secs_f64();
            assert!(tokens.iter().all(|v| v.len() == case.output_tokens));
            assert!(
                requests
                    .iter()
                    .all(|r| { r.is_finished() && r.failure().is_none() && r.mtp.rounds == 0 })
            );
            if let Some(previous) = &expected {
                assert_eq!(&tokens, previous, "Seeded output changed: {}", case.name);
            } else {
                expected = Some(tokens.clone());
            }
            let input_tokens = case.prompts.iter().map(Vec::len).sum::<usize>();
            let subsequent_tokens = requests.len() * (case.output_tokens - 1);
            records.push(json!({
                "scenario":case.name,"repetition":repetition,"warmup":repetition==0,
                "requests":requests.len(),"input_tokens":input_tokens,
                "concurrent_prefill":case.concurrent_prefill,
                "output_tokens_each":case.output_tokens,"tokens":tokens,
                "first_token_s":first_s,"last_token_s":last_s,"wall_s":wall_s,
                "prefill_tps":(requests.len()==1 || case.concurrent_prefill).then(||input_tokens as f64/first_s),
                "decode_s":decode_s,"decode_tokens":subsequent_tokens,
                "decode_tps":(subsequent_tokens>0).then(||subsequent_tokens as f64/decode_s),
            }));
            for request in &mut requests {
                model.finish_request(request, false).unwrap();
            }
            std::fs::write(
                &spec.output,
                serde_json::to_vec_pretty(&json!({
                    "scope":"Real complete text requests, no MTP/prefix cache, greedy seed20261002, EOS ignored. Prefill includes request setup/head/sampling. Concurrent prefill cases submit all waiting owners until every first output commits. Decode cohorts use a serial first-output barrier and measure all subsequent committed tokens; their serial prefill is not a simultaneous-prefill throughput measurement.",
                    "resident_experts":spec.resident_experts,
                    "records":records,
                }))
                .unwrap(),
            )
            .unwrap();
            eprintln!(
                "BENCH {} repetition={repetition} decode_tps={}",
                case.name,
                subsequent_tokens as f64 / decode_s
            );
        }
    }
}
