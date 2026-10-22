//! Requires a native expert-cache package and the shared GPU lock held by run.sh.
use orinfer_engine::{
    execution::{LoadOptions, ResidentExperts},
    model::{GenerationInput, Model},
    sampling, scheduler,
};
use std::{path::PathBuf, time::Instant};

#[derive(serde::Deserialize)]
struct Fixture {
    model: PathBuf,
    output: PathBuf,
    cases: Vec<Vec<u32>>,
}

fn batch(model: &mut Model, cases: &[Vec<u32>]) -> (Vec<Vec<u32>>, Vec<usize>) {
    let mut requests: Vec<_> = cases
        .iter()
        .map(|input| {
            model
                .start_request(
                    GenerationInput {
                        audios: vec![],
                        input_tokens: input.clone(),
                        images: vec![],
                        max_new_tokens: 8,
                        sampling: sampling::Options {
                            temperature: 0.,
                            seed: 20261002,
                            ..Default::default()
                        },
                        prefix_hints: vec![input.len()],
                    },
                    || false,
                )
                .unwrap()
        })
        .collect();
    let mut tokens = vec![vec![]; cases.len()];
    let scheduler = scheduler::Options {
        memory_reserve_bytes: 256 << 20,
        ..Default::default()
    };
    let began = Instant::now();
    while requests.iter().any(|r| !r.is_finished()) {
        assert!(
            began.elapsed().as_secs() < 600,
            "Expert-cache batch did not finish"
        );
        let mut refs = requests.iter_mut().collect::<Vec<_>>();
        for output in model.advance_requests(&mut refs, &scheduler).unwrap() {
            tokens[output.request].extend(output.tokens);
        }
    }
    let cached = requests
        .iter()
        .map(|r| r.prefix_statistics().cached_tokens)
        .collect();
    for r in &mut requests {
        model.finish_request(r, true).unwrap();
    }
    (tokens, cached)
}

#[test]
#[ignore = "Requires an SM120 native Flash package and exclusive GPU experiment lock"]
fn finite_experts_preserve_mtp_batch_and_prefix_outputs() {
    let fixture: Fixture = serde_json::from_slice(
        &std::fs::read(std::env::var("ORINFER_EXPERT_FIXTURE").unwrap()).unwrap(),
    )
    .unwrap();
    assert!(fixture.cases.len() >= 2);
    let mut baseline = None;
    let mut reports = vec![];
    for mode in [
        ResidentExperts::Full,
        ResidentExperts::Count(1),
        ResidentExperts::Count(256),
    ] {
        let mut model = Model::load_with_options(
            &fixture.model,
            LoadOptions {
                resident_experts: mode,
                prefix_cache_bytes: 512 << 20,
                mtp_drafts: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        let at = Instant::now();
        let (cold, _) = batch(&mut model, &fixture.cases);
        let (warm, hits) = batch(&mut model, &fixture.cases);
        let (single, _) = batch(&mut model, &fixture.cases[..1]);
        let scheduler = model.scheduler_statistics();
        assert!(
            scheduler.speculative_iterations > 0,
            "MTP was not exercised"
        );
        assert_eq!(cold, warm, "Prefix replay changed token choices for {mode}");
        assert!(hits.iter().all(|&n| n > 0), "No prompt cache hit");
        if let Some((reference, reference_single)) = &baseline {
            assert_eq!(
                &cold, reference,
                "Expert paging changed token choices for {mode}"
            );
            assert_eq!(&single, reference_single, "MTP paging changed tokens");
        } else {
            baseline = Some((cold.clone(), single.clone()));
        }
        let statistics = model.expert_cache_statistics();
        if let ResidentExperts::Count(n) = mode {
            let s = statistics.unwrap();
            assert_eq!(s.capacity_experts, n * 48);
            assert!(s.resident_experts <= s.capacity_experts);
            assert!(s.loaded_bytes > 0);
            if n == 1 {
                assert!(s.evictions > 0);
                assert!(
                    s.peak_transient_experts > 0,
                    "Large route union did not exercise transient paging"
                );
            }
        } else {
            assert!(statistics.is_none());
        }
        reports.push(serde_json::json!({"resident_experts":mode,"outputs":cold,"single_outputs":single,"scheduler":scheduler,"warm_cached_tokens":hits,"expert_cache":statistics,"requests_s":at.elapsed().as_secs_f64()}));
        std::fs::write(
            &fixture.output,
            serde_json::to_vec_pretty(&reports).unwrap(),
        )
        .unwrap();
        drop(model);
    }
}
