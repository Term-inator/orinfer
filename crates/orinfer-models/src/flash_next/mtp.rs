//! Native Flash HC speculation and complete recurrent/index prefix commit.
use super::plan::section;
use crate::{
    artifact::Result,
    model::{KvGrowth, Manifest, Operation},
};
use std::collections::BTreeSet;

pub(super) fn build(manifest: &mut Manifest) -> Result<()> {
    let Some(spec) = manifest.mtp.clone() else {
        return Ok(());
    };
    if spec.position != "MtpPosition"
        || spec.input != "MtpInput"
        || spec.token != "MtpToken"
        || spec.status != "MtpStatus"
        || spec.verification_tokens != "VerificationTokens"
        || spec.verification_status != "VerificationStatus"
        || spec.draft_logits != "DraftLogits"
        || spec.verification_logits != "VerificationLogits"
        || spec.accepted_inputs != "AcceptedInputs"
        || spec.target_length != "PositionOut"
        || spec.draft_program != "mtp_draft"
        || spec.hidden_ring.as_deref() != Some("MtpHiddenRing")
        || spec.feature_index.as_deref() != manifest.vision.as_ref().map(|_| "MtpFeatureIndex")
        || !spec.commit_always
        || spec.draft_snapshot_program.as_deref() != Some("mtp_snapshot")
        || spec.draft_restore_program.as_deref() != Some("mtp_restore_draft")
        || spec.default_verification_tokens != 4
    {
        return Err("Unsupported Flash MTP control/state contract".into());
    }
    let capture_widths: BTreeSet<_> = spec.capture_plans.iter().map(|p| p.tokens).collect();
    let verify_widths: BTreeSet<_> = spec.verification_widths().into_iter().collect();
    let dynamic_verify = spec.verification_plans.len() == 1
        && spec.verification_plans[0].min_tokens == 2
        && spec.verification_plans[0].tokens == 8;
    let expected_capture: BTreeSet<_> = std::iter::once(1)
        .chain(manifest.prefill_plans.iter().map(|p| p.chunk_tokens))
        .collect();
    let dynamic_warm = spec.warm_plans.len() == 1
        && spec.warm_plans[0].min_tokens == 1
        && spec.warm_plans[0].tokens == 512
        && spec.warm_plans[0].program == "mtp_warm";
    if !dynamic_warm
        || !dynamic_verify
        || spec.warm_plans[0].head_program != "mtp_head"
        || capture_widths != expected_capture
        || capture_widths.len() != spec.capture_plans.len()
        || verify_widths != (2..=8).collect()
    {
        return Err("Flash MTP requires dynamic warm and verification templates".into());
    }
    let programs = &mut manifest.programs;
    for p in &spec.verification_plans {
        if p.program != "mtp_verify"
            || p.restore_program != "mtp_restore"
            || p.capture_program != "mtp_capture"
        {
            return Err("Unknown Flash verification program".into());
        }
        let mut commit = vec![];
        for layer in 0..48 {
            commit.extend(section(
                "causal_rows/restore",
                &format!("layer{layer}"),
                if layer == 1 {
                    3
                } else if layer % 4 == 3 {
                    1
                } else {
                    2
                },
                None,
            ));
        }

        programs.insert(p.restore_program.clone(), commit);
    }
    for p in &spec.capture_plans {
        let name = format!("mtp_capture_m{}", p.tokens);
        if p.program != name {
            return Err("Unknown Flash capture program".into());
        }
        programs.insert(name.clone(), section(&name, "body", 1, None));
    }
    for (program, source, destination) in [
        ("mtp_snapshot", "State_48_pending", "DraftPendingSaved"),
        ("mtp_restore_draft", "DraftPendingSaved", "State_48_pending"),
    ] {
        programs.insert(
            program.into(),
            vec![Operation::Copy {
                source: source.into(),
                destination: destination.into(),
                bytes: 1024,
            }],
        );
    }
    let kv = manifest.kv_cache.as_mut().ok_or("Missing Flash KV")?;
    for p in &spec.verification_plans {
        kv.growth.insert(
            p.program.clone(),
            KvGrowth {
                length: (p.min_tokens > 0).then(|| "Length".into()),
                position: manifest.position.clone(),
                tokens: p.tokens,
                buffers: kv
                    .buffers
                    .keys()
                    .filter(|n| !n.starts_with("State_48_"))
                    .cloned()
                    .chain([super::plan::score_role(8, "Verify")])
                    .collect(),
            },
        );
    }
    for p in &spec.warm_plans {
        kv.growth.insert(
            p.program.clone(),
            KvGrowth {
                length: (p.min_tokens > 0).then(|| "MtpLength".into()),
                position: spec.position.clone(),
                tokens: p.tokens,
                buffers: kv
                    .buffers
                    .keys()
                    .filter(|n| n.starts_with("State_48_"))
                    .cloned()
                    .chain([super::plan::score_role(512, "Draft")])
                    .collect(),
            },
        );
    }
    kv.growth.insert(
        spec.draft_program.clone(),
        KvGrowth {
            length: None,
            position: spec.position,
            tokens: 1,
            buffers: kv
                .buffers
                .keys()
                .filter(|n| n.starts_with("State_48_"))
                .cloned()
                .chain([super::plan::score_role(512, "Draft")])
                .collect(),
        },
    );
    Ok(())
}
