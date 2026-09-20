//! What the editorial judge says about the four recorded drafts, against a real Jev.
//!
//! Ignored by default: it needs a key and the network, and it costs a fraction of a cent. Run it
//! when the questions in `chat::judge` change, because a reworded question is a different
//! measurement and nothing else will notice.
//!
//! ```text
//! TYPESAFE_API_KEY=… cargo test -p ghostreel-core --test eval_judge -- --ignored --nocapture
//! ```
//!
//! The drafts are replayed through the repair passes first, so the editorial read and the
//! mechanical score in `eval_replay` describe the same cut rather than two different films.

use ghostreel_core::chat::{self, metrics};
use ghostreel_core::config::JevConfig;
use ghostreel_core::script::Script;

mod common;
use common::fixture_project;

const BRIEF: &str = "Make a 40 second piece about the Northwest Hills neighbourhood in Austin, \
                     using the interviews with the people who live there. Show what the place is \
                     actually like, in their own words.";

fn drafts() -> [(&'static str, &'static str); 4] {
    [
        ("bonsai27b", include_str!("fixtures/eval/bonsai27b-talking-heads.json")),
        ("qwen35", include_str!("fixtures/eval/qwen35-alternating.json")),
        ("agy", include_str!("fixtures/eval/agy-four-voices.json")),
        ("bonsai2", include_str!("fixtures/eval/bonsai2-bedded.json")),
    ]
}

/// Judge every recorded draft and print both scores side by side.
///
/// The two numbers measure different things and are expected to disagree: the mechanical score
/// says whether a cut is well made, the editorial one whether it is worth watching. The first
/// run of this made the point — the draft with the best mechanical score, 91, had the *worst*
/// picture-to-voice match of the four, which is exactly the fault the editor reported by ear.
#[test]
#[ignore = "needs TYPESAFE_API_KEY and the network"]
fn the_judge_reads_every_recorded_draft() {
    let cfg = JevConfig { enabled: true, ..Default::default() };
    if ghostreel_core::jev::Jev::from_config(&cfg).is_none() {
        panic!("set TYPESAFE_API_KEY to run this");
    }
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let script_cfg = ghostreel_core::config::ScriptConfig::default();

    let mut any_mismatch = false;
    for (name, raw) in drafts() {
        let tmp = tempfile::tempdir().unwrap();
        let (db, project_id) = fixture_project(tmp.path());
        let project = db.project(project_id).unwrap();

        let draft = Script::parse_for_project(raw, &project).unwrap();
        let mut s = draft.clone();
        let mut grounding = chat::Grounding::default();
        for beat in &s.beats {
            for c in &beat.clips {
                grounding.add(c.video_id, c.in_s.max(0.0) - 60.0, c.out_s + 60.0);
            }
            if let Some(bed) = &beat.bed {
                grounding.add(bed.video_id, bed.in_s - 60.0, bed.out_s + 60.0);
            }
        }
        let mut issues = chat::repair_draft(&db, project_id, &mut s, &grounding, true, &script_cfg);
        issues.extend(chat::repair::finish_script(&db, project_id, &mut s, true, &script_cfg).unwrap().unwrap());
        let mechanical = metrics::score(&metrics::measure(&db, Some(&draft), &s, &issues));

        let planned = chat::judge::plan(&db, &s, Some(BRIEF), &cfg).expect("a judgeable cut");
        let j = rt.block_on(planned.ask()).unwrap_or_else(|e| panic!("{name}: {e}"));

        println!("\n{name}: editorial {:.0}/100, mechanical {:.0}/100", j.total, mechanical.total);
        for p in &j.parts {
            println!("   {:<26} {:.2}", p.name, p.value);
        }
        for m in &j.mismatched {
            println!("   pictures miss the voice in '{}' (p={:.2})", m.beat_id, m.match_p);
            any_mismatch = true;
        }

        assert!(j.total >= 0.0 && j.total <= 100.0, "{name} scored {}", j.total);
        assert!(!j.parts.is_empty(), "{name} came back with no dimensions at all");
        assert!(j.model.starts_with("jev"), "{name} was answered by '{}'", j.model);
    }

    // Every one of these drafts has at least one beat where the picture ignores the voice — the
    // editor said so about each of them. A run that finds none means the questions stopped
    // asking anything, which is the failure mode a wording change introduces silently.
    assert!(any_mismatch, "the judge found nothing wrong with any of four drafts the editor complained about");
}
