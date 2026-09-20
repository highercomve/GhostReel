//! Replay real drafts through the repair pipeline and measure what comes out.
//!
//! Every conclusion about the script chat before this was drawn from watching a preview, which
//! made the pipeline unmeasurable: a change could quietly break a cut and nothing would say so
//! until someone happened to watch the right ten seconds. These are the drafts four brains
//! actually produced in one afternoon — a wall of talking heads, an alternating cut, a
//! four-voice piece, a bedded one — replayed against the footage they refer to.
//!
//! No model, no GPU, no network: the drafts are recorded, and the repair passes are the thing
//! under test.

use ghostreel_core::chat::{self, metrics};
use ghostreel_core::script::Script;

mod common;
use common::fixture_project;

/// One recorded draft, repaired exactly as a turn would repair it.
fn replay(name: &str, raw: &str) -> metrics::ScriptMetrics {
    let tmp = tempfile::tempdir().unwrap();
    let (db, project_id) = fixture_project(tmp.path());
    let project = db.project(project_id).unwrap();
    let cfg = ghostreel_core::config::ScriptConfig::default();

    let draft: Script = Script::parse_for_project(raw, &project).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut s = draft.clone();

    // Everything the model looked at is legal to cut: a replay cannot know what it opened, and
    // grounding is not what these fixtures are testing.
    let mut grounding = chat::Grounding::default();
    for beat in &s.beats {
        for c in &beat.clips {
            grounding.add(c.video_id, c.in_s.max(0.0) - 60.0, c.out_s + 60.0);
        }
        if let Some(bed) = &beat.bed {
            grounding.add(bed.video_id, bed.in_s - 60.0, bed.out_s + 60.0);
        }
    }

    let mut issues = chat::repair_draft(&db, project_id, &mut s, &grounding, true, &cfg);
    match chat::repair::finish_script(&db, project_id, &mut s, true, &cfg).unwrap() {
        Some(more) => issues.extend(more),
        None => panic!("{name}: every clip was dropped"),
    }
    metrics::measure(&db, Some(&draft), &s, &issues)
}

macro_rules! drafts {
    ($($name:literal => $file:literal),* $(,)?) => {
        [$(($name, include_str!(concat!("fixtures/eval/", $file, ".json")))),*]
    };
}

fn all_drafts() -> [(&'static str, &'static str); 4] {
    drafts! {
        "bonsai27b" => "bonsai27b-talking-heads",
        "qwen35" => "qwen35-alternating",
        "agy" => "agy-four-voices",
        "bonsai2" => "bonsai2-bedded",
    }
}

/// Print what each draft measures, so the numbers in this file can be checked rather than
/// trusted: `cargo test -p ghostreel-core --test eval_replay -- --nocapture measured`.
#[test]
fn measured() {
    for (name, raw) in all_drafts() {
        let m = replay(name, raw);
        let sc = metrics::score(&m);
        println!(
            "{name:<10} {:.0} pts  {:.1}s/{:?}  {} voices  {} cuts  {:.1}s silent  {} dropped  {:?}",
            sc.total, m.total_s, m.target_s, m.speaking_sources, m.mid_sentence_cuts,
            m.silent_picture_s, m.dropped_clips, sc.parts
        );
    }
}

/// The promise the whole pipeline exists to keep. Every one of these drafts had a clip or a bed
/// that stopped while someone was still speaking; not one may come out of the repairs that way.
#[test]
fn no_replayed_draft_leaves_anyone_half_spoken() {
    for (name, raw) in all_drafts() {
        let m = replay(name, raw);
        assert_eq!(m.mid_sentence_cuts, 0, "{name} still cuts {} time(s) mid-sentence", m.mid_sentence_cuts);
    }
}

/// What each draft measures today. A change that moves one of these is either a regression or an
/// improvement, and either way it should be looked at rather than discovered in a preview weeks
/// later. Scores are held to a floor rather than an exact value, so ordinary drift does not fail
/// the build.
const BASELINE: &[(&str, f64, f64)] = &[
    // name, lowest acceptable score, most silent picture (s)
    ("bonsai27b", 20.0, 10.0),
    ("qwen35", 33.0, 2.5),
    ("agy", 84.0, 2.5),
    ("bonsai2", 50.0, 2.5),
];

/// B-roll left hanging after the voice stops reads as a pause between interviews. Where a beat
/// has a voice to carry it, the pictures give way and only the closing hold is left; the wall of
/// talking heads has beats with nothing to hear over them at all, which no repair can invent.
#[test]
fn pictures_are_not_left_hanging_in_silence() {
    for (name, raw) in all_drafts() {
        let m = replay(name, raw);
        let (_, _, allowed) = BASELINE.iter().find(|(n, _, _)| *n == name).expect("a baseline");
        assert!(
            m.silent_picture_s <= *allowed,
            "{name} plays {:.1} s of silent picture, against {allowed:.1} s before",
            m.silent_picture_s
        );
    }
}

/// A floor per draft, not a quality bar: falling through one means a pass stopped working.
#[test]
fn every_replayed_draft_survives_the_repairs() {
    for (name, raw) in all_drafts() {
        let m = replay(name, raw);
        let score = metrics::score(&m);
        let (_, floor, _) = BASELINE.iter().find(|(n, _, _)| *n == name).expect("a baseline");
        assert!(m.beats > 0 && m.clips > 0, "{name} came out empty");
        assert_eq!(m.errors, 0, "{name} finished with errors: {:?}", score.parts);
        assert!(score.total >= *floor, "{name} scored {:.0}, under its {floor:.0}: {:?}", score.total, score.parts);
    }
}

/// The repairs must not quietly gut a draft: a cut assembled from a third of what the model chose
/// is not the cut it wrote.
#[test]
fn the_repairs_keep_most_of_what_the_model_chose() {
    for (name, raw) in all_drafts() {
        let m = replay(name, raw);
        let kept = m.clips as f64 / (m.clips + m.dropped_clips) as f64;
        assert!(kept > 0.5, "{name} kept only {:.0}% of its clips", kept * 100.0);
    }
}

/// The passes have to settle. Repairing a script that has already been through them should move
/// it hardly at all, or every `script import` of a saved script inflates it a little more.
///
/// (Replaying the *fixtures* does move them, by design: they are scripts saved before the
/// sentence guarantee ran unconditionally, so the first pass finishes sentences that were left
/// half-spoken — the wall of talking heads grows from 83.9 s to 152.9 s doing it. That is the fix
/// arriving late, not a runaway. What must not move is the pass after that.)
#[test]
fn the_repairs_settle_instead_of_pushing_further_each_time() {
    for (name, raw) in all_drafts() {
        let tmp = tempfile::tempdir().unwrap();
        let (db, project_id) = fixture_project(tmp.path());
        let project = db.project(project_id).unwrap();
        let cfg = ghostreel_core::config::ScriptConfig::default();

        let mut s = Script::parse_for_project(raw, &project).unwrap();
        let mut grounding = chat::Grounding::default();
        for beat in &s.beats {
            for c in &beat.clips {
                grounding.add(c.video_id, c.in_s - 60.0, c.out_s + 60.0);
            }
            if let Some(bed) = &beat.bed {
                grounding.add(bed.video_id, bed.in_s - 60.0, bed.out_s + 60.0);
            }
        }

        chat::repair_draft(&db, project_id, &mut s, &grounding, true, &cfg);
        chat::repair::finish_script(&db, project_id, &mut s, true, &cfg).unwrap().expect("a script");
        let once = s.total_duration_s();

        chat::repair_draft(&db, project_id, &mut s, &grounding, true, &cfg);
        chat::repair::finish_script(&db, project_id, &mut s, true, &cfg).unwrap().expect("a script");
        let twice = s.total_duration_s();

        // 5%: what is left after pad_speech, end_on_sentences and the closing hold were each
        // taught to notice work already done — it was 82% before that. The residue is a second or
        // so on a bedded cut, worth tightening but not worth blocking on.
        assert!(
            (twice - once).abs() <= once * 0.05 + 0.5,
            "{name} moved from {once:.1} s to {twice:.1} s on a repair that should have found nothing to do"
        );
    }
}

/// Write each draft out as the pipeline finishes it, for judging outside this process.
///
/// `GHOSTREEL_EVAL_DUMP=<dir> cargo test -p ghostreel-core --test eval_replay -- --ignored dump`.
/// The mechanical score and the editorial one have to be read on the same cut, and the fixtures
/// are the drafts *before* the repairs — so judging those would compare two different films.
#[test]
#[ignore = "writes files; only useful with GHOSTREEL_EVAL_DUMP set"]
fn dump_repaired_scripts() {
    let Some(dir) = std::env::var_os("GHOSTREEL_EVAL_DUMP") else {
        panic!("set GHOSTREEL_EVAL_DUMP to a directory");
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();

    for (name, raw) in all_drafts() {
        let tmp = tempfile::tempdir().unwrap();
        let (db, project_id) = fixture_project(tmp.path());
        let project = db.project(project_id).unwrap();
        let cfg = ghostreel_core::config::ScriptConfig::default();

        let draft: Script = Script::parse_for_project(raw, &project).unwrap();
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
        let mut issues = chat::repair_draft(&db, project_id, &mut s, &grounding, true, &cfg);
        issues.extend(chat::repair::finish_script(&db, project_id, &mut s, true, &cfg).unwrap().unwrap());

        let m = metrics::measure(&db, Some(&draft), &s, &issues);
        std::fs::write(dir.join(format!("{name}.json")), serde_json::to_string_pretty(&s).unwrap()).unwrap();
        println!("{name}: {:.0} pts mechanical, {:.1} s", metrics::score(&m).total, m.total_s);
    }
}
