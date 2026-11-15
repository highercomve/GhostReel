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
            sc.total,
            m.total_s,
            m.target_s,
            m.speaking_sources,
            m.mid_sentence_cuts,
            m.silent_picture_s,
            m.dropped_clips,
            sc.parts
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
    // 83: the muted breath after a voice under b-roll (`broll_breath_s`) is time the fit cannot
    // take back from shots already at their floor; 0.1 s over its target.
    ("agy", 83.0, 2.5),
    ("bonsai2", 75.0, 2.5),
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

/// The passes have to settle. Repairing a script that has already been through them must leave it
/// exactly as it is — the same beats, the same clips, every edge within a frame — or every `script
/// import` of a saved script, and every chat revision that replays the stored cut, moves it a
/// little more. A tolerance here (it was 5% + 0.5 s) hid passes taking turns on an edge: a short
/// "Yeah." swallowed per repair until the fit dropped a whole quote.
///
/// (Replaying the *fixtures* does move them, by design: they are scripts saved before the
/// sentence guarantee ran unconditionally, so the first pass finishes sentences that were left
/// half-spoken. That is the fix arriving late, not a runaway. What must not move is the pass after
/// that.)
#[test]
fn the_repairs_settle_instead_of_pushing_further_each_time() {
    let mut problems = Vec::new();
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
        let once = s.clone();

        chat::repair_draft(&db, project_id, &mut s, &grounding, true, &cfg);
        chat::repair::finish_script(&db, project_id, &mut s, true, &cfg).unwrap().expect("a script");
        problems.extend(unsettled(name, &once, &s));
    }
    assert!(problems.is_empty(), "the repairs did not settle:\n{}", problems.join("\n"));
}

/// One frame at 25 fps: what a second repair may still move an edge by (rounding to the grid).
const FRAME_S: f64 = 0.04;

/// What changed between `once` and `twice`, which should be nothing: the same beats, the same clips
/// and beds, every edge within a frame.
fn unsettled(name: &str, once: &Script, twice: &Script) -> Vec<String> {
    let mut out = Vec::new();
    let ids = |s: &Script| s.beats.iter().map(|b| b.id.clone()).collect::<Vec<_>>();
    if ids(once) != ids(twice) {
        out.push(format!("{name}: a second repair changed the beats {:?} → {:?}", ids(once), ids(twice)));
        return out;
    }
    let moved = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() > FRAME_S || (a.1 - b.1).abs() > FRAME_S;
    for (a, b) in once.beats.iter().zip(&twice.beats) {
        if a.clips.len() != b.clips.len() {
            out.push(format!("{name}: beat {} went from {} to {} clips", a.id, a.clips.len(), b.clips.len()));
            continue;
        }
        for (i, (x, y)) in a.clips.iter().zip(&b.clips).enumerate() {
            if x.video_id != y.video_id || x.audio != y.audio || moved((x.in_s, x.out_s), (y.in_s, y.out_s)) {
                out.push(format!(
                    "{name}: beat {} clip {i} moved from #{} {:.2}–{:.2} {:?} to #{} {:.2}–{:.2} {:?}",
                    a.id, x.video_id, x.in_s, x.out_s, x.audio, y.video_id, y.in_s, y.out_s, y.audio
                ));
            }
        }
        match (&a.bed, &b.bed) {
            (None, None) => {}
            (Some(x), Some(y)) if !moved((x.in_s, x.out_s), (y.in_s, y.out_s)) => {}
            (x, y) => out.push(format!(
                "{name}: beat {} bed moved from {:?} to {:?}",
                a.id,
                x.as_ref().map(|x| (x.in_s, x.out_s)),
                y.as_ref().map(|y| (y.in_s, y.out_s))
            )),
        }
    }
    out
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

/// The same promise for the other ways a cut is finished: a tool save (`save_script`, `apply_fix`,
/// `read_cut`) and an import or Clips run go through the same chain with their own profile, and a
/// script finished once comes out of a second finish exactly as it went in.
#[test]
fn every_profile_settles_on_the_second_finish() {
    use ghostreel_core::finish::{self, Profile};
    let mut problems = Vec::new();
    for (label, profile) in
        [("save", Profile::save(None)), ("import", Profile::import(None)), ("chat", Profile::chat(true))]
    {
        for (name, raw) in all_drafts() {
            let tmp = tempfile::tempdir().unwrap();
            let (db, project_id) = fixture_project(tmp.path());
            let project = db.project(project_id).unwrap();
            let cfg = ghostreel_core::config::ScriptConfig::default();
            let mut s = Script::parse_for_project(raw, &project).unwrap();
            finish::run(&db, project_id, &mut s, &cfg, &profile).unwrap();
            let once = s.clone();
            let notes = finish::run(&db, project_id, &mut s, &cfg, &profile).unwrap();
            problems.extend(unsettled(&format!("{name} ({label})"), &once, &s));
            if s == once && !notes.is_empty() {
                problems.push(format!("{name} ({label}): unchanged, but reported {notes:?}"));
            }
        }
    }
    assert!(problems.is_empty(), "a finish did not settle:\n{}", problems.join("\n"));
}

/// One finish, whichever editor delivers the cut: a chat turn's review of a draft (the rules that
/// need the turn, then the chain) saves exactly what an MCP editor's `save_script` of the same draft
/// saves — the shake, the clip length, the project's footage, the sound and the narration are the
/// chain's for both. (The test this replaces compared `finish::run` with itself.) And an import is
/// the import profile of the same chain.
#[tokio::test]
async fn a_draft_finishes_the_same_however_it_is_delivered() {
    use ghostreel_core::finish::{self, Profile};
    for (name, raw) in all_drafts() {
        let tmp = tempfile::tempdir().unwrap();
        let (mut db, project_id) = fixture_project(tmp.path());
        let project = db.project(project_id).unwrap();
        let config = ghostreel_core::config::Config::default();
        let cfg = config.script.clone();
        let mut draft = Script::parse_for_project(raw, &project).unwrap();
        // And the rules a model can miss, which the recorded drafts happen not to: a closing quote
        // far longer than a clip may be, with narration written over the speaker.
        let mut extra = draft.beats[0].clone();
        extra.id = "extra".into();
        extra.narration = Some("words read over somebody speaking".into());
        extra.bed = None;
        extra.clips.truncate(1);
        // Footage of that video no other clip uses, so it is not dropped as a repeat.
        let video = extra.clips[0].video_id;
        let used = draft.beats.iter().flat_map(|b| &b.clips).filter(|c| c.video_id == video);
        let from = used.map(|c| c.out_s).fold(0.0, f64::max) + 5.0;
        let c = &mut extra.clips[0];
        c.audio = ghostreel_core::script::Audio::Source;
        c.sentences = None;
        (c.in_s, c.out_s) = (from, from + cfg.max_clip_s + 8.0);
        draft.beats.push(extra);

        let mut grounding = chat::Grounding::default();
        for beat in &draft.beats {
            for c in &beat.clips {
                grounding.add(c.video_id, c.in_s.max(0.0) - 60.0, c.out_s + 60.0);
            }
            if let Some(bed) = &beat.bed {
                grounding.add(bed.video_id, bed.in_s - 60.0, bed.out_s + 60.0);
            }
        }
        let reviewed = chat::repair::finish_as_reviewed(&db, &project, &draft, &grounding, false, &cfg)
            .unwrap()
            .unwrap_or_else(|| panic!("{name}: nothing left"));

        let paths = ghostreel_core::paths::Paths::in_dir(tmp.path());
        let mut embedder = None;
        let args = serde_json::json!({ "script": serde_json::to_value(&draft).unwrap() });
        let out = {
            let mut ctx = ghostreel_core::tools::ToolCtx::new(
                &paths,
                &config,
                &mut db,
                &mut embedder,
                ghostreel_core::tools::Origin::External,
            )
            .with_project(project.clone());
            ghostreel_core::tools::call(&mut ctx, "save_script", &args).await.unwrap_or_else(|e| panic!("{name}: {e}"))
        };
        let id = out.value["script_id"].as_i64().unwrap_or_else(|| panic!("{name}: not saved: {}", out.value));
        let saved = ghostreel_core::script::load(&db, id).unwrap().script;
        // Both as stored: a version is JSON, and reading a float back may differ from it in the last bit.
        let reviewed: Script = serde_json::from_str(&serde_json::to_string_pretty(&reviewed).unwrap()).unwrap();
        assert_eq!(reviewed, saved, "{name}: a chat turn's review and a tool save finished differently");

        let mut imported = draft.clone();
        ghostreel_core::script_check::repair_for_import(&db, project_id, &mut imported, &cfg, None).unwrap();
        let mut run = draft.clone();
        finish::run(&db, project_id, &mut run, &cfg, &Profile::import(None)).unwrap();
        assert_eq!(imported, run, "{name}: an import is not the import profile of the one chain");
    }
}
