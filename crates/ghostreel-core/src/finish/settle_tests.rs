//! The chain settles and keeps its promises when a cut goes through it again: probes for the draft
//! rules (the cap, the steady camera, the merge, narration and sound), then a generated fixture
//! that runs many cuts near those rules' thresholds through every profile twice.
//!
//! Each test runs [`run`] — the whole chain — never a pass on its own: a pass that is right alone
//! and undone by the passes after it is how the cap and the steady trim got through before.

use rusqlite::params;

use super::tests::{cut, sentence_at, video};
use super::*;
use crate::script::{Audio, CLOSING_PICTURE, ScriptClip, Side};

/// One frame at 30 fps: what "the same edge" means after the cut placement has settled it.
const FRAME_S: f64 = 1.0 / 30.0;

/// Shake on video 7: one window a second over `0..300`, shaky inside `spans`.
fn shake(db: &Db, spans: &[(f64, f64)]) {
    for i in 0..300 {
        let t = i as f64 + 0.5;
        let jerk = if spans.iter().any(|&(a, b)| t > a && t < b) { 50.0 } else { 0.1 };
        db.conn
            .execute(
                "INSERT INTO motion_windows(video_id, start_s, end_s, jerk, motion, sway) VALUES (7, ?1, ?2, ?3, 0.0, 0.0)",
                params![i as f64, i as f64 + 1.0, jerk],
            )
            .unwrap();
    }
}

/// Whether two finished scripts are the same cut: the same beats, clips, sound and words, every
/// edge within a frame.
fn same_cut(a: &Script, b: &Script) -> Result<(), String> {
    if a.beats.len() != b.beats.len() {
        return Err(format!("{} beats, then {}", a.beats.len(), b.beats.len()));
    }
    let near = |x: f64, y: f64| (x - y).abs() <= FRAME_S;
    for (i, (x, y)) in a.beats.iter().zip(&b.beats).enumerate() {
        if x.clips.len() != y.clips.len() {
            return Err(format!("beat {i}: {} clips, then {}", x.clips.len(), y.clips.len()));
        }
        if x.narration.as_deref().unwrap_or("").trim() != y.narration.as_deref().unwrap_or("").trim() {
            return Err(format!("beat {i}: narration {:?}, then {:?}", x.narration, y.narration));
        }
        for (j, (c, d)) in x.clips.iter().zip(&y.clips).enumerate() {
            if c.video_id != d.video_id || c.audio != d.audio || !near(c.in_s, d.in_s) || !near(c.out_s, d.out_s) {
                return Err(format!("beat {i} clip {j}: {c:?}, then {d:?}"));
            }
        }
        match (&x.bed, &y.bed) {
            (None, None) => {}
            (Some(p), Some(q)) if near(p.in_s, q.in_s) && near(p.out_s, q.out_s) => {}
            (p, q) => return Err(format!("beat {i}: bed {p:?}, then {q:?}")),
        }
    }
    Ok(())
}

/// The share of `in_s..out_s` the camera shakes through, as the steady-camera rule measures it.
fn shaky_share(db: &Db, cfg: &ScriptConfig, in_s: f64, out_s: f64) -> f64 {
    let windows = db.motion_windows(7).unwrap();
    let limit = crate::steadiness::shake_limit(&windows, cfg.max_shake_jerk, cfg.shake_relative);
    let spans = crate::steadiness::shaky_spans_with_sway(&windows, in_s, out_s, limit, cfg.max_sway);
    let shaky: f64 = spans.iter().map(|(a, b)| (b.min(out_s) - a.max(in_s)).max(0.0)).sum();
    shaky / (out_s - in_s).max(1e-9)
}

/// A clip of video 7 with a range, as an editor writes one.
fn ranged(in_s: f64, out_s: f64, from: &str, to: &str) -> ScriptClip {
    serde_json::from_value(serde_json::json!({ "video_id": 7, "in_s": in_s, "out_s": out_s, "from": from,
        "to": to, "audio": "source" }))
    .unwrap()
}

fn profiles() -> [(&'static str, Profile); 3] {
    [("save", Profile::save(None)), ("chat", Profile::chat(false)), ("import", Profile::import(None))]
}

/// Finish `s` twice under `profile`; the second finish must leave the first's cut as it was.
fn finish_twice(db: &Db, project_id: i64, s: &Script, cfg: &ScriptConfig, profile: &Profile) -> (Script, Script) {
    let mut once = s.clone();
    run(db, project_id, &mut once, cfg, profile).unwrap();
    let mut twice = once.clone();
    run(db, project_id, &mut twice, cfg, profile).unwrap();
    (once, twice)
}

/// opus-r3-1: a speaking clip a breath under the cap was cut mid-sentence by the next finish, and
/// the sentence passes then dropped a whole sentence it had chosen (0–29.8 → 0–30.70 → 0–20.35).
/// The cap speaks in sentences and in the length the clip will play: it ends on the last sentence
/// that fits with its tail, once, and the next finish leaves it there.
#[test]
fn the_clip_cap_ends_on_a_sentence_once_and_a_second_finish_keeps_it() {
    let (db, project_id) = video(&[
        (0.0, 10.0, "a one two three four.", false),
        (10.0, 20.0, "b one two three four.", false),
        (20.0, 29.8, "c five six seven eight.", false),
        (31.0, 40.0, "d nine ten eleven twelve.", false),
    ]);
    let cfg = ScriptConfig::default();
    for (name, profile) in [("save", Profile::save(None)), ("chat", Profile::chat(false))] {
        for (in_s, out_s) in [(0.0, 29.8), (0.0, 41.0), (4.0, 39.0)] {
            let s = cut(None, &[&[(in_s, out_s, "source")]]);
            let (once, twice) = finish_twice(&db, project_id, &s, &cfg, &profile);
            same_cut(&once, &twice).unwrap_or_else(|e| panic!("{name} {in_s}–{out_s}: a second finish moved it: {e}"));
            let c = &once.beats[0].clips[0];
            assert!(c.out_s - c.in_s <= cfg.max_clip_s + 1e-9, "{name} {in_s}–{out_s}: over the cap: {c:?}");
            let edges = sentence_edges(&db, 7);
            let ends_on_sentence = edges.iter().any(|&(_, e)| {
                c.out_s >= e + cfg.speech_overrun_s - 1e-6 && edges.iter().all(|&(a, _)| a <= e || a >= c.out_s - 1e-6)
            });
            assert!(ends_on_sentence, "{name} {in_s}–{out_s}: stops inside a sentence: {c:?}");
        }
    }
}

/// opus-r3-2, sol-r3-2: the steady-camera trim of a speaking clip was to raw steady seconds, and
/// the sentence passes grew it back over the shake (4–10 → 0–30.35, the whole shake played). A
/// speaking clip keeps the longest run of whole sentences inside the steady parts, and that trim
/// survives every later pass and a second finish; with no whole sentence steady, the clip goes.
#[test]
fn a_steady_trim_keeps_whole_sentences_and_survives_the_chain() {
    let cfg = ScriptConfig::default();
    // Five sentences, shake 9–20: the clip finishes as 0–18.35, more than half shake, and the first
    // sentence is the steady run in it.
    let (db, project_id) = video(&[
        (0.0, 6.0, "one two three four five.", false),
        (6.0, 12.0, "six seven eight nine ten.", false),
        (12.0, 18.0, "eleven twelve thirteen fourteen.", false),
        (18.0, 24.0, "fifteen sixteen seventeen eighteen.", false),
        (24.0, 30.0, "nineteen twenty twenty one.", false),
        (60.0, 70.0, "something else said much later on.", false),
    ]);
    shake(&db, &[(9.0, 20.0)]);
    for (name, profile) in [("save", Profile::save(None)), ("chat", Profile::chat(false))] {
        let s = cut(None, &[&[(1.0, 17.0, "source")], &[(60.0, 70.0, "source")]]);
        let (once, twice) = finish_twice(&db, project_id, &s, &cfg, &profile);
        same_cut(&once, &twice).unwrap_or_else(|e| panic!("{name}: a second finish moved it: {e}"));
        let c = &once.beats[0].clips[0];
        assert!(c.in_s <= 1e-9 && c.out_s >= 6.0 && c.out_s <= 9.0, "{name}: the first sentence, steady: {c:?}");
        assert!(shaky_share(&db, &cfg, c.in_s, c.out_s) <= 0.5, "{name}: the shake came back: {c:?}");
    }

    // One long sentence, mostly shake: nothing whole is steady, so the clip goes.
    let (db, project_id) = video(&[
        (0.0, 30.0, "one long answer that runs on and on and on.", false),
        (60.0, 70.0, "something else said much later on.", false),
    ]);
    shake(&db, &[(5.0, 25.0)]);
    for (name, profile) in [("save", Profile::save(None)), ("chat", Profile::chat(false))] {
        let s = cut(None, &[&[(4.0, 17.0, "source")], &[(60.0, 70.0, "source")]]);
        let (once, twice) = finish_twice(&db, project_id, &s, &cfg, &profile);
        same_cut(&once, &twice).unwrap_or_else(|e| panic!("{name}: a second finish moved it: {e}"));
        assert_eq!(once.beats.len(), 1, "{name}: the shaky sentence is not in the cut: {once:?}");
        assert!(once.beats[0].clips[0].in_s >= 59.0, "{name}: {once:?}");
    }
}

/// sol-r3-1, hydrogen-r3-1: joining a clip written on a word with a neighbour written in seconds
/// dropped the range, and with it the word edge on the outside: padding then pulled the start back
/// to the sentence. Either order, either profile, a second finish: the word edge stays.
#[test]
fn a_merge_never_loses_an_outer_word_edge() {
    let (db, project_id) = video(&[
        (0.0, 10.0, "the first whole sentence is here now.", false),
        (10.0, 18.0, "and then the second.", false),
    ]);
    let (s1, s2) = (sentence_at(&db, 0.0), sentence_at(&db, 10.0));
    let cfg = ScriptConfig { infer_audio_beds: false, ..ScriptConfig::default() };
    let plain = |a: f64, b: f64| ScriptClip { sentences: None, ..ranged(a, b, "s1", "s1") };
    // The word-written clip first (its start is pinned), and second (its end is pinned).
    let first = cut(None, &[&[(3.0, 10.0, "source")]]);
    let mut start_pinned = first.clone();
    start_pinned.beats[0].clips = vec![ranged(3.0, 10.0, &format!("{s1}@3"), &s1), plain(10.0, 18.0)];
    let mut end_pinned = first;
    end_pinned.beats[0].clips = vec![plain(0.0, 10.0), ranged(10.0, 15.0, &s2, &format!("{s2}@2"))];
    for (name, profile) in [("save", Profile::save(None)), ("chat", Profile::chat(false))] {
        let (once, twice) = finish_twice(&db, project_id, &start_pinned, &cfg, &profile);
        for (pass, s) in [("first", &once), ("second", &twice)] {
            let c = &s.beats[0].clips[0];
            assert_eq!(c.in_s, 3.0, "{name}, {pass} finish: the start written on a word moved: {s:?}");
            assert!(!c.movable(Side::In), "{name}, {pass} finish: the start is no longer pinned: {c:?}");
        }
        let (once, twice) = finish_twice(&db, project_id, &end_pinned, &cfg, &profile);
        for (pass, s) in [("first", &once), ("second", &twice)] {
            let c = s.beats[0].clips.iter().rfind(|c| c.why.as_deref() != Some(CLOSING_PICTURE)).unwrap();
            assert_eq!(c.out_s, 15.0, "{name}, {pass} finish: the end written on a word moved: {s:?}");
            assert!(!c.movable(Side::Out), "{name}, {pass} finish: the end is no longer pinned: {c:?}");
        }
    }
}

/// sol-r3-3: a beat whose voice is an explicit bed under muted pictures plays speech, and kept its
/// narration because the rule only looked at clips. Every profile, a second finish.
#[test]
fn a_beat_voiced_by_a_bed_has_no_narration_over_it() {
    let (db, project_id) = video(&[(0.0, 10.0, "the answer that carries the beat here.", false)]);
    let cfg = ScriptConfig::default();
    let mut s = cut(None, &[&[(100.0, 105.0, "mute"), (110.0, 115.0, "mute")]]);
    s.beats[0].narration = Some("a voice-over nobody should hear on top".into());
    s.beats[0].bed =
        Some(serde_json::from_value(serde_json::json!({ "video_id": 7, "in_s": 0.0, "out_s": 10.0 })).unwrap());
    for (name, profile) in profiles() {
        let (once, twice) = finish_twice(&db, project_id, &s, &cfg, &profile);
        for (pass, f) in [("first", &once), ("second", &twice)] {
            assert!(f.beats[0].bed.is_some(), "{name}, {pass}: the bed is speech and stays: {f:?}");
            assert_eq!(f.beats[0].narration.as_deref().unwrap_or(""), "", "{name}, {pass}: narration over a voice");
        }
    }
}

/// The user's rule, from Hydrogen's cut #228: a voice under b-roll is seen first. Mary Ellen spoke
/// over tote bags and the magazine stand from the beat's first frame; the beat now opens on her,
/// in sync with her voice, and the cutaways give that time back, so the beat keeps its length. A
/// second finish leaves it so; an import (a measured clip) is not rearranged.
#[test]
fn a_voice_under_b_roll_opens_on_the_speaker() {
    let (db, project_id) = video(&[
        (0.0, 4.0, "what I love about this place is the people.", false),
        (4.2, 10.0, "we are tight knit and we support each other always.", false),
    ]);
    db.conn
        .execute(
            "INSERT INTO videos(id, content_hash, size, duration_s, category, category_confidence)
             VALUES (8, 'b', 1, 60.0, 'broll', 0.8)",
            [],
        )
        .unwrap();
    let folder: i64 = db.conn.query_row("SELECT folder_id FROM project_folders LIMIT 1", [], |r| r.get(0)).unwrap();
    db.conn
        .execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (8, ?1, '/m/b.mp4', 1, 0, 0)",
            [folder],
        )
        .unwrap();
    let cfg = ScriptConfig::default();
    let mut s = cut(None, &[&[(0.0, 5.0, "mute"), (5.0, 10.0, "mute")]]);
    for c in &mut s.beats[0].clips {
        c.video_id = 8;
    }
    s.beats[0].bed =
        Some(serde_json::from_value(serde_json::json!({ "video_id": 7, "in_s": 0.0, "out_s": 10.0 })).unwrap());
    for (name, profile) in [("save", Profile::save(None)), ("chat", Profile::chat(false))] {
        let (once, twice) = finish_twice(&db, project_id, &s, &cfg, &profile);
        same_cut(&once, &twice).unwrap_or_else(|e| panic!("{name}: a second finish moved it: {e}"));
        let beat = &once.beats[0];
        let bed = beat.bed.as_ref().expect("the voice stays under the beat");
        let face = &beat.clips[0];
        assert_eq!(face.video_id, 7, "{name}: the beat opens on the speaker: {beat:?}");
        assert!((face.in_s - bed.in_s).abs() < 0.01, "{name}: in sync with the voice: {face:?} / {bed:?}");
        assert!(face.out_s - face.in_s >= pictures::FACE_FIRST_S - 0.01, "{name}: {face:?}");
        assert!(beat.clips[1..].iter().all(|c| c.video_id == 8), "{name}: then the b-roll: {beat:?}");
        // The closing hold is the cut's, after the voice: not part of the beat's length.
        let pictures: f64 =
            beat.clips.iter().filter(|c| c.why.as_deref() != Some(CLOSING_PICTURE)).map(|c| c.out_s - c.in_s).sum();
        assert!((pictures - (bed.out_s - bed.in_s)).abs() < 0.1, "{name}: the beat keeps its length: {beat:?}");
    }
    let (imported, _) = finish_twice(&db, project_id, &s, &cfg, &Profile::import(None));
    assert_eq!(imported.beats[0].clips[0].video_id, 8, "an import keeps its pictures");
}

/// opus-r3-3: one mute rule muted every speechless `source` clip on every save and import, exact
/// ones too, so natural sound an editor kept on purpose (applause, the room) went silent. A clip's
/// own sound goes only under something else heard over its beat — narration, a bed, another clip
/// speaking — and never off an `exact` clip.
#[test]
fn natural_sound_is_muted_only_under_another_voice_and_never_on_an_exact_clip() {
    let (db, project_id) = video(&[(0.0, 10.0, "somebody speaking over here now.", false)]);
    let cfg = ScriptConfig { infer_audio_beds: false, ..ScriptConfig::default() };
    // b0: the room alone. b1: the room under narration. b2: an exact room under narration.
    // b3: the room beside somebody speaking.
    let mut s = cut(
        None,
        &[
            &[(100.0, 106.0, "source")],
            &[(120.0, 126.0, "source")],
            &[(140.0, 146.0, "source")],
            &[(0.0, 10.0, "source"), (160.0, 166.0, "source")],
        ],
    );
    s.beats[1].narration = Some("words read over the view".into());
    s.beats[2].narration = Some("words read over the view".into());
    s.beats[2].clips[0].exact = true;
    for (name, profile) in profiles() {
        let (once, twice) = finish_twice(&db, project_id, &s, &cfg, &profile);
        for (pass, f) in [("first", &once), ("second", &twice)] {
            let audio = |b: usize, c: usize| f.beats[b].clips[c].audio;
            assert_eq!(audio(0, 0), Audio::Source, "{name}, {pass}: the room alone is the beat's sound");
            assert_eq!(audio(1, 0), Audio::Mute, "{name}, {pass}: the room under narration goes");
            assert_eq!(audio(2, 0), Audio::Source, "{name}, {pass}: an exact clip's sound is an instruction");
            assert_eq!(f.beats[2].narration.as_deref(), Some("words read over the view"), "{name}, {pass}");
            assert_eq!(audio(3, 1), Audio::Mute, "{name}, {pass}: the room under somebody speaking goes");
        }
    }
}

/// A small deterministic generator, so the fixture is the same on every run (no new dependency).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[(self.next() as usize) % xs.len()]
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next() % 10_000) as f64 / 10_000.0)
    }
}

/// How a generated clip's edges were written.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pin {
    Free,
    Exact,
    StartOnWord,
    EndOnWord,
}

/// Generated cuts near every draft rule's threshold — speaking clips around `max_clip_s`, shake
/// over about half a clip, exact and word-pinned edges, word- and seconds-written neighbours
/// touching (the merge), explicit beds under pictures with narration — each finished twice under
/// save, chat and import. The second finish leaves the first's cut as it was (within a frame), and
/// the cut keeps the promises: a pinned edge where it was written, no draft clip past
/// `max_clip_s`, no speaking clip that is more than half shake, no narration over speech.
#[test]
fn generated_cuts_settle_and_keep_the_rules_through_every_profile() {
    let cfg = ScriptConfig::default();
    let mut rng = Lcg(0x5eed_6a57);
    let mut checked = 0usize;
    for layout in 0..6 {
        // Sentences of 3–11.5 s over 0–220 s, back to back or a short pause apart.
        let mut segments: Vec<(f64, f64, String, bool)> = Vec::new();
        let mut t = 0.0;
        while t < 220.0 {
            let len = rng.range(3.0, 11.5);
            segments.push((t, t + len, format!("layout {layout} sentence {} says some words.", segments.len()), false));
            t += len + rng.pick(&[0.0, 0.0, 0.3, 0.8, 1.6]);
        }
        let rows: Vec<(f64, f64, &str, bool)> = segments.iter().map(|(a, b, x, o)| (*a, *b, x.as_str(), *o)).collect();
        let (db, project_id) = video(&rows);
        // Two or three shaky stretches.
        let spans: Vec<(f64, f64)> = (0..rng.pick(&[2, 3]))
            .map(|_| {
                let a = rng.range(5.0, 200.0).floor();
                (a, a + rng.range(4.0, 14.0).floor())
            })
            .collect();
        shake(&db, &spans);
        let edges = sentence_edges(&db, 7);
        if std::env::var("SETTLE_DEBUG").is_ok() {
            eprintln!("layout {layout}: edges {edges:?} shake {spans:?}");
        }
        let id_of = |i: usize| sentence_at(&db, edges[i].0);

        for case in 0..40 {
            let mut beats: Vec<serde_json::Value> = Vec::new();
            let mut pins: Vec<(String, Pin, f64, f64)> = Vec::new();
            let mut next_clip = 0usize;
            for b in 0..rng.pick(&[1, 2, 3]) {
                // A spoken stretch starting on a sentence, or inside one.
                let i = (rng.next() as usize) % (edges.len() - 8);
                let start = if rng.pick(&[true, false]) { edges[i].0 } else { rng.range(edges[i].0, edges[i].1) };
                let short = rng.range(4.0, 14.0);
                let len = rng.pick(&[
                    cfg.max_clip_s - 1.5,
                    cfg.max_clip_s - 0.4,
                    cfg.max_clip_s + 0.3,
                    cfg.max_clip_s + 4.0,
                    short,
                    short,
                ]);
                let out = (start + len).min(299.0);
                let kind = rng.pick(&["one", "one", "pair", "bed"]);
                let mut clips: Vec<serde_json::Value> = Vec::new();
                let mut clip = |a: f64, z: f64, audio: &str, pin: Pin, neighbour_ranged: bool| {
                    let why = format!("c{next_clip}");
                    next_clip += 1;
                    let mut c = serde_json::json!({ "video_id": 7, "in_s": a, "out_s": z, "audio": audio, "why": why });
                    // The sentence each edge lies in: a word edge names a word of it.
                    let first = edges.iter().position(|e| e.1 > a + 0.05).unwrap_or(0);
                    let last = edges.iter().rposition(|e| e.0 < z - 0.05).unwrap_or(first).max(first);
                    match pin {
                        Pin::Exact => c["exact"] = true.into(),
                        Pin::StartOnWord => {
                            c["from"] = format!("{}@2", id_of(first)).into();
                            c["to"] = id_of(last).into();
                        }
                        Pin::EndOnWord => {
                            c["from"] = id_of(first).into();
                            c["to"] = format!("{}@2", id_of(last)).into();
                        }
                        Pin::Free if neighbour_ranged => {
                            c["from"] = id_of(first).into();
                            c["to"] = id_of(last).into();
                        }
                        Pin::Free => {}
                    }
                    pins.push((why, pin, a, z));
                    c
                };
                let pin = rng.pick(&[Pin::Free, Pin::Free, Pin::Exact, Pin::StartOnWord, Pin::EndOnWord]);
                let mut narration = "";
                let mut bed = serde_json::Value::Null;
                match kind {
                    "pair" => {
                        // Two touching cuts of one take, one written on words and one in seconds.
                        let mid = edges.iter().map(|e| e.1).find(|&e| e > start + 3.0 && e < out - 3.0).unwrap_or(out);
                        let (p, q) = if rng.pick(&[true, false]) { (pin, Pin::Free) } else { (Pin::Free, pin) };
                        clips.push(clip(start, mid, "source", p, rng.pick(&[true, false])));
                        if mid < out {
                            clips.push(clip(mid, out, "source", q, rng.pick(&[true, false])));
                        }
                    }
                    "bed" => {
                        // An explicit bed under pictures, with narration that may not play over it.
                        let bed_len = rng.range(6.0, 14.0);
                        bed = serde_json::json!({ "video_id": 7, "in_s": start, "out_s": start + bed_len });
                        let pic = 230.0 + 20.0 * b as f64;
                        clips.push(clip(pic, pic + bed_len / 2.0, "mute", Pin::Free, false));
                        clips.push(clip(pic + 10.0, pic + 10.0 + bed_len / 2.0, "mute", Pin::Free, false));
                        narration = "words read over the pictures";
                    }
                    _ => {
                        clips.push(clip(start, out, "source", pin, false));
                        if rng.pick(&[true, false, false]) {
                            narration = "a voice-over that may not play over speech";
                        }
                    }
                }
                let mut beat = serde_json::json!({ "id": format!("b{b}"), "purpose": "p", "narration": narration,
                    "clips": clips });
                if !bed.is_null() {
                    beat["bed"] = bed;
                }
                beats.push(beat);
            }
            let s: Script =
                serde_json::from_value(serde_json::json!({ "title": "t", "beats": beats })).expect("a valid script");

            for (name, profile) in profiles() {
                let (once, twice) = finish_twice(&db, project_id, &s, &cfg, &profile);
                let label = format!("layout {layout} case {case} {name}");
                same_cut(&once, &twice).unwrap_or_else(|e| {
                    panic!("{label}: a second finish moved it: {e}\nwritten: {s:?}\nonce: {once:?}\ntwice: {twice:?}")
                });
                for f in [&once, &twice] {
                    for beat in &f.beats {
                        // Nobody is read over somebody speaking.
                        let speaks = beat
                            .clips
                            .iter()
                            .any(|c| c.audio == Audio::Source && clip_has_speech(&db, c.video_id, c.in_s, c.out_s))
                            || beat.bed.as_ref().is_some_and(|b| clip_has_speech(&db, b.video_id, b.in_s, b.out_s));
                        assert!(
                            !speaks || beat.narration.as_deref().unwrap_or("").trim().is_empty(),
                            "{label}: narration over speech: {beat:?}"
                        );
                        for c in &beat.clips {
                            let Some((_, pin, a, z)) = pins.iter().find(|p| Some(p.0.as_str()) == c.why.as_deref())
                            else {
                                continue;
                            };
                            // A pinned edge is where it was written.
                            if matches!(pin, Pin::Exact | Pin::StartOnWord) {
                                assert_eq!(c.in_s, *a, "{label}: a pinned start moved: {c:?}");
                            }
                            if matches!(pin, Pin::Exact | Pin::EndOnWord) {
                                assert!(
                                    beat.clips.iter().any(|d| d.out_s == *z),
                                    "{label}: a pinned end moved: {c:?} in {beat:?}"
                                );
                            }
                            if !profile.draft || !c.movable(Side::Out) || c.why.as_deref() == Some(CLOSING_PICTURE) {
                                continue;
                            }
                            // No draft clip runs past the cap (its tail settled on the sound aside).
                            assert!(
                                c.out_s - c.in_s <= cfg.max_clip_s + FRAME_S,
                                "{label}: a clip over max_clip_s: {c:?}"
                            );
                            // No speaking clip code may trim is mostly shake.
                            if c.audio == Audio::Source && c.movable(Side::In) {
                                let share = shaky_share(&db, &cfg, c.in_s, c.out_s);
                                assert!(share <= 0.5 + 1e-9, "{label}: {:.0} % shake: {c:?}", share * 100.0);
                            }
                        }
                    }
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 6 * 40 * 3);
}

#[test]
fn silent_tracks_do_not_hush_narration() {
    let (db, _) = video(&[(0.0, 10.0, "the answer carries this beat.", false)]);
    let mut s = cut(None, &[&[(0.0, 8.0, "source")]]);
    s.beats[0].narration = Some("keep this narration".into());
    s.beats[0].clips[0].tracks = Some(vec![]);
    assert_eq!(super::draft::hush_narration_over_speech(&db, &mut s), 0);
    s.beats[0].bed = Some(
        serde_json::from_value(serde_json::json!({
            "video_id":7,"in_s":0.0,"out_s":8.0,"tracks":[]
        }))
        .unwrap(),
    );
    assert_eq!(super::draft::hush_narration_over_speech(&db, &mut s), 0);
}

#[test]
fn pinned_pictures_are_fixed_budget_and_short_pins_survive() {
    let (db, _) = video(&[]);
    let mut s = cut(Some(15.0), &[&[(0.0, 10.0, "mute"), (20.0, 30.0, "mute")]]);
    s.beats[0].clips[0].exact = true;
    super::fit::fit_to_target(&db, &mut s, &ScriptConfig::default());
    assert!((s.total_duration_s() - 15.0).abs() < 0.01);
    assert_eq!(s.beats[0].clips[0].out_s, 10.0);
    s.beats[0].clips[0].out_s = 0.8;
    super::fit::clamp_to_duration(&db, &mut s);
    assert_eq!(s.clip_count(), 2);
    assert_eq!(s.beats[0].clips[0].out_s, 0.8);
}

#[test]
fn inferred_beds_preserve_selected_tracks_and_skip_silent_sources() {
    let (db, _) = video(&[(0.0, 6.0, "the answer continues under the picture.", false)]);
    let cfg = ScriptConfig { infer_audio_beds: true, ..ScriptConfig::default() };
    for tracks in [Some(vec![1]), Some(vec![])] {
        let mut s = cut(None, &[&[(0.0, 3.0, "source"), (100.0, 105.0, "mute")]]);
        s.beats[0].clips[0].tracks = tracks.clone();
        super::pictures::lay_audio_beds(&db, &mut s, &cfg);
        if tracks.as_ref().unwrap().is_empty() {
            assert!(s.beats[0].bed.is_none());
        } else {
            assert_eq!(s.beats[0].bed.as_ref().expect("inferred bed").tracks, tracks);
        }
    }
}
