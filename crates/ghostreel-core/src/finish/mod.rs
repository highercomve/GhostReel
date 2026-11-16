//! Finishing a cut: the deterministic passes every script goes through between the editor's draft
//! and the saved file, whoever wrote it — a chat turn (`chat::repair::finish_script`), an MCP
//! `save_script`/`apply_fix`/`read_cut` (`tools::scripts`), a Clips run or `script import`
//! (`script_check::repair_for_import`).
//!
//! There is one chain, [`run`], with one fixed order. What differs between callers is a
//! [`Profile`] (fit to the target or not, let speech breathe or not, the shortest a clip may be
//! trimmed to, whether a model wrote it), never the passes or their order: the same draft comes out
//! the same whichever way it was delivered.
//!
//! An edge an editor pinned — an `exact` clip, an edge written on a word — is never moved by any
//! pass ([`crate::script::ScriptClip::movable`]). A person's edit in the app does not come through
//! here at all: it is kept as made, and the check lists what this chain would change as fixes.
//!
//! Every pass decides where an edge belongs by the same rules ([`edges`]), so a finished script
//! run through the chain again comes out as it went in.
//!
//! This module sits below `chat` and `tools`: it depends on the script, the index and the cut
//! placement (`cuts`), never on the chat loop.

use rusqlite::params;

use crate::Error;
use crate::config::ScriptConfig;
use crate::db::Db;
use crate::script::Script;

pub mod draft;
pub mod edges;
pub mod fit;
pub mod pictures;
pub mod speech;
pub mod words;

#[cfg(test)]
mod settle_tests;

pub use draft::{merge_contiguous_clips, mute_clips_without_speech};
pub use edges::{Unspoken, name_ranges, sentence_edges, snap_to_sentences, trim_unspoken_edges};
pub use fit::{clamp_to_duration, drop_repeated_footage, fit_to_target, trim_to_target};
pub use pictures::{
    TALKING_SHARE, clamp_beds_to_beats, drop_closing_picture, hold_the_last_picture, lay_audio_beds, on_mic_share,
    trim_pictures_to_bed,
};
pub use speech::{end_on_sentences, end_on_turns, pad_speech};

/// What a caller changes about the chain. Everything else is the same for every cut.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Profile {
    /// Fit the cut to its `target_duration_s`. A chat turn's first draft does; a revision keeps the
    /// length the user's feedback asked for, and a tool save or an import keeps what was written.
    pub fit_target: bool,
    /// Let speaking clips breathe: finish the sentence, hold a moment of the person, lead in before
    /// the first word ([`pad_speech`]). A Clips run measures its clips to a length range and does not.
    pub pad: bool,
    /// The shortest the framing repair may trim a clip to (a Clips run's length range); `None` keeps
    /// most of each clip.
    pub min_clip_s: Option<f64>,
    /// A model editor wrote the cut clip by clip (a chat turn, a tool save): hold it to the rules a
    /// model is told and can miss ([`draft`]: footage of this project, a steady camera, no clip
    /// past `max_clip_s`, back-to-back cuts of one shot joined). A Clips run or an import measured
    /// its clips and keeps them.
    pub draft: bool,
}

impl Profile {
    /// A chat turn's cut. `fit_target`: a first draft is fitted to its length, a revision is not.
    pub fn chat(fit_target: bool) -> Self {
        Self { fit_target, pad: true, min_clip_s: None, draft: true }
    }

    /// A script an editor saved through the tools (`save_script`, `apply_fix`, `read_cut`): the same
    /// finish as a chat turn's, at the length it was written.
    pub fn save(min_clip_s: Option<f64>) -> Self {
        Self { fit_target: false, pad: true, min_clip_s, draft: true }
    }

    /// A script finished outside an editor (`script import`, a Clips run): its clips are the length
    /// they were measured to, so nothing pads them.
    pub fn import(min_clip_s: Option<f64>) -> Self {
        Self { fit_target: false, pad: false, min_clip_s, draft: false }
    }
}

/// Which pass changed something.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// The rules a model editor can miss ([`draft`]).
    Draft,
    Repeats,
    Unspoken,
    Beds,
    Muted,
    Fit,
    Turns,
    Sentences,
    Pictures,
    Hold,
    Framing,
    Cuts,
    /// Sentence ids renamed to what a clip plays once its edges moved.
    Ranges,
}

/// One thing the chain did to the script, for the editor and the user.
#[derive(Debug, Clone, PartialEq)]
pub struct Note {
    pub pass: Pass,
    pub message: String,
}

/// Finish `s` in place: the one chain every script goes through, in a fixed order. Returns a note
/// for each pass that changed something, and none when the script comes out as it went in (the
/// passes may take turns on the way — the transcript's times pull an edge one way, the sound puts
/// it back — and a script left exactly as it came was not changed).
///
/// A script with no clips is left alone.
pub fn run(
    db: &Db,
    project_id: i64,
    s: &mut Script,
    cfg: &ScriptConfig,
    profile: &Profile,
) -> Result<Vec<Note>, Error> {
    let before = s.clone();
    let mut notes = Vec::new();
    let mut note = |pass: Pass, message: String| notes.push(Note { pass, message });
    if s.clip_count() == 0 {
        return Ok(Vec::new());
    }

    // Anything a previous finish added comes off first: the fit counts clips, and a picture this
    // chain put there is not one the editor chose.
    drop_closing_picture(s);

    // What a model was told and may have missed: footage of this project. (A steady camera and
    // the longest a clip may be are held once the clips have their final shape, below.)
    if profile.draft {
        for message in draft::drop_foreign_clips(db, project_id, s) {
            note(Pass::Draft, message);
        }
        if s.clip_count() == 0 {
            return Ok(notes);
        }
    }

    // A montage has no speech to end on, pad or lay under anything: it is fitted and placed.
    if cfg.style.broll {
        if profile.draft {
            for message in draft::cap_clip_length(db, s, cfg, false) {
                note(Pass::Draft, message);
            }
        }
        for message in clamp_to_duration(db, s) {
            note(Pass::Fit, message);
        }
        if profile.fit_target {
            let before = s.total_duration_s();
            if fit_to_target(db, s, cfg) {
                note(Pass::Fit, format!("fitted to target: {before:.1} s → {:.1} s", s.total_duration_s()));
            }
        }
        name_ranges(db, s, cfg);
        return Ok(unless_unchanged(&before, s, notes));
    }

    // Who is heard. A clip nobody speaks in plays no sound of its own under somebody else's voice.
    let muted = mute_clips_without_speech(db, s);
    if muted > 0 {
        note(Pass::Muted, format!("muted {muted} clip(s) with no speech in range under another voice"));
    }
    // A draft's back-to-back cuts of one shot, played the same way, are one clip split while looking.
    if profile.draft {
        let merged = merge_contiguous_clips(s, cfg);
        if merged > 0 {
            note(Pass::Draft, format!("joined {merged} back-to-back cut(s) of the same shot into continuous clips"));
        }
    }

    // Which clips are in the cut. The same line twice is never the edit asked for: the second use goes.
    let repeats = drop_repeated_footage(s);
    if repeats > 0 {
        note(
            Pass::Repeats,
            format!(
                "dropped {repeats} clip(s) that repeated footage already used earlier in the cut: choose other footage \
                 if the cut is now short"
            ),
        );
    }

    // Where each speaking range starts and ends, by the transcript: on whole sentences, without the
    // interviewer or a filler opener at either end.
    snap_to_sentences(db, s, cfg)?;
    let unspoken = trim_unspoken_edges(db, s);
    if unspoken.moved > 0 {
        note(
            Pass::Unspoken,
            format!("took {} edge(s) off lines said off the microphone or a filler opener", unspoken.moved),
        );
    }
    if unspoken.dropped > 0 {
        note(
            Pass::Unspoken,
            format!("dropped {} clip(s) where only someone off the microphone speaks", unspoken.dropped),
        );
    }
    if s.clip_count() == 0 {
        return Ok(notes);
    }
    // A range that lost a sentence off an end no longer names it, so the passes below (cut placement
    // reads the words a range was written on) see what plays.
    name_ranges(db, s, cfg);
    // Now that it is settled who speaks, a beat that would fall silent when it cuts away keeps the
    // voice running under the pictures.
    for issue in lay_audio_beds(db, s, cfg) {
        note(Pass::Beds, issue.message);
    }
    // A beat that plays somebody speaking, on a clip or on the bed just checked or laid, has no
    // narration over them.
    let hushed = draft::hush_narration_over_speech(db, s);
    if hushed > 0 {
        note(Pass::Muted, format!("dropped narration from {hushed} beat(s) that play somebody speaking"));
    }

    if profile.pad {
        pad_speech(db, s, cfg);
    }
    for message in clamp_to_duration(db, s) {
        note(Pass::Fit, message);
    }

    // These finish the cut rather than fit it. A range stops where the speaker stopped, then on a
    // whole sentence (`end_on_turns` moves an edge and `end_on_sentences` puts it on a boundary).
    let turns = end_on_turns(db, s, cfg);
    if turns > 0 {
        note(
            Pass::Turns,
            format!(
                "ended {} where the speaker stopped, before the reply over the top",
                if turns == 1 { "one quote".to_string() } else { format!("{turns} quotes") },
            ),
        );
    }
    let mended = end_on_sentences(db, s, cfg);
    if mended > 0 {
        note(
            Pass::Sentences,
            if mended == 1 {
                "restored one trimmed quote to its full sentence".to_string()
            } else {
                format!("restored {mended} trimmed quotes to their full sentences")
            },
        );
    }
    // The one length limit, after every pass that grows a clip and before anything that only
    // shortens one: it ends a clip where those passes would have, so they find nothing to grow
    // on the next finish (`draft::cap_clip_length`).
    if profile.draft {
        for message in draft::cap_clip_length(db, s, cfg, profile.pad) {
            note(Pass::Draft, message);
        }
    }
    let shortened = trim_pictures_to_bed(s, cfg);
    // Each beat with its breath before the fit, so the fit measures the cut as it will play.
    let (_, breathed) = pictures::fit_pictures_to_bed(db, s, &[], true, cfg);
    // A picture that went for the voice can leave the voice outlasting the pictures. Put it back
    // inside its beat now, before the closing hold: held first, the hold covered the overhang,
    // and the next finish — which takes the hold back first — cut the voice to the pictures then.
    clamp_beds_to_beats(db, s, cfg);
    if shortened > 0 {
        note(
            Pass::Pictures,
            format!(
                "cut the pictures back to the voice in {}",
                if shortened == 1 { "one beat".to_string() } else { format!("{shortened} beats") },
            ),
        );
    }
    // A voice under b-roll is seen first: the beat opens on the speaker, then cuts away.
    if profile.draft {
        let opened = pictures::open_on_the_speaker(db, s);
        if opened > 0 {
            clamp_beds_to_beats(db, s, cfg);
            note(
                Pass::Pictures,
                format!(
                    "opened {} on the speaker before the b-roll",
                    if opened == 1 { "one beat".to_string() } else { format!("{opened} beats") },
                ),
            );
        }
    }
    // A steady camera, judged on each clip as it will now play (`draft::steady_camera`).
    if profile.draft {
        let steadied = draft::steady_camera(db, s, cfg);
        if !steadied.is_empty() {
            clamp_beds_to_beats(db, s, cfg);
        }
        for message in steadied {
            note(Pass::Draft, message);
        }
    }
    // Padding and sentence ends can grow two clips of the same video into each other, and a
    // picture moved off a shaky stretch can land on footage the cut already plays. Checked once
    // clips have their final place, as the next finish's first check will see them.
    let overlapped = drop_repeated_footage(s);
    if overlapped > 0 {
        clamp_beds_to_beats(db, s, cfg);
        note(Pass::Repeats, format!("dropped {overlapped} clip(s) that overlapped another once finished"));
    }
    if s.clip_count() == 0 {
        return Ok(notes);
    }
    // The length, measured on clips at the length they will play.
    // After the passes that settle each beat's length and before the closing hold, which it
    // leaves room for: the fit sees the cut as it will play, so the next finish sees the same.
    if profile.fit_target {
        let before = s.total_duration_s();
        let hold = pictures::closing_hold_s(db, project_id, s, cfg);
        if fit::fit_to_target_leaving(db, s, cfg, hold) {
            note(Pass::Fit, format!("fitted to target: {before:.1} s → {:.1} s", s.total_duration_s()));
        }
    }

    let held = hold_the_last_picture(db, project_id, s, cfg);
    if held > 0.0 {
        note(Pass::Hold, format!("held the closing picture for {held:.1} s of quiet"));
    }
    // A vertical cut neither opens nor closes on a screen share when a sentence edge allows.
    for message in crate::script_fix::repair_framing(db, s, cfg.speech_overrun_s, profile.min_clip_s) {
        note(Pass::Framing, message);
    }
    // A bed may not outlast the pictures it plays under, and everything above moves them.
    clamp_beds_to_beats(db, s, cfg);
    // Last, on the sound itself: everything above cut by the transcript's times, which are approximate.
    let mut beds_placed: Vec<usize> = Vec::new();
    for m in crate::cuts::place_cuts(db, s, cfg) {
        if m.clip_index.is_none() {
            beds_placed.push(m.beat_index);
        }
        note(Pass::Cuts, m.message());
    }
    // A bed placement shortened (an "umm" after its last word left out) no longer covers its
    // pictures: the beat's last picture gives that time back, so the next beat does not open on
    // silence.
    let (given, _) = pictures::fit_pictures_to_bed(db, s, &beds_placed, false, cfg);

    if breathed > 0 {
        note(
            Pass::Pictures,
            format!(
                "let the b-roll run on {:.1} s, muted, after the voice in {}",
                cfg.broll_breath_s,
                if breathed == 1 { "one beat".to_string() } else { format!("{breathed} beats") },
            ),
        );
    }
    if given > 0 {
        note(
            Pass::Pictures,
            format!(
                "cut the pictures back to the voice in {}",
                if given == 1 { "one beat".to_string() } else { format!("{given} beats") },
            ),
        );
    }
    // The sentence ids each clip was written with, for what it plays now.
    let renamed = name_ranges(db, s, cfg);
    if renamed > 0 {
        note(Pass::Ranges, format!("updated the sentence ids of {renamed} range(s) to the sentences they now play"));
    }
    // A finished cut has no beat without pictures, whichever pass emptied it.
    s.beats.retain(|b| !b.clips.is_empty());

    Ok(unless_unchanged(&before, s, notes))
}

/// `notes`, or none when the script is exactly what it was.
fn unless_unchanged(before: &Script, after: &Script, notes: Vec<Note>) -> Vec<Note> {
    if after == before { Vec::new() } else { notes }
}

/// A video's length, when known.
pub(crate) fn video_duration(db: &Db, video_id: i64) -> Option<f64> {
    db.conn
        .query_row("SELECT duration_s FROM videos WHERE id = ?1", [video_id], |r| r.get::<_, Option<f64>>(0))
        .ok()
        .flatten()
}

/// The on-mic words a clip actually carries, as one line.
pub(crate) fn clip_speech(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> String {
    let Ok(mut st) = db.conn.prepare(
        "SELECT text FROM transcript_segments
          WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3 AND COALESCE(off_mic, 0) = 0
          ORDER BY start_s",
    ) else {
        return String::new();
    };
    st.query_map(params![video_id, in_s, out_s], |r| r.get::<_, String>(0))
        .map(|rows| rows.flatten().collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

/// Words below which a clip's speech is an acknowledgement rather than something said.
///
/// The repair passes check length, sentence boundaries and grounding; none of them ever asked
/// whether the words were worth hearing. A cut came back at 39.9 s against a 40 s target — a
/// perfect duration score — with three of its eight clips being five seconds each of "Thank you."
/// and "Okay.", a third of the piece. `chat/build.rs` has held this rule since it was written;
/// a model writing the script was never told it.
pub(crate) const MIN_CLIP_WORDS: usize = 4;

/// Whether anyone is transcribed speaking in `in_s..out_s` of a video.
pub(crate) fn clip_has_speech(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> bool {
    // B-roll's talk is the crowd (`classify::plays_ambience`): its pictures are what a cut takes, so
    // it plays under a voice and gives way to the fit like any picture.
    if crate::classify::video_plays_ambience(db, video_id) {
        return false;
    }
    db.conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM transcript_segments WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3)",
            params![video_id, in_s, out_s],
            |r| r.get::<_, bool>(0),
        )
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    /// One video, its transcript as `(start, end, text, off_mic)`, one sentence per segment.
    pub(super) fn video(segments: &[(f64, f64, &str, bool)]) -> (Db, i64) {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("P")).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (7, 'h', 1, 300.0)", [])
            .unwrap();
        // In the project, as every clip a draft may keep is.
        let folder = db.add_folder(project.id, &std::env::temp_dir(), true).unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (7, ?1, '/m/v.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        for (a, b, text, off) in segments {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES (7, ?1, ?2, ?3, ?4)",
                    params![a, b, text, *off as i64],
                )
                .unwrap();
        }
        crate::sentence_split::mirror_segments(&db, 7);
        (db, project.id)
    }

    /// A script of beats, each a list of `(in, out, audio)` clips of video 7.
    pub(super) fn cut(target: Option<f64>, beats: &[&[(f64, f64, &str)]]) -> Script {
        let beats: Vec<serde_json::Value> = beats
            .iter()
            .enumerate()
            .map(|(i, clips)| {
                serde_json::json!({ "id": format!("b{i}"), "purpose": "p", "narration": "",
                    "clips": clips.iter().map(|(a, b, audio)| serde_json::json!({
                        "video_id": 7, "in_s": a, "out_s": b, "audio": audio })).collect::<Vec<_>>() })
            })
            .collect();
        serde_json::from_value(serde_json::json!({ "title": "t", "target_duration_s": target, "beats": beats }))
            .unwrap()
    }

    /// The `qwen35` replay draft's beat 4: a sentence, then two short "Yeah."s touching it. A clip
    /// that has finished the first and carries its tail into the next is that sentence and its
    /// tail. Snapping it to the nearest end (the "Yeah." the tail runs into) finished a sentence
    /// nobody chose, and the padding then added a tail to that one — once per repair.
    #[test]
    fn an_end_in_its_sentences_tail_belongs_to_that_sentence_and_stays() {
        let (db, _) = video(&[
            (70.0, 79.88, "but tell me about it.", false),
            (79.88, 80.34, "Yeah.", false),
            (80.34, 83.9, "Something else entirely now.", false),
        ]);
        let cfg = ScriptConfig::default();
        let edges = sentence_edges(&db, 7);
        let tail_out = 79.88 + cfg.speech_overrun_s;
        assert_eq!(
            edges::out_owner(&edges, tail_out, cfg.speech_overrun_s),
            Some(0),
            "the tail is the first sentence's"
        );
        assert_eq!(
            edges::out_owner(&edges, 80.34, cfg.speech_overrun_s),
            Some(1),
            "an end on a boundary is that sentence's"
        );
        assert_eq!(edges::out_owner(&edges, 82.0, cfg.speech_overrun_s), None, "inside a sentence");

        let mut s = cut(None, &[&[(70.0, tail_out, "source")]]);
        snap_to_sentences(&db, &mut s, &cfg).unwrap();
        assert!((s.beats[0].clips[0].out_s - tail_out).abs() < 1e-9, "{:?}", s.beats[0].clips[0]);
    }

    /// The same for a start: a range that opens in the lead before a sentence opens on it. Snapping
    /// the lead away and putting it back was an import that moved on every repair.
    #[test]
    fn a_start_in_the_lead_before_a_sentence_belongs_to_that_sentence() {
        let (db, _) =
            video(&[(10.0, 20.0, "the first whole sentence.", false), (21.0, 30.0, "the second one here.", false)]);
        let cfg = ScriptConfig::default();
        let edges = sentence_edges(&db, 7);
        assert_eq!(edges::in_owner(&edges, 21.0 - cfg.speech_lead_s, cfg.speech_lead_s), Some(1));
        let mut s = cut(None, &[&[(20.5, 30.0, "source")]]);
        snap_to_sentences(&db, &mut s, &cfg).unwrap();
        assert!((s.beats[0].clips[0].in_s - 20.5).abs() < 1e-9, "{:?}", s.beats[0].clips[0]);
    }

    /// 0.4 s over the ceiling is not a quote's worth: the breath each quote holds after its last
    /// sentence goes first, and every beat stays.
    #[test]
    fn a_cut_just_over_its_ceiling_gives_up_breaths_not_a_quote() {
        // Four answers, each a sentence with a pause after it.
        let (db, _) = video(&[
            (0.0, 10.0, "the first answer is here.", false),
            (20.0, 30.0, "the second answer is here.", false),
            (40.0, 50.0, "the third answer is here.", false),
            (60.0, 70.0, "the fourth answer is here.", false),
        ]);
        let cfg = ScriptConfig::default();
        // Each quote padded 1.5 s past its sentence; 4 × 11.5 + 4.4 = 50.4 s against a 40 s target
        // (ceiling 50 s).
        let mut s = cut(
            Some(40.0),
            &[
                &[(0.0, 11.5, "source")],
                &[(20.0, 31.5, "source")],
                &[(40.0, 51.5, "source")],
                &[(60.0, 75.9, "source")],
            ],
        );
        assert!((s.total_duration_s() - 50.4).abs() < 1e-6);
        assert!(fit_to_target(&db, &mut s, &cfg));
        assert_eq!(s.beats.len(), 4, "a whole quote went for 0.4 s: {s:?}");
        for c in s.beats.iter().flat_map(|b| &b.clips) {
            let sentence_end = (c.in_s / 20.0).round() * 20.0 + 10.0;
            assert!(c.out_s >= sentence_end + cfg.speech_overrun_s - 1e-6, "a quote lost words: {c:?}");
        }
        // And the fit is one decision: a second one finds nothing to do.
        let once = s.clone();
        fit_to_target(&db, &mut s, &cfg);
        assert_eq!(s, once);
    }

    /// A clip that opens on the interviewer is trimmed to the answer, by the span rule, not slid
    /// forward keeping its length — which played the next question too — and a clip that is all
    /// interviewer goes.
    #[test]
    fn a_clip_opening_off_the_microphone_is_trimmed_and_one_all_off_it_is_dropped() {
        let (db, project_id) = video(&[
            (1.0, 6.0, "Every event we have come to, we walked away with a friend.", false),
            (6.5, 8.5, "What made you want to sponsor the bar?", true),
            (9.0, 14.0, "We love this hotel and the people who come here.", false),
            (14.5, 17.0, "And what about the music?", true),
        ]);
        let cfg = ScriptConfig::default();
        let mut s = cut(None, &[&[(6.5, 14.0, "source")], &[(14.5, 17.0, "source")]]);
        let notes = run(&db, project_id, &mut s, &cfg, &Profile::import(None)).unwrap();
        let c = &s.beats[0].clips[0];
        assert!((8.5..=9.0).contains(&c.in_s) && c.out_s < 14.5, "trimmed to the answer: {c:?}");
        assert_eq!(s.beats.len(), 1, "the question alone is not a clip: {s:?}");
        assert!(notes.iter().any(|n| n.pass == Pass::Unspoken), "{notes:?}");
    }

    /// The chain decides edges with the caller's settings, the snap included: with a longer lead
    /// set, a start 0.7 s before a sentence is that sentence's lead, and snapping it onto the first
    /// word by the default lead took the lead away again.
    #[test]
    fn the_snap_uses_the_callers_settings() {
        let (db, project_id) =
            video(&[(0.0, 5.0, "the first whole sentence.", false), (10.0, 20.0, "the second one is here.", false)]);
        let cfg = ScriptConfig { speech_lead_s: 0.9, ..ScriptConfig::default() };
        let mut s = cut(None, &[&[(9.3, 20.0 + cfg.speech_overrun_s, "source")]]);
        for pass in ["first", "second"] {
            run(&db, project_id, &mut s, &cfg, &Profile::import(None)).unwrap();
            let c = &s.beats[0].clips[0];
            assert!((c.in_s - 9.3).abs() < 1e-9, "{pass} finish: the configured lead was taken off: {c:?}");
        }
    }

    /// A clip written as the interviewer's question and the answer, trimmed to the answer, names the
    /// answer: a range still naming the trimmed question described words that no longer play.
    #[test]
    fn a_range_names_what_plays_once_an_edge_moves() {
        let (db, project_id) = video(&[
            (1.0, 6.0, "Every event we have come to, we walked away with a friend.", false),
            (6.5, 8.5, "What made you want to sponsor the bar?", true),
            (9.0, 14.0, "We love this hotel and the people who come here.", false),
        ]);
        let (question, answer) = (sentence_at(&db, 6.5), sentence_at(&db, 9.0));
        let cfg = ScriptConfig::default();
        let mut s: Script = serde_json::from_value(serde_json::json!({ "title": "t", "beats": [{ "id": "b0",
            "purpose": "p", "narration": "", "clips": [{ "video_id": 7, "in_s": 6.5, "out_s": 14.35,
            "from": question, "to": answer, "audio": "source" }] }] }))
        .unwrap();
        for profile in [Profile::import(None), Profile::save(None)] {
            let mut s = s.clone();
            run(&db, project_id, &mut s, &cfg, &profile).unwrap();
            let c = &s.beats[0].clips[0];
            assert!(c.in_s >= 8.5, "trimmed to the answer: {c:?}");
            let r = c.sentences.as_ref().expect("still named");
            assert_eq!((r.from.as_str(), r.to.as_str()), (answer.as_str(), answer.as_str()), "{c:?}");
        }
        // A start code moved into the middle of a sentence is named by none: the range goes.
        s.beats[0].clips[0].in_s = 11.0;
        assert_eq!(name_ranges(&db, &mut s, &cfg), 1);
        assert_eq!(s.beats[0].clips[0].sentences, None);
    }

    /// A clip on a shaky stretch that carries someone's voice keeps the whole sentences the camera
    /// holds steady through: sliding it to a steady stretch elsewhere, keeping its length, played
    /// other words, and trimming it to steady seconds left it mid-sentence for the sentence passes to
    /// grow back over the shake. A picture nobody speaks in moves to the nearest steady stretch; a
    /// clip whose edges were set by hand is kept. (Through the whole chain:
    /// `settle_tests::a_steady_trim_keeps_whole_sentences_and_survives_the_chain`.)
    #[test]
    fn a_shaky_speaking_clip_keeps_its_steady_sentences_not_slid_onto_other_words() {
        let (db, _) = video(&[
            (0.0, 6.0, "one two three four five.", false),
            (6.0, 12.0, "six seven eight nine ten.", false),
            (12.0, 18.0, "eleven twelve thirteen fourteen.", false),
            (18.0, 24.0, "fifteen sixteen seventeen eighteen.", false),
            (24.0, 30.0, "nineteen twenty twenty one.", false),
        ]);
        for i in 0..30 {
            let jerk = if (9..20).contains(&i) { 50.0 } else { 0.1 };
            db.conn
                .execute(
                    "INSERT INTO motion_windows(video_id, start_s, end_s, jerk, motion, sway) VALUES (7, ?1, ?2, ?3, 0.0, 0.0)",
                    params![i as f64, i as f64 + 1.0, jerk],
                )
                .unwrap();
        }
        let cfg = ScriptConfig::default();
        let mut s = cut(None, &[&[(1.0, 19.0, "source")], &[(12.0, 18.0, "mute")], &[(1.0, 19.0, "source")]]);
        s.beats[2].clips[0].exact = true;
        let notes = draft::steady_camera(&db, &mut s, &cfg);
        let speech = &s.beats[0].clips[0];
        assert_eq!(
            (speech.in_s, speech.out_s),
            (0.0, 6.0 + cfg.speech_overrun_s),
            "the first sentence, whole and steady: {notes:?}"
        );
        let picture = &s.beats[1].clips[0];
        assert!((picture.out_s - picture.in_s - 6.0).abs() < 1e-9, "a picture keeps its length: {picture:?}");
        assert!(picture.out_s <= 9.0 || picture.in_s >= 20.0, "and moves off the shake: {picture:?}");
        assert_eq!((s.beats[2].clips[0].in_s, s.beats[2].clips[0].out_s), (1.0, 19.0), "kept as written");
        assert_eq!(notes.len(), 3, "{notes:?}");
    }

    /// The id of the sentence of video 7 that starts at `start_s`.
    pub(super) fn sentence_at(db: &Db, start_s: f64) -> String {
        let id: i64 = db
            .conn
            .query_row(
                "SELECT id FROM transcript_sentences WHERE video_id = 7 AND ABS(start_s - ?1) < 1e-6",
                [start_s],
                |r| r.get(0),
            )
            .unwrap();
        crate::sentences::format_id(id)
    }

    /// An editor's instruction is not a draft: an `exact` clip keeps both edges and an edge written
    /// on a word keeps that edge, through every profile and a second finish. Padding ignored both,
    /// and every save ran it: a 3–7 s clip came out 0–11.5 s.
    #[test]
    fn exact_clips_and_word_edges_survive_every_profile() {
        let (db, project_id) = video(&[
            (0.0, 10.0, "the first whole sentence is here.", false),
            (12.0, 22.0, "and the second one.", false),
        ]);
        let s1 = sentence_at(&db, 0.0);
        let cfg = ScriptConfig::default();
        // (label, exact, from, to, in kept, out kept)
        type Case<'a> = (&'a str, bool, Option<String>, Option<String>, bool, bool);
        let cases: [Case; 4] = [
            ("exact", true, None, None, true, true),
            ("both on words", false, Some(format!("{s1}@3")), Some(format!("{s1}@6")), true, true),
            ("start on a word", false, Some(format!("{s1}@3")), Some(s1.clone()), true, false),
            ("end on a word", false, Some(s1.clone()), Some(format!("{s1}@6")), false, true),
        ];
        for (label, exact, from, to, keep_in, keep_out) in cases {
            for (pname, profile) in [
                ("save", Profile::save(None)),
                ("import", Profile::import(None)),
                ("chat", Profile::chat(true)),
                ("revision", Profile::chat(false)),
            ] {
                let mut clip = serde_json::json!({ "video_id": 7, "in_s": 3.0, "out_s": 7.0, "audio": "source" });
                if exact {
                    clip["exact"] = true.into();
                }
                if let (Some(f), Some(t)) = (&from, &to) {
                    clip["from"] = f.clone().into();
                    clip["to"] = t.clone().into();
                }
                let mut s: Script = serde_json::from_value(serde_json::json!({ "title": "t", "target_duration_s": 8.0,
                    "beats": [{ "id": "b0", "purpose": "p", "narration": "", "clips": [clip] }] }))
                .unwrap();
                for pass in ["first", "second"] {
                    run(&db, project_id, &mut s, &cfg, &profile).unwrap();
                    let c = &s.beats[0].clips[0];
                    if keep_in {
                        assert_eq!(c.in_s, 3.0, "{label}, {pname}, {pass} finish: the start moved: {c:?}");
                    }
                    if keep_out {
                        assert_eq!(c.out_s, 7.0, "{label}, {pname}, {pass} finish: the end moved: {c:?}");
                    }
                    assert_eq!(c.exact, exact, "{label}: {c:?}");
                    if from.is_some() {
                        let r = c.sentences.as_ref().unwrap_or_else(|| panic!("{label}, {pname}: range lost: {c:?}"));
                        if keep_in {
                            assert_eq!(Some(&r.from), from.as_ref(), "{label}, {pname}");
                        }
                        if keep_out {
                            assert_eq!(Some(&r.to), to.as_ref(), "{label}, {pname}");
                        }
                    }
                }
            }
        }
    }

    /// The closing hold continues the last shot; when the cut has already shown that footage it
    /// stops where that shot starts instead of playing it again.
    #[test]
    fn the_closing_hold_never_repeats_footage_the_cut_already_showed() {
        let (db, project_id) = video(&[
            (0.0, 5.0, "welcome to the unboxing today.", false),
            (5.0, 12.0, "here is the board itself.", false),
        ]);
        let cfg = ScriptConfig::default();
        let mut s = cut(None, &[&[(5.0, 12.0, "source")], &[(0.0, 5.0, "source")]]);
        let held = pictures::hold_the_last_picture(&db, project_id, &mut s, &cfg);
        assert_eq!(held, 0.0, "the footage after the last clip is the first beat's: {s:?}");
        assert_eq!(s.clip_count(), 2);
    }

    /// The same shot holds only while its speaker stays quiet: one who talks again inside the hold
    /// would be seen talking with the sound gone ("it goes to mute before it ends").
    #[test]
    fn the_closing_hold_stops_before_the_speaker_talks_again() {
        let (db, project_id) = video(&[
            (0.0, 5.0, "welcome to the unboxing today.", false),
            (5.0, 12.0, "here is the board itself.", false),
            (13.5, 20.0, "and this is the case.", false),
        ]);
        db.set_speech_regions(
            7,
            &[
                crate::speech::SpeechRegion::new(0.1, 4.9),
                crate::speech::SpeechRegion::new(5.1, 11.8),
                crate::speech::SpeechRegion::new(13.5, 19.8),
            ],
        )
        .unwrap();
        let cfg = ScriptConfig { closing_hold_s: 2.0, ..ScriptConfig::default() };
        // Quiet for 1.7 s after the last word: the same shot is held, and placement stops it a
        // moment before the next voice.
        let mut s = cut(None, &[&[(0.0, 12.0, "source")]]);
        assert!(pictures::hold_the_last_picture(&db, project_id, &mut s, &cfg) > 0.0);
        crate::cuts::place_cuts(&db, &mut s, &cfg);
        let hold = s.beats[0].clips.last().unwrap();
        assert_eq!(hold.why.as_deref(), Some(crate::script::CLOSING_PICTURE));
        assert!(hold.out_s <= 13.35 + 1e-6, "the hold runs into the next voice: {s:?}");
        let once = s.clone();
        pictures::drop_closing_picture(&mut s);
        pictures::hold_the_last_picture(&db, project_id, &mut s, &cfg);
        crate::cuts::place_cuts(&db, &mut s, &cfg);
        assert_eq!(s, once, "the same cut gets the same hold");
    }
    /// A voice under b-roll that is not the last beat ends with its last b-roll shot running on,
    /// muted, for the breath; a beat that ends on the speaker's face, and the closing beat, do not.
    #[test]
    fn b_roll_breathes_after_the_voice_before_the_next_beat() {
        let (db, _) = video(&[(0.0, 5.0, "welcome to the unboxing today.", false)]);
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (8, 'broll', 1, 60.0)", [])
            .unwrap();
        let cfg = ScriptConfig { broll_breath_s: 2.0, ..ScriptConfig::default() };
        let mute = |v: i64, a: f64, b: f64| crate::script::ScriptClip {
            exact: false,
            gain_db: None,
            sentences: None,
            framing: None,
            video_id: v,
            in_s: a,
            out_s: b,
            audio: crate::script::Audio::Mute,
            why: None,
            tracks: None,
        };
        let bed = crate::script::AudioBed {
            video_id: 7,
            in_s: 0.0,
            out_s: 5.0,
            why: None,
            inferred: false,
            tracks: None,
            gain_db: None,
            sentences: None,
        };
        let mut s = cut(None, &[&[(0.0, 2.5, "mute")], &[(0.0, 5.0, "source")]]);
        s.beats[0].clips.push(mute(8, 10.0, 12.5));
        s.beats[0].bed = Some(bed.clone());
        assert_eq!(pictures::breath_after(&s, 0, &cfg), 2.0);
        assert_eq!(pictures::breath_after(&s, 1, &cfg), 0.0, "the closing beat has the closing hold");
        pictures::fit_pictures_to_bed(&db, &mut s, &[], true, &cfg);
        assert!((s.beats[0].clips[1].out_s - 14.5).abs() < 1e-9, "the b-roll runs on 2 s: {s:?}");
        let once = s.clone();
        pictures::fit_pictures_to_bed(&db, &mut s, &[], true, &cfg);
        assert_eq!(s, once, "a second pass changes nothing");

        // Ending on the speaker's own face: no breath, nobody seen talking with the sound gone.
        let mut face = cut(None, &[&[(0.0, 5.0, "mute")], &[(0.0, 5.0, "source")]]);
        face.beats[0].bed = Some(bed);
        assert_eq!(pictures::breath_after(&face, 0, &cfg), 0.0);
        let off = ScriptConfig { broll_breath_s: 0.0, ..cfg.clone() };
        assert_eq!(pictures::breath_after(&s, 0, &off), 0.0, "0 turns it off");
    }
}
