//! Pictures and the voice under them: audio beds laid and kept inside their beat, pictures cut back
//! to the voice, and the quiet picture a cut closes on.

use rusqlite::params;
use serde_json::Value;

use crate::db::Db;
use crate::script::{CLOSING_PICTURE, Issue, IssueSeverity, Script, ScriptClip, Side};
use crate::sentence_split::cuts_into_a_sentence;

use super::edges::sentence_edges;
use super::{clip_has_speech, clip_speech, video_duration};

/// Cut the pictures back to the voice carrying them, so a cutaway is not a pause.
///
/// A beat that ends its bed before its last picture leaves the cutaway hanging in silence — the
/// b-roll stops being something you see *while* someone talks and becomes a gap between
/// interviews, which is exactly what it should not be. Two of four beats in an agy cut ran on for
/// 2.6 s and 4.4 s after the speaker had finished.
///
/// The trailing pictures give way, last first, and a clip trimmed below `min_trimmed_clip_s`
/// goes rather than becoming a flash. The clip that opens the beat is never dropped: it is the
/// face, and the beat exists to show it.
pub fn trim_pictures_to_bed(script: &mut Script, cfg: &crate::config::ScriptConfig) -> usize {
    let mut trimmed = 0usize;
    let targets: Vec<Option<f64>> = (0..script.beats.len()).map(|b| bed_picture_len(script, b, cfg)).collect();
    for (beat, target) in script.beats.iter_mut().zip(targets) {
        // Narration covers the whole beat, so pictures may outlast a bed when there is some.
        if beat.narration.as_deref().map(str::trim).is_some_and(|n| !n.is_empty()) {
            continue;
        }
        // The voice, and the breath after it.
        let Some(bed_len) = target else { continue };

        let beat_len = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum::<f64>();
        let mut excess = beat_len - bed_len;
        if excess <= 0.1 {
            continue;
        }
        // This trims a tail, not a piece. When the voice covers less than half the beat the
        // model has built something else — pictures that needed narration, or a second speaker —
        // and cutting to the bed would throw most of the cut away: a Qwen draft collapsed from
        // 53.6 s to 8.0 s. Leave it, and let the silent picture show in the score.
        if excess > beat_len * 0.5 {
            continue;
        }
        let mut touched = false;

        // Take it from wherever there is room, in proportion: holding the face a shade less is
        // better than losing the cutaway, which is the picture the beat cut away to show.
        // An end code may not move gives nothing.
        let room = |c: &crate::script::ScriptClip| {
            if c.movable(Side::Out) { ((c.out_s - c.in_s) - cfg.min_trimmed_clip_s).max(0.0) } else { 0.0 }
        };
        let total_room: f64 = beat.clips.iter().map(room).sum();
        if total_room >= excess {
            let share = excess / total_room;
            for c in beat.clips.iter_mut() {
                let give = room(c) * share;
                if give > 0.01 {
                    c.out_s -= give;
                    touched = true;
                }
            }
        } else {
            // Not enough give: the trailing pictures go, last first, rather than becoming
            // flashes. The clip that opens the beat stays — it is the face the beat is for.
            while excess > 0.1 && beat.clips.len() > 1 && beat.clips.last().is_some_and(|c| c.movable(Side::Out)) {
                let last = beat.clips.len() - 1;
                let len = (beat.clips[last].out_s - beat.clips[last].in_s).max(0.0);
                excess -= len;
                beat.clips.remove(last);
                touched = true;
            }
            if excess > 0.1
                && let Some(first) = beat.clips.first_mut().filter(|c| c.movable(Side::Out))
            {
                let give = ((first.out_s - first.in_s) - cfg.min_trimmed_clip_s).max(0.0).min(excess);
                if give > 0.01 {
                    first.out_s -= give;
                    touched = true;
                }
            }
        }

        if touched {
            trimmed += 1;
        }
    }
    trimmed
}

/// How long the last b-roll shot of beat `b` runs on, muted, after its bed's voice stops (s):
/// [`crate::config::ScriptConfig::broll_breath_s`], so the next beat's voice does not land on the
/// heels of this one. Only a beat whose voice is a bed under b-roll and whose last shot is b-roll:
/// a face held in silence is someone seen talking with the sound gone. Never the closing beat,
/// which has the closing hold, nor a narrated one, whose narration covers the beat.
pub fn breath_after(script: &Script, b: usize, cfg: &crate::config::ScriptConfig) -> f64 {
    let Some(beat) = script.beats.get(b) else { return 0.0 };
    if cfg.broll_breath_s <= 0.0 || b + 1 >= script.beats.len() {
        return 0.0;
    }
    if beat.narration.as_deref().map(str::trim).is_some_and(|n| !n.is_empty()) {
        return 0.0;
    }
    let (Some(bed), Some(last)) = (&beat.bed, beat.clips.last()) else { return 0.0 };
    if last.video_id == bed.video_id || bed.duration_s() <= 0.0 {
        return 0.0;
    }
    cfg.broll_breath_s
}

/// How long beat `b`'s pictures run when a bed carries it: the voice, then the breath
/// ([`breath_after`]). `None` without a bed.
pub fn bed_picture_len(script: &Script, b: usize, cfg: &crate::config::ScriptConfig) -> Option<f64> {
    let bed_len = script.beats.get(b)?.bed.as_ref()?.duration_s();
    (bed_len > 0.0).then(|| bed_len + breath_after(script, b, cfg))
}

/// Bring every bedded beat's pictures to [`bed_picture_len`] from the end of its last picture.
///
/// Placement runs after every pass that fits pictures to a voice, and it can shorten a bed (it
/// leaves out an "umm" after the last word): for the beds in `placed`, the last picture's tail
/// gives that time back, down to `min_trimmed_clip_s` (a muted cutaway a second shorter is the
/// same shot, and nothing is dropped; the time is code's, so it is taken even from a clip written
/// to exact seconds). Other beats were cut back by [`trim_pictures_to_bed`], whose rules stand. A beat short of its breath has its last b-roll shot run on for it,
/// as far as its footage goes, when `grow` (before the fit to the target, which may then take
/// some of it back: the target wins). Returns (beats cut back, beats given their breath).
pub fn fit_pictures_to_bed(
    db: &Db,
    script: &mut Script,
    placed: &[usize],
    grow: bool,
    cfg: &crate::config::ScriptConfig,
) -> (usize, usize) {
    let (mut given, mut breathed) = (0, 0);
    for b in 0..script.beats.len() {
        if script.beats[b].narration.as_deref().map(str::trim).is_some_and(|n| !n.is_empty()) {
            continue;
        }
        let Some(target) = bed_picture_len(script, b, cfg) else { continue };
        let breath = breath_after(script, b, cfg);
        let beat = &mut script.beats[b];
        let beat_len: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();
        let Some(last) = beat.clips.last_mut() else { continue };
        let excess = beat_len - target;
        if excess > 0.1 && placed.contains(&b) {
            let give = ((last.out_s - last.in_s) - cfg.min_trimmed_clip_s).max(0.0).min(excess);
            if give > 0.01 {
                last.out_s -= give;
                given += 1;
            }
        } else if grow && breath > 0.0 && excess < -0.1 && last.movable(Side::Out) {
            let room = (video_duration(db, last.video_id).unwrap_or(last.out_s) - last.out_s).max(0.0);
            let grow = (-excess).min(room);
            if grow > 0.05 {
                last.out_s += grow;
                breathed += 1;
            }
        }
    }
    (given, breathed)
}

/// How long a beat whose voice runs under b-roll shows the speaker first (s): long enough to see
/// who is talking before the cutaway.
pub const FACE_FIRST_S: f64 = 2.5;
/// The shortest a cutaway is trimmed to to make room for the face (s); below it the cutaway goes.
const CUTAWAY_FLOOR_S: f64 = 1.5;

/// Open each beat whose voice is a bed under b-roll on the speaker's own face.
///
/// A voice heard over pictures of something else, with the person never seen, is a stranger
/// talking: the viewer needs the face first to know who it is, and the b-roll after it is what they
/// talk about (the user's rule, from a cut where three of seven beats played a voice over b-roll
/// from the first frame). The beat opens on the bed's own footage, in sync with the voice, for
/// [`FACE_FIRST_S`]; the cutaways give that time back — trimmed down to [`CUTAWAY_FLOOR_S`], and
/// the last one dropped (its time going to the face) when they cannot — so the beat keeps its
/// length. A beat that already opens on the speaker's footage, one whose speaker is not on screen
/// there, b-roll talking over b-roll, a narrated beat, and one with a cutaway the editor pinned are
/// left alone. Returns how many beats changed.
pub fn open_on_the_speaker(db: &Db, script: &mut Script) -> usize {
    let mut changed = 0;
    for beat in &mut script.beats {
        if beat.narration.as_deref().map(str::trim).is_some_and(|n| !n.is_empty()) {
            continue;
        }
        let Some(bed) = beat.bed.clone() else { continue };
        let Some(first) = beat.clips.first() else { continue };
        if first.video_id == bed.video_id || beat.clips.iter().any(|c| c.pinned()) {
            continue;
        }
        if crate::classify::video_plays_ambience(db, bed.video_id) {
            continue;
        }
        let bed_len = bed.out_s - bed.in_s;
        if bed_len < FACE_FIRST_S + CUTAWAY_FLOOR_S {
            continue;
        }
        let (a, b) = (bed.in_s, bed.in_s + FACE_FIRST_S);
        // Only a face that is there: a voice recorded off camera has nobody to show.
        if let Ok(Some(spans)) = crate::scenes::for_range(db, bed.video_id, a, b)
            && crate::scenes::face_share(&spans, a, b) < 0.5
        {
            continue;
        }

        let mut clips = beat.clips.clone();
        let mut face_s = FACE_FIRST_S;
        let mut need = FACE_FIRST_S;
        loop {
            let room = |c: &ScriptClip| ((c.out_s - c.in_s) - CUTAWAY_FLOOR_S).max(0.0);
            let total: f64 = clips.iter().map(room).sum();
            if total + 1e-6 >= need {
                let share = if total > 0.0 { need / total } else { 0.0 };
                for c in clips.iter_mut() {
                    c.out_s -= room(c) * share;
                }
                need = 0.0;
                break;
            }
            if clips.len() <= 1 {
                break;
            }
            // Not enough give: the last cutaway goes, and what it frees beyond the need is the face's.
            let last = clips.pop().expect("more than one clip");
            let len = (last.out_s - last.in_s).max(0.0);
            if len >= need {
                face_s += len - need;
                need = 0.0;
                break;
            }
            need -= len;
        }
        if need > 0.05 {
            continue;
        }
        face_s = face_s.min(bed_len - CUTAWAY_FLOOR_S);
        let face = ScriptClip {
            exact: false,
            gain_db: None,
            sentences: None,
            framing: None,
            video_id: bed.video_id,
            in_s: a,
            out_s: a + face_s,
            audio: crate::script::Audio::Mute,
            why: Some("the speaker, seen before the cutaway".into()),
            tracks: None,
        };
        clips.insert(0, face);
        beat.clips = clips;
        changed += 1;
    }
    changed
}

/// Take back the closing picture a previous repair added, so the fit sees the model's own cut.
pub fn drop_closing_picture(script: &mut Script) -> bool {
    let Some(beat) = script.beats.last_mut() else { return false };
    if beat.clips.last().and_then(|c| c.why.as_deref()) == Some(CLOSING_PICTURE) {
        beat.clips.pop();
        return true;
    }
    false
}

/// How long [`hold_the_last_picture`] would hold the closing picture of `script` as it stands (s),
/// without changing it: what the fit, which runs before the hold, leaves room for.
pub fn closing_hold_s(db: &Db, project_id: i64, script: &Script, cfg: &crate::config::ScriptConfig) -> f64 {
    let mut probe = script.clone();
    hold_the_last_picture(db, project_id, &mut probe, cfg)
}

/// The least quiet a same-shot closing hold needs to end on that face; with less, the cut closes on
/// a cutaway (s).
const MIN_QUIET_HOLD_S: f64 = 1.0;

/// Let the last picture run on after the last word, in silence, so the piece ends instead of
/// stopping. Returns how long it is held (s).
///
/// Every brain tested cuts on the final syllable. An editor holds the closing image for a moment
/// and lets it go quiet — it is what tells a viewer the thing is over. The hold comes out of
/// footage already chosen and already grounded: the last shot plays on as a quiet clip of its own
/// (`CLOSING_PICTURE`, which the next finish takes back first), and the bed is deliberately not
/// extended with it, because the silence is the point.
pub fn hold_the_last_picture(db: &Db, project_id: i64, script: &mut Script, cfg: &crate::config::ScriptConfig) -> f64 {
    if cfg.closing_hold_s <= 0.0 {
        return 0.0;
    }
    let Some(beat) = script.beats.last_mut() else { return 0.0 };
    // Narration read to the end is already an ending; a hold under it would be dead air.
    if beat.narration.as_deref().map(str::trim).is_some_and(|n| !n.is_empty()) {
        return 0.0;
    }
    // Already held? A hold leaves the last picture outlasting the sound under it, and doing it
    // again on every repair would add two seconds each time.
    let pictures: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();
    match &beat.bed {
        Some(bed) if pictures > bed.duration_s() + 0.2 => return 0.0,
        // Without a bed the voice is the last clip's own, so a silent last picture is a held one.
        None if beat.clips.last().is_some_and(|c| c.audio == crate::script::Audio::Mute) => return 0.0,
        _ => {}
    }

    let Some(last) = beat.clips.last() else { return 0.0 };
    let (last_video, last_in, last_out) = (last.video_id, last.in_s, last.out_s);

    // Closing on the speaker means closing on a talking head with the sound already finished —
    // two seconds of somebody moving their hands in silence, which the editor described as the
    // cut ending "without an image that represents what we were talking about". When the last
    // picture is a face we have been listening to, the hold becomes a cutaway instead.
    //
    // Only under a bed. Without one the closing voice is that picture's own audio, and the hold
    // below mutes the clip it extends — so cutting away there would be swapping the sound out as
    // well as the image, and it changes what the fit is allowed to trim: a draft whose last beat
    // has no bed walked 84.5 s → 52.5 s across two repairs on the strength of that one difference.
    let bedded = beat_has_bed(script);
    if bedded && voices(script).contains(&last_video) {
        let heard = closing_words(db, script);
        if let Some(shot) = closing_shot(db, project_id, script, &heard, cfg.closing_hold_s, &cfg.footage) {
            let beat = script.beats.last_mut().expect("checked above");
            beat.clips.push(ScriptClip {
                exact: false,
                gain_db: None,
                sentences: None,
                framing: None,
                video_id: shot.video_id,
                in_s: shot.t_s,
                out_s: shot.t_s + cfg.closing_hold_s,
                audio: crate::script::Audio::Mute,
                why: Some(CLOSING_PICTURE.into()),
                tracks: None,
            });
            return cfg.closing_hold_s;
        }
    }

    // The hold plays on into the footage after the last clip. When the cut has already shown that
    // footage, the quiet ending is a repeat of an earlier shot: it stops where that one starts.
    let shown = script
        .beats
        .iter()
        .flat_map(|b| &b.clips)
        .filter(|c| c.video_id == last_video && c.out_s > last_out + 0.05 && c.in_s < last_out + cfg.closing_hold_s)
        .map(|c| c.in_s.max(last_out))
        .fold(f64::MAX, f64::min);
    // The same shot holds only while it stays quiet: a speaker who talks again inside the hold is
    // seen talking with the sound gone, and the cut seems to go mute before it ends ("…that we plan
    // to work with" — and Kara carries on, silent, for two seconds). With too little quiet after
    // the last word to end on that face, the cut closes on a cutaway when the footage has one;
    // otherwise cut placement stops the hold before the voice comes back
    // (`cuts::hold_until_quiet`), where the final cut point is known.
    let quiet = crate::cuts::quiet_after_voice(db, last_video, last_in, last_out);
    if quiet.is_some_and(|q| q < MIN_QUIET_HOLD_S) {
        let heard = closing_words(db, script);
        if let Some(shot) = closing_shot(db, project_id, script, &heard, cfg.closing_hold_s, &cfg.footage) {
            let beat = script.beats.last_mut().expect("checked above");
            beat.clips.push(ScriptClip {
                exact: false,
                gain_db: None,
                sentences: None,
                framing: None,
                video_id: shot.video_id,
                in_s: shot.t_s,
                out_s: shot.t_s + cfg.closing_hold_s,
                audio: crate::script::Audio::Mute,
                why: Some(CLOSING_PICTURE.into()),
                tracks: None,
            });
            return cfg.closing_hold_s;
        }
    }
    let Some(beat) = script.beats.last_mut() else { return 0.0 };
    let Some(last) = beat.clips.last_mut() else { return 0.0 };
    let duration = video_duration(db, last.video_id).unwrap_or(f64::MAX).min(shown);
    let room = (duration - last_out).max(0.0);
    let held = cfg.closing_hold_s.min(room);
    if held < 0.2 {
        return 0.0;
    }
    // Whatever was being heard stops where it stopped: this is the quiet at the end. Audio is on or
    // off for a whole clip, so the quiet is its own clip continuing the same shot. Muting the
    // extended clip instead silenced everything said in it: a one-clip cut (a social clip) of
    // someone talking came out with no sound at all.
    //
    // Its own clip under a bed too, where the last picture is already silent: lengthening that
    // picture in place left nothing for `drop_closing_picture` to take back, so the next finish saw
    // a longer picture, grew the bed to it, and held two seconds more on top — every repair.
    let hold = ScriptClip {
        exact: false,
        gain_db: None,
        sentences: None,
        framing: last.framing.clone(),
        video_id: last.video_id,
        in_s: last.out_s,
        out_s: last.out_s + held,
        audio: crate::script::Audio::Mute,
        why: Some(CLOSING_PICTURE.into()),
        tracks: None,
    };
    beat.clips.push(hold);
    held
}

/// Whether the last beat carries its sound on a bed rather than in the pictures themselves.
fn beat_has_bed(script: &Script) -> bool {
    script.beats.last().is_some_and(|b| b.bed.is_some())
}

/// Every video whose voice is heard anywhere in the cut.
fn voices(script: &Script) -> std::collections::HashSet<i64> {
    let mut v = std::collections::HashSet::new();
    for beat in &script.beats {
        if let Some(bed) = &beat.bed {
            v.insert(bed.video_id);
        }
        for c in &beat.clips {
            if c.audio == crate::script::Audio::Source {
                v.insert(c.video_id);
            }
        }
    }
    v
}

/// The last thing said in the cut, which is what the closing image should be about.
fn closing_words(db: &Db, script: &Script) -> String {
    let Some(beat) = script.beats.last() else { return String::new() };
    if let Some(bed) = &beat.bed {
        return clip_speech(db, bed.video_id, bed.in_s, bed.out_s);
    }
    match beat.clips.iter().rev().find(|c| c.audio == crate::script::Audio::Source) {
        Some(c) => clip_speech(db, c.video_id, c.in_s, c.out_s),
        None => String::new(),
    }
}

/// The described picture a cut closes on: where it is.
struct ClosingShot {
    video_id: i64,
    t_s: f64,
}

/// A described shot to close on: nobody talking in it, nothing already used, and of everything
/// left, whatever shares the most with the closing line.
///
/// "Nobody talking" is the part that matters and it is stricter than the rule `chat/build.rs`
/// picks cutaways by. Excluding only the voices *heard in this cut* still closed one on a frame
/// of a different interviewee, lav mic on his collar, mid-sentence — a talking head is a talking
/// head whether or not this piece happens to use his voice. A shot with any speech under it is
/// somebody being interviewed; the closing image wants the place, not a person answering a
/// question. Word overlap and not embeddings, deliberately: this is a tie-break over a handful of
/// eligible shots, not a search.
fn closing_shot(
    db: &Db,
    project_id: i64,
    script: &Script,
    heard: &str,
    need_s: f64,
    footage: &[i64],
) -> Option<ClosingShot> {
    let voices = voices(script);
    let used: std::collections::HashSet<i64> =
        script.beats.iter().flat_map(|b| b.clips.iter().map(|c| c.video_id)).collect();
    let wanted = super::words::content_words(heard);

    let rows: Vec<(i64, f64, String)> = db
        .conn
        .prepare_cached(
            "SELECT f.video_id, f.t_s, f.description_json, v.duration_s
               FROM frames f
               JOIN videos v ON v.id = f.video_id
               JOIN video_files vf ON vf.video_id = f.video_id
               JOIN folders fo ON fo.id = vf.folder_id
               JOIN project_folders pf ON pf.folder_id = fo.id
              WHERE pf.project_id = ?1 AND f.description_json IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM project_exclusions x
                  WHERE x.project_id = pf.project_id AND x.video_id = f.video_id)
              GROUP BY f.video_id, f.t_s",
        )
        .ok()?
        .query_map(params![project_id], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?, r.get::<_, String>(2)?, r.get::<_, f64>(3)?))
        })
        .ok()?
        .flatten()
        // Room to play the whole hold, not a voice we have been listening to, and not somebody
        // talking on camera.
        .filter(|(vid, t_s, _, dur)| {
            !voices.contains(vid)
                && !used.contains(vid)
                && (footage.is_empty() || footage.contains(vid))
                && dur - t_s >= need_s
                && !clip_has_speech(db, *vid, *t_s, t_s + need_s)
        })
        .filter_map(|(vid, t_s, json, _)| {
            let text = serde_json::from_str::<Value>(&json).ok()?["description"].as_str()?.to_string();
            Some((vid, t_s, text))
        })
        .collect();

    rows.into_iter()
        .max_by_key(|(_, _, text)| super::words::relevance(&wanted, text))
        .map(|(video_id, t_s, _)| ClosingShot { video_id, t_s })
}

/// Keep every bed inside the beat it plays under.
///
/// Beds are laid before the cut is fitted to its target, and fitting trims the pictures. A bed
/// left at its old length then outlives them: it plays on over the next beat's pictures, and the
/// export puts overlapping clips on A1. Whatever moved the clips, this puts the sound back inside
/// its beat.
pub fn clamp_beds_to_beats(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) {
    for beat in &mut script.beats {
        let beat_len: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();
        let Some(bed) = &beat.bed else { continue };
        if beat_len <= 0.0 {
            beat.bed = None;
            continue;
        }
        if bed.duration_s() <= beat_len {
            continue;
        }

        // This is the last thing that touches a bed, and it is where a speaker gets cut off:
        // `end_on_sentences` puts the voice on a sentence, then `trim_pictures_to_bed` and the
        // closing hold move the pictures, and trimming the bed to fit them lands mid-word.
        //
        // The pictures give way to the voice, so when cutting the bed here would land inside a
        // sentence, the beat's last shot is held for the seconds the voice still needs instead.
        // Bounded by the budget a clip gets for the same job; past that, the trim stands and the
        // score reports the cut.
        let wanted = bed.duration_s();
        let short_by = wanted - beat_len;
        let segs = sentence_edges(db, bed.video_id);
        if short_by <= cfg.max_speech_extend_s && cuts_into_a_sentence(&segs, bed.in_s + beat_len) {
            if let Some(last) = beat.clips.last_mut().filter(|c| c.movable(Side::Out)) {
                let room = (video_duration(db, last.video_id).unwrap_or(f64::MAX) - last.out_s).max(0.0);
                last.out_s += short_by.min(room);
            }
            let beat_len: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();
            if let Some(bed) = &mut beat.bed
                && bed.duration_s() > beat_len
            {
                bed.out_s = bed.in_s + beat_len;
            }
            continue;
        }
        if let Some(bed) = &mut beat.bed {
            bed.out_s = bed.in_s + beat_len;
        }
    }
}

/// How far a voice can keep going from `from_s` towards `wanted_end`: never into the interviewer's
/// next question, and stopping on a whole sentence. Shared by the beds the editor asks for and the
/// ones laid here, so both end the same way.
fn speech_runs_until(db: &Db, video_id: i64, from_s: f64, wanted_end: f64, cfg: &crate::config::ScriptConfig) -> f64 {
    let mut end = wanted_end;
    let segs: Vec<(f64, f64, bool)> = db
        .conn
        .prepare(
            "SELECT start_s, end_s, COALESCE(off_mic, 0) FROM transcript_segments
             WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3 ORDER BY start_s",
        )
        .and_then(|mut st| {
            st.query_map(params![video_id, from_s, end], |r| {
                Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?, r.get::<_, i64>(2)? != 0))
            })
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
        })
        .unwrap_or_default();
    if let Some((off_start, _, _)) = segs.iter().find(|(_, _, off)| *off) {
        end = end.min(*off_start);
    }
    match segs.iter().rev().find(|(_, e, off)| !*off && *e <= end + cfg.speech_overrun_s) {
        // Stop where the sentence stops, the way a clip does.
        Some((_, seg_end, _)) => (*seg_end + cfg.speech_overrun_s).min(wanted_end),
        // Nobody speaks out here. Running the bed on would only add silence that has been faded
        // out anyway, and would claim in the export that there is audio to hear.
        None => from_s,
    }
}

/// A clip someone talks through this much of (on the mic) is a talking head, not a cutaway (0–1).
pub const TALKING_SHARE: f64 = 0.3;

/// Share of `in_s..out_s` of a video covered by on-mic speech (0–1). A line nobody has marked
/// either way is on the microphone, as everywhere else: `NOT off_mic` alone read unmarked speech
/// as nobody talking.
pub fn on_mic_share(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> f64 {
    let len = out_s - in_s;
    if len <= 0.0 {
        return 0.0;
    }
    let spoken: f64 = db
        .conn
        .query_row(
            "SELECT COALESCE(SUM(MIN(end_s, ?3) - MAX(start_s, ?2)), 0) FROM transcript_segments \
             WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3 AND COALESCE(off_mic, 0) = 0",
            params![video_id, in_s, out_s],
            |r| r.get(0),
        )
        .unwrap_or(0.0);
    (spoken / len).clamp(0.0, 1.0)
}

/// Let a speaker's voice run under the pictures that follow, instead of stopping at the cutaway.
///
/// A beat usually opens on the person talking and then cuts to what they are describing. Until
/// beds existed the second half went silent, because sound belonged to whichever clip was on
/// screen — so every teaser alternated a talking head with a mute postcard. Here the beat's
/// speech is extended across the whole beat and the pictures play under it.
///
/// The bed is aligned so the face stays in sync: it starts as far *before* the speaking clip's
/// own in-point as that clip sits into the beat, so when we reach the face, the audio is exactly
/// where it would have been. It stops at a sentence end, before the interviewer's next question,
/// and never runs further than `max_bed_extend_s` past the clip it came from.
pub fn lay_audio_beds(db: &Db, s: &mut Script, cfg: &crate::config::ScriptConfig) -> Vec<Issue> {
    let mut issues = Vec::new();
    if !cfg.infer_audio_beds {
        return issues;
    }
    let mut laid = 0usize;

    let breaths: Vec<f64> = (0..s.beats.len()).map(|b| breath_after(s, b, cfg)).collect();
    for (beat, &breath) in s.beats.iter_mut().zip(&breaths) {
        let beat_len: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();

        // A bed the editor asked for is checked, not trusted: it has to be speech, and it cannot
        // outlast the pictures it plays under or it would run into the next beat.
        if let Some(bed) = beat.bed.take() {
            let mut bed = bed;
            if bed.duration_s() > beat_len && beat_len > 0.0 {
                bed.out_s = bed.in_s + beat_len;
            } else if beat_len > bed.duration_s() + 0.05 && bed.movable(Side::Out) && !bed.inferred {
                // Written to the length of the speaker's own clip, not the beat's: the pictures
                // after it would play silent, which is the thing a bed exists to prevent. A bed
                // this pass laid on an earlier finish was sized by the rule below and is left at
                // it: grown again by this one, it came out longer on every finish.
                // The breath after the voice is the pictures' own ([`breath_after`]): the voice
                // is not grown into it.
                let wanted = (bed.in_s + beat_len - breath).min(bed.out_s + cfg.max_bed_extend_s);
                let grown = speech_runs_until(db, bed.video_id, bed.out_s, wanted, cfg);
                if grown > bed.out_s + 0.05 {
                    bed.out_s = grown;
                }
            }
            if !clip_has_speech(db, bed.video_id, bed.in_s, bed.out_s) {
                issues.push(Issue {
                    fixes: Vec::new(),
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: None,
                    message: format!(
                        "dropped the audio bed: nobody speaks at video #{} {:.1}–{:.1} s",
                        bed.video_id, bed.in_s, bed.out_s
                    ),
                });
            } else {
                for c in &mut beat.clips {
                    c.audio = crate::script::Audio::Mute;
                }
                beat.bed = Some(bed);
                continue;
            }
        }

        if beat.clips.len() < 2 {
            continue;
        }
        // Where does the voice come from, and how far into the beat does its picture sit?
        let mut offset = 0.0f64;
        let mut source: Option<(usize, f64)> = None;
        for (i, c) in beat.clips.iter().enumerate() {
            if c.audio == crate::script::Audio::Source
                && c.tracks.as_ref().is_none_or(|t| !t.is_empty())
                && clip_has_speech(db, c.video_id, c.in_s, c.out_s)
            {
                source = Some((i, offset));
                break;
            }
            offset += (c.out_s - c.in_s).max(0.0);
        }
        let Some((idx, lead)) = source else { continue };

        let beat_dur = beat_len;
        let clip = &beat.clips[idx];
        let clip_dur = (clip.out_s - clip.in_s).max(0.0);
        // Only worth a bed when something other than this clip is on screen.
        if beat_dur <= clip_dur + 0.05 {
            continue;
        }

        let start = (clip.in_s - lead).max(0.0);
        let wanted_end = start + beat_dur;
        let cap = clip.out_s + cfg.max_bed_extend_s;
        let capped = wanted_end.min(cap);

        // Never run into the interviewer's next question, and stop on a whole sentence.
        let end = speech_runs_until(db, clip.video_id, clip.out_s, capped, cfg);

        // Under a voice, only pictures nobody speaks in: b-roll, or the same footage inside the
        // stretch the voice comes from (its lips are in sync). Another person talking on camera,
        // muted under this voice, is a mouth moving to someone else's words — the husband on
        // screen with his wife's line. Then each clip keeps its own sound instead.
        // In sync means playing the very moment the bed plays at that point of the beat, the rule
        // `script_check` holds a bed to: the same take further along is the same face saying
        // something else.
        let source_video = clip.video_id;
        let mut at = 0.0;
        let others_talk = beat.clips.iter().enumerate().any(|(j, c)| {
            let in_sync = c.video_id == source_video && (c.in_s - (start + at)).abs() <= 0.05;
            at += (c.out_s - c.in_s).max(0.0);
            j != idx && !in_sync && on_mic_share(db, c.video_id, c.in_s, c.out_s) >= TALKING_SHARE
        });
        if others_talk {
            continue;
        }
        // Worth a bed when the voice carries past its own picture, or begins under one before
        // it — the two halves of a J-cut. When it does neither, the old behaviour is already right.
        let trails = end > clip.out_s + 0.05;
        let leads = start < clip.in_s - 0.05;
        if !trails && !leads {
            continue;
        }

        beat.bed = Some(crate::script::AudioBed {
            sentences: None,
            video_id: clip.video_id,
            in_s: start,
            out_s: end,
            why: None,
            inferred: true,
            tracks: clip.tracks.clone(),
        });
        for c in &mut beat.clips {
            c.audio = crate::script::Audio::Mute;
        }
        laid += 1;
    }

    if laid > 0 {
        issues.push(Issue {
            fixes: Vec::new(),
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("let the speaker's voice run under the pictures in {laid} beat(s)"),
        });
    }
    issues
}

#[cfg(test)]
mod closing_tests {
    use super::*;

    #[test]
    fn closing_shot_respects_project_exclusions() {
        let (db, project) = super::super::tests::video(&[]);
        let script = super::super::tests::cut(None, &[]);
        db.conn
            .execute(
                "INSERT INTO frames(video_id,t_s,description_json) VALUES (7,0,'{\"description\":\"landscape\"}')",
                [],
            )
            .unwrap();
        assert!(closing_shot(&db, project, &script, "landscape", 2.0, &[]).is_some());
        db.conn
            .execute("INSERT INTO project_exclusions(project_id,video_id,excluded_at) VALUES (?1,7,0)", [project])
            .unwrap();
        assert!(closing_shot(&db, project, &script, "landscape", 2.0, &[]).is_none());
    }
}
