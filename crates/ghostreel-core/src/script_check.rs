//! Shared script checker: checks bounds, motion steadiness, sentence boundaries,
//! speech breath, clip duration limits, and footage reuse.

use serde_json::Value;

use crate::Error;
use crate::config::ScriptConfig;
use crate::db::Db;
use crate::script::{Audio, Issue, IssueSeverity, Script, ScriptClip};

/// Share of a clip that may be shaky before it counts as shaky, the line the chat pipeline uses
/// to move clips.
pub const SHAKY_SHARE: f64 = 0.5;

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Silence before a clip's first word (or past the tail after its last) longer than this plays as
/// dead air (s).
const DEAD_AIR_S: f64 = 0.5;

/// Options to customize script checking.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CheckOptions {
    pub clip_min_s: Option<f64>,
    pub clip_max_s: Option<f64>,
    pub forbid_overlap: bool,
    /// The shortest a fix (or repair) may trim a clip to, e.g. a clip run's minimum length.
    /// `None` keeps most of each clip.
    pub trim_min_s: Option<f64>,
}

/// How steady `in_s..out_s` of a video is, judged with the script settings' limits; `None` when
/// its motion was never measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Steadiness {
    pub camera: &'static str,
    pub jerk_limit: f64,
    pub sway_limit: f64,
    pub worst_jerk: f64,
    pub worst_sway: f64,
    pub spans: Vec<(f64, f64)>,
    pub shaky_s: f64,
    /// When over half the range is shaky: start of the nearest steady stretch as long as the range.
    pub steady_start: Option<f64>,
    pub len: f64,
}

impl Steadiness {
    pub fn measure(
        windows: &[crate::steadiness::Window],
        cfg: &crate::config::ScriptConfig,
        in_s: f64,
        out_s: f64,
    ) -> Option<Self> {
        use crate::steadiness as st;
        let inside: Vec<_> = windows.iter().filter(|w| w.end_s > in_s && w.start_s < out_s).collect();
        if inside.is_empty() {
            return None;
        }
        let jerk_limit = st::shake_limit(windows, cfg.max_shake_jerk, cfg.shake_relative);
        let spans = st::shaky_spans_with_sway(windows, in_s, out_s, jerk_limit, cfg.max_sway);
        // A fold from 0.0: an empty float `sum()` is -0.0.
        let shaky_s = spans.iter().fold(0.0, |acc, (a, b)| acc + (b - a));
        let len = (out_s - in_s).max(0.0);
        let steady_start = (len > 0.0 && shaky_s > len * SHAKY_SHARE)
            .then(|| st::steady_stretch_near(windows, in_s, len, jerk_limit, cfg.max_sway))
            .flatten();
        Some(Self {
            camera: st::camera_style(windows).as_str(),
            jerk_limit,
            sway_limit: cfg.max_sway,
            worst_jerk: inside.iter().map(|w| w.jerk).fold(0.0, f64::max),
            worst_sway: inside.iter().map(|w| w.sway).fold(0.0, f64::max),
            spans,
            shaky_s,
            steady_start,
            len,
        })
    }

    pub fn verdict(&self) -> &'static str {
        if self.shaky_s <= 0.0 {
            "steady"
        } else if self.shaky_s > self.len * SHAKY_SHARE {
            "shaky"
        } else {
            "some shake"
        }
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "verdict": self.verdict(),
            "camera": self.camera,
            "shaky_s": round2(self.shaky_s),
            "shaky_spans": self.spans.iter().map(|(a, b)| [round2(*a), round2(*b)]).collect::<Vec<_>>(),
            "worst_jerk": round2(self.worst_jerk),
            "jerk_limit": round2(self.jerk_limit),
            "worst_sway": round2(self.worst_sway),
            "sway_limit": round2(self.sway_limit),
            "steady_alternative": self.steady_start.map(|s| [round2(s), round2(s + self.len)]),
        })
    }
}

/// The repair passes a script written outside the chat gets before it is checked and saved
/// (`ghostreel script import`, clip runs): snap to speech segments, lay audio beds, end on the
/// speaker's turn and on whole sentences, cut pictures back to the voice, hold the last picture,
/// keep vertical framing off screen shares, clamp beds, and last of all place every cut on the
/// sound itself (`cuts.rs`, for videos the speech stage has measured). Returns one line per pass
/// that changed something, and none when the script comes out as it went in.
///
/// `min_clip_s`: the shortest a clip may be trimmed to (a clip run's length range); `None` keeps
/// most of each clip.
pub fn repair_for_import(
    db: &Db,
    project_id: i64,
    script: &mut Script,
    cfg: &ScriptConfig,
    min_clip_s: Option<f64>,
) -> Result<Vec<String>, Error> {
    let mut notes = Vec::new();
    let before = script.clone();
    let snapped = crate::script::snap_to_segments(db, script)?;
    if snapped > 0 {
        notes.push(format!("snapped {snapped} clip boundary(ies) to speech segments"));
    }
    // A beat that would fall silent when it cuts away keeps the voice running under the pictures.
    for issue in crate::chat::lay_audio_beds(db, script, cfg) {
        notes.push(issue.message);
    }
    // A range stops where the speaker stopped, and then on a whole sentence.
    let turns = crate::chat::end_on_turns(db, script, cfg);
    if turns > 0 {
        notes.push(format!(
            "ended {} where the speaker stopped, before the reply over the top",
            if turns == 1 { "one quote".to_string() } else { format!("{turns} quotes") }
        ));
    }
    let mended = crate::chat::end_on_sentences(db, script, cfg);
    if mended > 0 {
        notes.push(if mended == 1 {
            "restored one trimmed quote to its full sentence".to_string()
        } else {
            format!("restored {mended} trimmed quotes to their full sentences")
        });
    }
    let shortened = crate::chat::trim_pictures_to_bed(script, cfg);
    if shortened > 0 {
        notes.push(format!(
            "cut the pictures back to the voice in {}",
            if shortened == 1 { "one beat".to_string() } else { format!("{shortened} beats") }
        ));
    }
    let held = crate::chat::hold_the_last_picture(db, project_id, script, cfg);
    if held > 0.0 {
        notes.push(format!("held the closing picture for {held:.1} s of quiet"));
    }
    // A vertical cut neither opens nor closes on a screen share when a sentence edge allows.
    notes.extend(crate::script_fix::repair_framing(db, script, cfg.speech_overrun_s, min_clip_s));
    // A bed may not outlast the pictures it plays under, and everything above moves them.
    crate::chat::clamp_beds_to_beats(db, script, cfg);
    // Last: every pass above moves clips by the transcript's times, which are approximate.
    notes.extend(crate::cuts::place_cuts(db, script, cfg).iter().map(crate::cuts::CutMove::message));
    // The passes take turns (the transcript's times pull a cut one way, the sound puts it back):
    // a script they leave exactly as it came in was not changed, whatever each pass did on the way.
    if *script == before {
        notes.clear();
    }
    Ok(notes)
}

/// Warnings for clips that sit mostly on shaky stretches, with where to move them.
pub fn shake_issues(db: &Db, cfg: &ScriptConfig, s: &Script) -> Vec<Issue> {
    let mut issues = Vec::new();
    if cfg.max_shake_jerk <= 0.0 {
        return issues;
    }
    for beat in &s.beats {
        for (i, c) in beat.clips.iter().enumerate() {
            let windows = db.motion_windows(c.video_id).unwrap_or_default();
            let Some(st) = Steadiness::measure(&windows, cfg, c.in_s, c.out_s) else { continue };
            if st.verdict() != "shaky" {
                continue;
            }
            let fix = match st.steady_start {
                Some(a) => format!("; nearest steady stretch {a:.1}–{:.1} s", a + st.len),
                None => "; no steady stretch that long in this video".to_string(),
            };
            issues.push(Issue {
                fixes: Vec::new(),
                severity: IssueSeverity::Warning,
                beat_id: Some(beat.id.clone()),
                clip_index: Some(i),
                message: format!(
                    "shaky: video #{} {:.1}–{:.1} s shakes for {:.1} of {:.1} s{fix}",
                    c.video_id, c.in_s, c.out_s, st.shaky_s, st.len
                ),
            });
        }
    }
    issues
}

/// Framed clips (vertical or square cuts of wider footage) that will not look right, each with
/// what would fix it:
/// - the faces (or, for follow the speaker, the speakers) of a video were never indexed: the
///   stage is offered (and the app queues it when the project has it on);
/// - a speaking clip opens or closes on a screen share: a trim onto whole sentences, when one
///   keeps the clip long enough, else other layouts for that stretch;
/// - the closing hold shows no face: a shorter hold;
/// - the framed face is missing for more than a second, cut off, or too small: other layouts.
///
/// Needs the script's width and height (fill them from the project). Computed from the index as
/// it is now, so it never goes stale.
pub fn framing_issues(
    db: &Db,
    project_id: i64,
    script: &Script,
    cfg: &ScriptConfig,
    min_clip_s: Option<f64>,
) -> Vec<Issue> {
    use crate::framing::Layout;
    use crate::script::{Fix, FixAction};
    let Ok(plans) = crate::framing::plans_for_script(db, None, script) else { return Vec::new() };
    let pipeline = db.project(project_id).ok().map(|p| p.pipeline);
    let enabled = |stage: &str| crate::index::stage_enabled(stage, None, pipeline.as_ref());
    let stage_fix = |stage: &str, what: &str| {
        let on = enabled(stage);
        Fix::new(
            if on { format!("Find {what}") } else { format!("Turn on {what} and find them") },
            FixAction::RunStage { stage: stage.to_string(), enable: !on },
        )
    };
    let mut issues = Vec::new();
    let mut asked: std::collections::HashSet<(i64, &str)> = std::collections::HashSet::new();
    let hold = crate::script_fix::closing_hold(script);
    for cf in plans {
        let Some(beat) = script.beats.get(cf.beat_index) else { continue };
        let Some(clip) = beat.clips.get(cf.clip_index) else { continue };
        let beat_id = beat.id.clone();
        let issue = |severity: IssueSeverity, message: String, fixes: Vec<Fix>| Issue {
            severity,
            beat_id: Some(beat_id.clone()),
            clip_index: Some(cf.clip_index),
            message,
            fixes,
        };
        // Missing data is a job to run, not a verdict on the clip.
        if !crate::scenes::faces_analysed(db, cf.video_id) {
            if asked.insert((cf.video_id, "faces")) {
                issues.push(issue(
                    IssueSeverity::Warning,
                    format!(
                        "framing: the faces of video #{} aren't indexed yet, so the crop has no one to frame and shows the whole picture",
                        cf.video_id
                    ),
                    vec![stage_fix("faces", "faces")],
                ));
            }
            continue;
        }
        let plan = &cf.plan;
        let two_people = plan.subjects.iter().filter(|s| s.presence >= crate::framing::MIN_PRESENCE).count() >= 2;
        let follows =
            plan.segments.iter().any(|s| matches!(s.layout, Layout::FollowSpeaker | Layout::Stacked | Layout::Screen));
        if two_people
            && follows
            && !plan.used_turns
            && !speakers_analysed(db, cf.video_id)
            && asked.insert((cf.video_id, "speakers"))
        {
            issues.push(issue(
                IssueSeverity::Warning,
                format!(
                    "framing: the speakers of video #{} aren't indexed yet, so the crop holds on one face instead of following whoever talks",
                    cf.video_id
                ),
                vec![stage_fix("speakers", "speakers")],
            ));
        }
        let is_hold = hold == Some((cf.beat_index, cf.clip_index));
        if is_hold {
            if let Some((_, _, good)) = crate::script_fix::hold_on_face(db, script) {
                let len = clip.out_s - clip.in_s;
                if good < len - 0.05 {
                    let fixes = if good >= crate::script_fix::MIN_HOLD_S {
                        vec![Fix::new(
                            format!("Shorten the hold to {good:.1} s"),
                            FixAction::Trim {
                                beat_id: beat_id.clone(),
                                clip_index: cf.clip_index,
                                in_s: clip.in_s,
                                out_s: clip.in_s + good,
                            },
                        )]
                    } else {
                        Vec::new()
                    };
                    issues.push(issue(
                        IssueSeverity::Warning,
                        format!(
                            "framing: the closing hold shows a screen share, not a face ({:.1} of {len:.1} s on a face)",
                            good
                        ),
                        fixes,
                    ));
                }
            }
            continue;
        }
        // Stretches not framed on people: a screen share at the clip's edge can often be trimmed.
        let trim =
            crate::script_fix::framing_trim(db, script, cf.beat_index, cf.clip_index, min_clip_s, cfg.speech_overrun_s);
        let mut trim_offered = false;
        let framing = crate::framing::effective(script, clip).cloned().unwrap_or_default();
        for seg in plan.segments.iter().filter(|s| !crate::script_fix::frames_people(s.layout)) {
            let (a, b) = (cf.in_s + seg.start, cf.in_s + seg.end);
            // Too short to matter, or framed this way because the editor chose it.
            if b - a < crate::script_fix::MIN_BAD_EDGE_S || crate::script_fix::chosen(&framing, a, b) {
                continue;
            }
            let edge = if seg.start <= 0.05 {
                Some("opens on")
            } else if (plan.segments.last().map(|l| l.end).unwrap_or(0.0) - seg.end).abs() < 0.05 {
                Some("closes on")
            } else {
                None
            };
            let what = seg.scene.phrase();
            let mut fixes = Vec::new();
            if let (Some(_), Some((ti, to)), false) = (edge, trim, trim_offered) {
                trim_offered = true;
                fixes.push(Fix::new(
                    format!("Trim to {ti:.1}–{to:.1} s (whole sentences)"),
                    FixAction::Trim { beat_id: beat_id.clone(), clip_index: cf.clip_index, in_s: ti, out_s: to },
                ));
            }
            fixes.extend(crate::script_fix::layout_fixes(&beat_id, cf.clip_index, a, b, seg.layout));
            // Framed deterministically already: worth knowing, not a problem, unless a trim
            // would take it out.
            let severity = if fixes.first().is_some_and(|f| matches!(f.action, FixAction::Trim { .. })) {
                IssueSeverity::Warning
            } else {
                IssueSeverity::Info
            };
            let place = match edge {
                Some(e) => format!("the clip {e} {what}"),
                None => format!("{what} in the middle of the clip"),
            };
            issues.push(issue(
                severity,
                format!(
                    "framing: {place} ({a:.1}–{b:.1} s of video #{}), framed as {}",
                    cf.video_id,
                    seg.layout.label()
                ),
                fixes,
            ));
        }
        // How well the faces fit, stretch by stretch.
        for seg in plan.segments.iter().filter(|s| !s.paths.is_empty()) {
            let q = seg.quality;
            let (a, b) = (cf.in_s + seg.start, cf.in_s + seg.end);
            let layout = seg.layout.as_str();
            let fixes = || crate::script_fix::layout_fixes(&beat_id, cf.clip_index, a, b, seg.layout);
            if q.longest_no_face_s > 1.0 {
                let at = cf.in_s + q.no_face_at.unwrap_or(seg.start);
                issues.push(issue(
                    IssueSeverity::Warning,
                    format!(
                        "framing ({layout}): no face for {:.1} s from {at:.1} s of video #{}; the crop holds there",
                        q.longest_no_face_s, cf.video_id
                    ),
                    fixes(),
                ));
            }
            if q.cut_off_share > 0.2 {
                issues.push(issue(
                    IssueSeverity::Warning,
                    format!(
                        "framing ({layout}): the framed face is cut off in {:.0}% of {a:.1}–{b:.1} s (video #{})",
                        q.cut_off_share * 100.0,
                        cf.video_id
                    ),
                    fixes(),
                ));
            }
            if q.face_frac > 0.0 && q.face_frac < 0.12 {
                issues.push(issue(
                    IssueSeverity::Warning,
                    format!(
                        "framing ({layout}): the face is small for the crop in {a:.1}–{b:.1} s ({:.0}% of the frame height; aim for 20-28%)",
                        q.face_frac * 100.0
                    ),
                    fixes(),
                ));
            }
        }
    }
    issues
}

/// Whether the speakers stage has measured this video's voices.
pub fn speakers_analysed(db: &Db, video_id: i64) -> bool {
    db.conn
        .query_row(
            "SELECT EXISTS (SELECT 1 FROM speaker_windows WHERE video_id = ?1)
                 OR EXISTS (SELECT 1 FROM speaker_turns WHERE video_id = ?1)",
            [video_id],
            |r| r.get::<_, bool>(0),
        )
        .unwrap_or(false)
}

/// The indexing stages the issues ask to run without turning anything on (the project has them
/// on): what a host queues on its own.
pub fn stages_to_queue(issues: &[Issue]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for f in issues.iter().flat_map(|i| &i.fixes) {
        if let crate::script::FixAction::RunStage { stage, enable: false } = &f.action
            && !out.contains(stage)
        {
            out.push(stage.clone());
        }
    }
    out
}

/// Say on each issue asking for `stage` that it has been queued.
pub fn mark_queued(issues: &mut [Issue], stage: &str) {
    for i in issues.iter_mut() {
        let asks = i
            .fixes
            .iter()
            .any(|f| matches!(&f.action, crate::script::FixAction::RunStage { stage: s, .. } if s == stage));
        if asks && !i.message.ends_with("re-plans when it finishes") {
            i.message.push_str("; queued, the crop re-plans when it finishes");
        }
    }
}

/// Check a script against project footage, steadiness, sentences, durations, and boundaries.
///
/// Returns issues in order:
/// a. everything `script::validate` returns;
/// b. shake warnings;
/// c. sentence warnings;
/// d. breath info;
/// e. clip duration warnings;
/// f. clip overlap errors;
/// g. framing warnings (framed clips whose face is missing, cut off or too small).
pub fn check(
    db: &Db,
    project_id: i64,
    script: &Script,
    cfg: &ScriptConfig,
    opts: &CheckOptions,
) -> Result<Vec<Issue>, Error> {
    // a. everything script::validate returns
    let mut issues = crate::script::validate(db, project_id, script)?;

    // b. shake warnings
    issues.extend(shake_issues(db, cfg, script));

    // Cache sentence edges per video_id
    let mut sentence_cache: std::collections::HashMap<i64, Vec<(f64, f64)>> = std::collections::HashMap::new();

    // c. sentence warnings: for clips with audio == Audio::Source on a video that has transcript segments,
    // Warning if in_s or out_s falls inside a sentence (not within 0.3 s of a sentence edge from sentence_edges).
    // Message names video id, the time, and the nearest sentence edge.
    for beat in &script.beats {
        for (clip_idx, clip) in beat.clips.iter().enumerate() {
            if clip.audio != Audio::Source {
                continue;
            }
            let edges =
                sentence_cache.entry(clip.video_id).or_insert_with(|| crate::chat::sentence_edges(db, clip.video_id));
            if edges.is_empty() {
                continue;
            }

            // An edge written as a word ("s412@3") is inside a sentence on purpose: the editor asked
            // for it. Seconds that land inside one are what this is for.
            let on_word = |r: Option<&str>| r.is_some_and(|t| t.contains('@'));
            let in_on_word = on_word(clip.sentences.as_ref().map(|r| r.from.as_str()));
            let out_on_word = on_word(clip.sentences.as_ref().map(|r| r.to.as_str()));

            // Check in_s
            for &(start, end) in edges.iter().filter(|_| !in_on_word) {
                if clip.in_s > start && clip.in_s < end {
                    let d_start = clip.in_s - start;
                    let d_end = end - clip.in_s;
                    if d_start > 0.3 && d_end > 0.3 {
                        let nearest = if d_start <= d_end { start } else { end };
                        issues.push(Issue {
                            fixes: Vec::new(),
                            severity: IssueSeverity::Warning,
                            beat_id: Some(beat.id.clone()),
                            clip_index: Some(clip_idx),
                            message: format!(
                                "video #{}: in_s {:.1} s falls inside a sentence; nearest sentence edge is at {:.1} s",
                                clip.video_id, clip.in_s, nearest
                            ),
                        });
                    }
                    break;
                }
            }

            // Check out_s
            for &(start, end) in edges.iter().filter(|_| !out_on_word) {
                if clip.out_s > start && clip.out_s < end {
                    let d_start = clip.out_s - start;
                    let d_end = end - clip.out_s;
                    // The repair passes run a cut `speech_overrun_s` into the next line so the
                    // last word's decay survives; that is not cutting into a sentence.
                    let overrun_ok = (cfg.speech_overrun_s + 0.05).max(0.3);
                    if d_start > overrun_ok && d_end > 0.3 {
                        let nearest = if d_start <= d_end { start } else { end };
                        issues.push(Issue {
                            fixes: Vec::new(),
                            severity: IssueSeverity::Warning,
                            beat_id: Some(beat.id.clone()),
                            clip_index: Some(clip_idx),
                            message: format!(
                                "video #{}: out_s {:.1} s falls inside a sentence; nearest sentence edge is at {:.1} s",
                                clip.video_id, clip.out_s, nearest
                            ),
                        });
                    }
                    break;
                }
            }
        }
    }

    // c1. a silent talking head: a muted clip over someone speaking, with no bed or voice-over in its
    // beat to carry a voice, shows a mouth moving and plays nothing. Muting is right for b-roll and
    // under a bed; over the speech the piece is cut from, it is a slip whatever model made it.
    // The closing hold the save path lays after the last words is muted on purpose.
    let hold = crate::script_fix::closing_hold(script);
    for (beat_idx, beat) in script.beats.iter().enumerate() {
        let voiced = beat.bed.is_some() || beat.narration.as_deref().is_some_and(|n| !n.trim().is_empty());
        if voiced {
            continue;
        }
        for (clip_idx, clip) in beat.clips.iter().enumerate() {
            let len = clip.out_s - clip.in_s;
            if clip.audio != Audio::Mute || len <= 0.0 || hold == Some((beat_idx, clip_idx)) {
                continue;
            }
            let spoken: f64 = db
                .conn
                .query_row(
                    "SELECT COALESCE(SUM(MIN(end_s, ?3) - MAX(start_s, ?2)), 0) FROM transcript_segments \
                     WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3",
                    rusqlite::params![clip.video_id, clip.in_s, clip.out_s],
                    |r| r.get(0),
                )
                .unwrap_or(0.0);
            if spoken < 0.5 * len {
                continue;
            }
            issues.push(Issue {
                fixes: vec![crate::script::Fix::new(
                    "Play its sound",
                    crate::script::FixAction::SetAudio { beat_id: beat.id.clone(), clip_index: clip_idx, audio: Audio::Source },
                )],
                severity: IssueSeverity::Warning,
                beat_id: Some(beat.id.clone()),
                clip_index: Some(clip_idx),
                message: format!(
                    "video #{}: {:.1}–{:.1} s is muted, but someone speaks through {:.0}% of it and the beat has no \
                     bed or narration: the viewer sees a mouth move and hears nothing. Set its audio to \"source\".",
                    clip.video_id,
                    clip.in_s,
                    clip.out_s,
                    100.0 * spoken / len
                ),
            });
        }
    }

    // c3. edges on a change of picture: a clip that opens inside a dissolve or just before a change
    // shows a flash of the shot it was not cut for (and the same at its end). Measured from the
    // pixels around each edge (`transitions`); an edge that runs straight on into the next clip of
    // the same footage is no cut at all.
    let clips: Vec<(&crate::script::Beat, usize, &ScriptClip)> =
        script.beats.iter().flat_map(|b| b.clips.iter().enumerate().map(move |(i, c)| (b, i, c))).collect();
    for (k, &(beat, clip_idx, clip)) in clips.iter().enumerate() {
        let runs_on = |other: Option<&(&crate::script::Beat, usize, &ScriptClip)>, a: f64, b: f64| {
            other.is_some_and(|o| o.2.video_id == clip.video_id && (a - b).abs() < 0.05)
        };
        let prev = k.checked_sub(1).and_then(|p| clips.get(p));
        let next = clips.get(k + 1);
        let edges = [
            (true, clip.in_s, prev.is_some_and(|p| runs_on(Some(p), p.2.out_s, clip.in_s))),
            (false, clip.out_s, next.is_some_and(|n| runs_on(Some(n), clip.out_s, n.2.in_s))),
        ];
        for (opening, edge, continuous) in edges {
            if continuous {
                continue;
            }
            let near = crate::transitions::around(db, clip.video_id, edge);
            let Some(t) = crate::transitions::edge_hazard(&near, edge, opening) else { continue };
            let what = if t.is_cut() {
                format!("a cut in the source at {:.2} s", t.start)
            } else {
                format!("a change of picture over {:.2}–{:.2} s", t.start, t.end)
            };
            let (moved, side) = if opening {
                ((t.end + 0.05, clip.out_s), "opens on")
            } else {
                ((clip.in_s, t.start - 0.05), "closes on")
            };
            let fixes = if moved.1 - moved.0 >= 1.0 {
                vec![crate::script::Fix::new(
                    format!("Trim to {:.2}–{:.2}", moved.0, moved.1),
                    crate::script::FixAction::Trim {
                        beat_id: beat.id.clone(),
                        clip_index: clip_idx,
                        in_s: moved.0,
                        out_s: moved.1,
                    },
                )]
            } else {
                Vec::new()
            };
            issues.push(Issue {
                fixes,
                severity: IssueSeverity::Warning,
                beat_id: Some(beat.id.clone()),
                clip_index: Some(clip_idx),
                message: format!(
                    "video #{}: the clip {side} {what}: its {} frames show the other picture",
                    clip.video_id,
                    if opening { "first" } else { "last" }
                ),
            });
        }
    }

    // c4. a voice over a screen share, in a cut the shape of its footage: the speaker is only a
    // camera tile in a corner. When the screen is not the point, their own tile can be framed for
    // that stretch (the lips stay in sync, unlike a face from another moment under a bed).
    let out = (script.width.unwrap_or(0) as f64, script.height.unwrap_or(0) as f64);
    for beat in &script.beats {
        for (clip_idx, clip) in beat.clips.iter().enumerate() {
            if clip.audio != Audio::Source || beat.bed.is_some() {
                continue;
            }
            let Ok(Some((src_w, src_h))) = crate::framing::video_size(db, clip.video_id) else { continue };
            if crate::framing::needs_crop(src_w, src_h, out.0, out.1) {
                continue; // a vertical cut frames its screen shares already (framing_issues)
            }
            let framed = |a: f64, b: f64| {
                clip.framing.as_ref().is_some_and(|f| f.overrides.iter().any(|o| o.from_s <= a + 0.05 && o.to_s >= b - 0.05))
            };
            let Ok(Some(spans)) = crate::scenes::for_range(db, clip.video_id, clip.in_s, clip.out_s) else { continue };
            for s in spans.iter().filter(|s| s.scene == crate::scenes::Scene::ScreenShare && s.len() >= 1.5) {
                if framed(s.start, s.end) {
                    continue;
                }
                issues.push(Issue {
                    fixes: vec![crate::script::Fix::new(
                        format!("Frame the speaker's tile over {:.1}–{:.1}", s.start, s.end),
                        crate::script::FixAction::SetLayout {
                            beat_id: beat.id.clone(),
                            clip_index: clip_idx,
                            from_s: s.start,
                            to_s: s.end,
                            layout: crate::framing::Layout::FollowSpeaker,
                        },
                    )],
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(clip_idx),
                    message: format!(
                        "video #{}: {:.1} s of this line ({:.1}–{:.1} s) plays over a screen share, the speaker only a \
                         camera tile; if the screen is not the point, frame their own tile there (lips in sync, \
                         zoomed from a small tile)",
                        clip.video_id,
                        s.len(),
                        s.start,
                        s.end
                    ),
                });
            }
        }
    }

    // c2. whole thoughts: a clip that opens on a reply or a connective leaning on a line the viewer
    // never hears, or ends on a question whose answer is cut (`readcut::story`). Not when the line
    // before plays right before it in the cut (the previous clip runs up to it).
    let mut prev: Option<&crate::script::ScriptClip> = None;
    for beat in &script.beats {
        for (clip_idx, clip) in beat.clips.iter().enumerate() {
            let heard =
                beat.bed.is_none() && clip.audio == Audio::Source && !clip.tracks.as_ref().is_some_and(Vec::is_empty);
            if heard {
                let continues = prev.is_some_and(|p| {
                    p.video_id == clip.video_id && p.audio == Audio::Source && (clip.in_s - p.out_s).abs() < 3.0
                });
                for note in crate::readcut::story(db, clip.video_id, clip.in_s, clip.out_s) {
                    let opener = note.starts_with("opens on");
                    if !(opener || note.starts_with("ends on a question")) || (opener && continues) {
                        continue;
                    }
                    issues.push(Issue {
                        fixes: Vec::new(),
                        severity: IssueSeverity::Warning,
                        beat_id: Some(beat.id.clone()),
                        clip_index: Some(clip_idx),
                        message: format!("story: {note}"),
                    });
                }
            }
            prev = Some(clip);
        }
    }

    // d. cut edges. Where the speech stage has measured the video, the sound decides: a cut in
    // silence, a little after the last word and a guard before the next voice, is clean and says
    // nothing; a clipped word, a leak of the next speaker or a start inside a word is flagged, in
    // the words `read_cut` uses (`cuts::judge_range`). Without measurements the transcript's
    // times are all there is: a cut right on a segment's end gets the old hint.
    let fps = script.fps.map(|f| f.as_f64()).filter(|f| *f > 0.0).unwrap_or(25.0);
    for beat in &script.beats {
        for (clip_idx, clip) in beat.clips.iter().enumerate() {
            if clip.audio != Audio::Source || clip.tracks.as_ref().is_some_and(Vec::is_empty) {
                continue;
            }
            if let Some((start, end)) = crate::cuts::judge_range(
                db,
                clip.video_id,
                clip.in_s,
                clip.out_s,
                crate::sentences::WordEdges::of(db, clip.sentences.as_ref()),
                cfg,
                fps,
            ) {
                let fixes = || -> Vec<crate::script::Fix> {
                    crate::cuts::placed_range(db, clip.video_id, clip.in_s, clip.out_s, cfg, fps)
                        .map(|(in_s, out_s)| {
                            vec![crate::script::Fix::new(
                                format!("place the cut on the speech ({in_s:.2}–{out_s:.2} s)"),
                                crate::script::FixAction::Trim {
                                    beat_id: beat.id.clone(),
                                    clip_index: clip_idx,
                                    in_s,
                                    out_s,
                                },
                            )]
                        })
                        .unwrap_or_default()
                };
                if !end.flag.is_clean() {
                    issues.push(Issue {
                        fixes: fixes(),
                        severity: IssueSeverity::Warning,
                        beat_id: Some(beat.id.clone()),
                        clip_index: Some(clip_idx),
                        message: format!("end {:.2} s of video #{}: {}", clip.out_s, clip.video_id, end.detail()),
                    });
                }
                if !start.flag.is_clean() {
                    issues.push(Issue {
                        fixes: fixes(),
                        severity: IssueSeverity::Warning,
                        beat_id: Some(beat.id.clone()),
                        clip_index: Some(clip_idx),
                        message: format!("start {:.2} s of video #{}: {}", clip.in_s, clip.video_id, start.detail()),
                    });
                }
                // Dead air: a clean cut can still leave a long silence before the first word or after
                // the last one, which plays as a stall. Trimmed to the lead and tail the save path
                // would give.
                let trim = |in_s: f64, out_s: f64| {
                    vec![crate::script::Fix::new(
                        format!("Trim to {in_s:.2}–{out_s:.2}"),
                        crate::script::FixAction::Trim { beat_id: beat.id.clone(), clip_index: clip_idx, in_s, out_s },
                    )]
                };
                if start.flag.is_clean()
                    && let Some(b) = start.before_voice_s.filter(|b| *b > DEAD_AIR_S)
                {
                    let to = clip.in_s + b - cfg.cut_lead_s;
                    issues.push(Issue {
                        fixes: if clip.out_s - to >= 1.0 { trim(to, clip.out_s) } else { Vec::new() },
                        severity: IssueSeverity::Warning,
                        beat_id: Some(beat.id.clone()),
                        clip_index: Some(clip_idx),
                        message: format!(
                            "start {:.2} s of video #{}: {b:.2} s of silence before the first word plays as dead air",
                            clip.in_s, clip.video_id
                        ),
                    });
                }
                if end.flag.is_clean()
                    && let Some(a) = end.after_voice_s.filter(|a| *a > DEAD_AIR_S + cfg.speech_overrun_s)
                {
                    let to = clip.out_s - a + cfg.speech_overrun_s;
                    issues.push(Issue {
                        fixes: if to - clip.in_s >= 1.0 { trim(clip.in_s, to) } else { Vec::new() },
                        severity: IssueSeverity::Warning,
                        beat_id: Some(beat.id.clone()),
                        clip_index: Some(clip_idx),
                        message: format!(
                            "end {:.2} s of video #{}: {a:.2} s of silence after the last word plays as dead air",
                            clip.out_s, clip.video_id
                        ),
                    });
                }
                continue;
            }
            let edges =
                sentence_cache.entry(clip.video_id).or_insert_with(|| crate::chat::sentence_edges(db, clip.video_id));
            if edges.is_empty() {
                continue;
            }

            let last_seg_end = edges
                .iter()
                .filter(|(s_start, s_end)| *s_end > clip.in_s + 0.05 && *s_start < clip.out_s - 0.05)
                .filter(|(_s_start, s_end)| *s_end <= clip.out_s + 0.05)
                .map(|(_s_start, s_end)| *s_end)
                .last();

            if let Some(seg_end) = last_seg_end
                && clip.out_s >= seg_end - 0.05
                && clip.out_s <= seg_end + 0.2
            {
                issues.push(Issue {
                    fixes: Vec::new(),
                    severity: IssueSeverity::Info,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(clip_idx),
                    message: "cut lands on the last word; hold ~1 s".to_string(),
                });
            }
        }
    }

    // e. length: Warning per clip outside clip_min_s/clip_max_s when set.
    for beat in &script.beats {
        for (clip_idx, clip) in beat.clips.iter().enumerate() {
            let len = (clip.out_s - clip.in_s).max(0.0);
            if let Some(min_s) = opts.clip_min_s {
                if len < min_s {
                    issues.push(Issue {
                        fixes: Vec::new(),
                        severity: IssueSeverity::Warning,
                        beat_id: Some(beat.id.clone()),
                        clip_index: Some(clip_idx),
                        message: format!("clip duration {:.1}s is shorter than minimum {:.1}s", len, min_s),
                    });
                }
            }
            if let Some(max_s) = opts.clip_max_s {
                if len > max_s {
                    issues.push(Issue {
                        fixes: Vec::new(),
                        severity: IssueSeverity::Warning,
                        beat_id: Some(beat.id.clone()),
                        clip_index: Some(clip_idx),
                        message: format!("clip duration {:.1}s is longer than maximum {:.1}s", len, max_s),
                    });
                }
            }
        }
    }

    // f. overlap: when forbid_overlap, Error for two clips on the same video_id whose ranges overlap by more than 0.1 s.
    if opts.forbid_overlap {
        struct ClipRef<'a> {
            beat_id: &'a str,
            clip_idx: usize,
            clip: &'a ScriptClip,
        }
        let all_clips: Vec<ClipRef> = script
            .beats
            .iter()
            .flat_map(|b| {
                b.clips.iter().enumerate().map(move |(idx, c)| ClipRef { beat_id: &b.id, clip_idx: idx, clip: c })
            })
            .collect();

        for i in 0..all_clips.len() {
            for j in (i + 1)..all_clips.len() {
                let c1 = &all_clips[i];
                let c2 = &all_clips[j];
                if c1.clip.video_id == c2.clip.video_id {
                    let overlap = (c1.clip.out_s.min(c2.clip.out_s) - c1.clip.in_s.max(c2.clip.in_s)).max(0.0);
                    if overlap > 0.1 {
                        issues.push(Issue {
                            fixes: Vec::new(),
                            severity: IssueSeverity::Error,
                            beat_id: Some(c2.beat_id.to_string()),
                            clip_index: Some(c2.clip_idx),
                            message: format!(
                                "clips on video #{} overlap by {:.1}s ({:.1}–{:.1}s and {:.1}–{:.1}s)",
                                c1.clip.video_id, overlap, c1.clip.in_s, c1.clip.out_s, c2.clip.in_s, c2.clip.out_s
                            ),
                        });
                    }
                }
            }
        }
    }

    // g. framing: a framed clip should keep its face in the crop.
    issues.extend(framing_issues(db, project_id, script, cfg, opts.trim_min_s));

    Ok(issues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script::{Beat, Fps, ScriptClip};

    fn setup_test_db() -> (Db, i64, tempfile::TempDir) {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&crate::projects::NewProject::named("test_proj")).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let folder = db.add_folder(project.id, dir.path(), true).unwrap();
        let clip_path = dir.path().join("v1.mp4");
        std::fs::write(&clip_path, b"test").unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h1', 1, 300.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, ?2, 1, 0, 0)",
                rusqlite::params![folder.id, clip_path.to_str().unwrap()],
            )
            .unwrap();
        (db, project.id, dir)
    }

    /// A face track sampled at 5 fps over `a..b` (normalised centre and height).
    #[allow(clippy::too_many_arguments)]
    fn insert_track(db: &Db, id: i64, video: i64, a: f64, b: f64, cx: f64, cy: f64, h: f64, skip: Option<(f64, f64)>) {
        db.conn
            .execute(
                "INSERT INTO face_tracks(id, video_id, start_s, end_s, n_faces, mean_embedding, avg_box_h)
                 VALUES (?1, ?2, ?3, ?4, 0, x'', ?5)",
                rusqlite::params![id, video, a, b, h],
            )
            .unwrap();
        let w = h * 0.56;
        let n = ((b - a) * 5.0).round() as i64;
        for i in 0..n {
            let t = a + i as f64 * 0.2;
            if skip.is_some_and(|(s, e)| t >= s && t < e) {
                continue;
            }
            db.conn
                .execute(
                    "INSERT INTO face_boxes(track_id, t_s, x, y, w, h, score, landmarks_json, sharpness, cut_off)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0.9, '[]', 100.0, 0)",
                    rusqlite::params![id, t, cx - w / 2.0, cy - h / 2.0, w, h],
                )
                .unwrap();
        }
    }

    /// Video 1 as a podcast composite: two hosts side by side 0–40 s, a screen share with their
    /// camera tiles in the corner 40–60 s, a sentence every few seconds.
    fn podcast(db: &Db) {
        db.conn.execute("UPDATE videos SET width = 1920, height = 1080 WHERE id = 1", []).unwrap();
        insert_track(db, 1, 1, 0.0, 40.0, 0.25, 0.42, 0.45, None);
        insert_track(db, 2, 1, 0.0, 40.0, 0.75, 0.42, 0.45, None);
        insert_track(db, 3, 1, 40.0, 60.0, 0.72, 0.88, 0.076, None);
        insert_track(db, 4, 1, 40.0, 60.0, 0.90, 0.88, 0.07, None);
        // The speakers stage has run (one voice, not linked to a face).
        db.conn
            .execute("INSERT INTO speaker_turns(video_id, start_s, end_s, speaker) VALUES (1, 0.0, 60.0, 0)", [])
            .unwrap();
        for (a, b) in [(2.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 38.5), (38.6, 46.0), (46.0, 55.0)] {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, ?1, ?2, 'una frase.')",
                    [a, b],
                )
                .unwrap();
        }
    }

    fn vertical(clips: Vec<ScriptClip>) -> Script {
        let mut s = make_script(clips);
        s.width = Some(1080);
        s.height = Some(1920);
        s.framing = Some(crate::framing::Framing::default());
        s
    }

    fn speech(in_s: f64, out_s: f64) -> ScriptClip {
        ScriptClip { exact: false,
            sentences: None,
            video_id: 1,
            in_s,
            out_s,
            audio: Audio::Source,
            why: None,
            tracks: None,
            framing: None,
        }
    }

    fn hold(in_s: f64) -> ScriptClip {
        ScriptClip { exact: false,
            sentences: None,
            video_id: 1,
            in_s,
            out_s: in_s + 2.0,
            audio: Audio::Mute,
            why: Some(crate::chat::CLOSING_PICTURE.into()),
            tracks: None,
            framing: None,
        }
    }

    fn framing_msgs(db: &Db, pid: i64, s: &Script) -> Vec<Issue> {
        check(db, pid, s, &ScriptConfig::default(), &CheckOptions { trim_min_s: Some(20.0), ..Default::default() })
            .unwrap()
            .into_iter()
            .filter(|i| i.message.starts_with("framing"))
            .collect()
    }

    #[test]
    fn framed_clip_without_face_or_with_a_tiny_one() {
        let (db, pid, _dir) = setup_test_db();
        db.conn.execute("UPDATE videos SET width = 1920, height = 1080 WHERE id = 1", []).unwrap();
        // A tiny face (2.5% of the height, a detection on a pattern): nobody to frame.
        insert_track(&db, 1, 1, 10.0, 20.0, 0.5, 0.4, 0.025, None);
        let mut s = vertical(vec![speech(10.0, 20.0)]);
        let msgs: Vec<String> = framing_msgs(&db, pid, &s).into_iter().map(|i| i.message).collect();
        assert!(msgs.iter().any(|m| m.contains("no one in it") && m.contains("whole picture")), "{msgs:?}");
        // A real face lost for 3 s holds the crop and says so.
        insert_track(&db, 2, 1, 30.0, 40.0, 0.5, 0.4, 0.3, Some((33.0, 36.0)));
        s.beats[0].clips[0] = speech(30.0, 40.0);
        let msgs: Vec<String> = framing_msgs(&db, pid, &s).into_iter().map(|i| i.message).collect();
        assert!(msgs.iter().any(|m| m.contains("no face for 3.0 s from 32.8 s")), "{msgs:?}");
        // Unframed: nothing to say.
        s.width = Some(1920);
        s.height = Some(1080);
        assert!(framing_issues(&db, pid, &s, &ScriptConfig::default(), None).is_empty());
    }

    #[test]
    fn missing_faces_ask_for_the_stage_and_say_when_it_is_queued() {
        let (mut db, pid, _dir) = setup_test_db();
        db.conn.execute("UPDATE videos SET width = 1920, height = 1080 WHERE id = 1", []).unwrap();
        let s = vertical(vec![speech(10.0, 20.0)]);
        let mut issues = framing_msgs(&db, pid, &s);
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert!(issues[0].message.contains("aren't indexed yet"));
        assert!(matches!(
            &issues[0].fixes[0].action,
            crate::script::FixAction::RunStage { stage, enable: false } if stage == "faces"
        ));
        assert_eq!(stages_to_queue(&issues), vec!["faces".to_string()]);
        mark_queued(&mut issues, "faces");
        assert!(issues[0].message.ends_with("queued, the crop re-plans when it finishes"));
        // Turned off for the project: the fix turns it on, and nothing is queued on its own.
        let mut pipeline = db.project(pid).unwrap().pipeline;
        pipeline.set_enabled("faces", false).unwrap();
        db.update_project_pipeline(pid, &pipeline).unwrap();
        let issues = framing_msgs(&db, pid, &s);
        assert!(stages_to_queue(&issues).is_empty());
        let fix = issues[0].fixes[0].clone();
        assert!(matches!(&fix.action, crate::script::FixAction::RunStage { enable: true, .. }), "{fix:?}");
        let mut s2 = s.clone();
        let done = crate::script_fix::apply_fix(&mut db, pid, &mut s2, &fix.action).unwrap();
        assert_eq!(done.run_stage.as_deref(), Some("faces"));
        assert!(db.project(pid).unwrap().pipeline.is_enabled("faces"));
        assert_eq!(s2, s, "a stage fix leaves the script alone");
    }

    #[test]
    fn repair_trims_off_the_screen_share_on_sentences_and_keeps_the_hold_on_a_face() {
        let (db, _pid, _dir) = setup_test_db();
        podcast(&db);
        let mut s = vertical(vec![speech(2.0, 55.35), hold(55.35)]);
        let notes = crate::script_fix::repair_framing(&db, &mut s, 0.35, Some(20.0));
        let clips = &s.beats[0].clips;
        assert_eq!(
            (clips[0].in_s, clips[0].out_s),
            (2.0, 38.85),
            "ends on the last sentence before the share: {notes:?}"
        );
        // The hold followed the speech and was cut back to the faces (they end at 40 s).
        assert_eq!(clips.len(), 2);
        assert!((clips[1].in_s - 38.85).abs() < 1e-9);
        assert!((clips[1].out_s - 40.0).abs() < 0.45, "hold {:?}", clips[1]);
        assert!(notes.iter().any(|n| n.starts_with("trimmed video #1")), "{notes:?}");
        assert!(notes.iter().any(|n| n.contains("closing hold")), "{notes:?}");
        // Repairing again changes nothing.
        let again = s.clone();
        assert!(crate::script_fix::repair_framing(&db, &mut s, 0.35, Some(20.0)).is_empty());
        assert_eq!(s, again);
    }

    #[test]
    fn a_hold_with_no_face_left_is_dropped_and_a_too_short_trim_is_not_made() {
        let (db, _pid, _dir) = setup_test_db();
        podcast(&db);
        // Trimming to 38.85 s would leave 36.85 s, under the 50 s minimum: the speech stays, and
        // its hold, all screen share, goes.
        let mut s = vertical(vec![speech(2.0, 55.35), hold(55.35)]);
        let notes = crate::script_fix::repair_framing(&db, &mut s, 0.35, Some(50.0));
        assert_eq!(s.beats[0].clips.len(), 1, "{notes:?}");
        assert_eq!((s.beats[0].clips[0].in_s, s.beats[0].clips[0].out_s), (2.0, 55.35));
        assert_eq!(notes, vec!["dropped the closing hold: it would have shown a screen share, not a face".to_string()]);
    }

    #[test]
    fn issues_carry_fixes_that_apply_and_check_clean() {
        let (mut db, pid, _dir) = setup_test_db();
        podcast(&db);
        let s = vertical(vec![speech(2.0, 55.35)]);
        let issues = framing_msgs(&db, pid, &s);
        let edge = issues.iter().find(|i| i.message.contains("closes on a screen share")).expect("edge issue");
        assert_eq!(edge.severity, IssueSeverity::Warning);
        let trim = edge.fixes.first().unwrap();
        assert!(matches!(trim.action, crate::script::FixAction::Trim { .. }), "{trim:?}");
        assert!(trim.label.starts_with("Trim to 2.0–38.9"), "{}", trim.label);
        // Apply the trim: checked again, the clip is clean.
        let mut fixed = s.clone();
        crate::script_fix::apply_fix(&mut db, pid, &mut fixed, &trim.action).unwrap();
        assert_eq!(fixed.beats[0].clips[0].out_s, 38.85);
        let after = framing_msgs(&db, pid, &fixed);
        assert!(after.iter().all(|i| i.severity == IssueSeverity::Info), "{after:?}");
        // Or keep the share and frame it whole instead: the plan follows the override.
        let fit = edge
            .fixes
            .iter()
            .find(|f| {
                matches!(f.action, crate::script::FixAction::SetLayout { layout: crate::framing::Layout::Fit, .. })
            })
            .unwrap();
        let mut framed = s.clone();
        crate::script_fix::apply_fix(&mut db, pid, &mut framed, &fit.action).unwrap();
        let plan = &crate::framing::plans_for_script(&db, None, &framed).unwrap()[0].plan;
        let last = plan.segments.last().unwrap();
        assert_eq!((last.scene, last.layout), (crate::scenes::Scene::ScreenShare, crate::framing::Layout::Fit));
        assert!(framed.beats[0].clips[0].framing.as_ref().unwrap().overrides.len() == 1);
        // The editor's choice stands: no more word about that stretch, and repair leaves it.
        assert!(!framing_msgs(&db, pid, &framed).iter().any(|i| i.message.contains("screen share")));
        let before = framed.clone();
        assert!(crate::script_fix::repair_framing(&db, &mut framed, 0.35, Some(20.0)).is_empty());
        assert_eq!(framed, before);
    }

    /// A muted clip over someone talking, with nothing else to carry a voice, is flagged with a fix
    /// that plays its sound; under a bed it is how a cutaway works and nothing is said.
    #[test]
    fn a_muted_talking_head_is_flagged_unless_a_bed_carries_the_voice() {
        let (mut db, pid, _dir) = setup_test_db();
        podcast(&db);
        let mut muted = speech(2.0, 20.0);
        muted.audio = Audio::Mute;
        let s = make_script(vec![muted.clone()]);
        let silent = |s: &Script, db: &Db| -> Vec<Issue> {
            check(db, pid, s, &ScriptConfig::default(), &CheckOptions::default())
                .unwrap()
                .into_iter()
                .filter(|i| i.message.contains("is muted, but someone speaks"))
                .collect()
        };
        let issues = silent(&s, &db);
        assert_eq!(issues.len(), 1, "{issues:?}");
        let fix = &issues[0].fixes[0].action;
        let mut fixed = s.clone();
        crate::script_fix::apply_fix(&mut db, pid, &mut fixed, fix).unwrap();
        assert_eq!(fixed.beats[0].clips[0].audio, Audio::Source);
        assert!(silent(&fixed, &db).is_empty());

        let mut bedded = s.clone();
        bedded.beats[0].bed = Some(crate::script::AudioBed {
            sentences: None,
            video_id: 1,
            in_s: 2.0,
            out_s: 20.0,
            why: None,
            inferred: false,
            tracks: None,
        });
        assert!(silent(&bedded, &db).is_empty(), "a bed carries the voice");
    }

    #[test]
    fn clip_set_warnings_are_checked_live_not_stored() {
        let (db, pid, _dir) = setup_test_db();
        db.conn.execute("UPDATE videos SET width = 1920, height = 1080 WHERE id = 1", []).unwrap();
        let s = vertical(vec![speech(2.0, 30.0)]);
        let id = crate::script::save_version(&db, pid, &s, None).unwrap();
        db.conn.execute("UPDATE scripts SET set_id = 'cs1', kind = 'clip' WHERE id = ?1", [id]).unwrap();
        let item = serde_json::json!([{
            "script_id": id, "title": "t", "hook": "h", "quote": "", "why": "", "score": 8.0,
            "video_id": 1, "in_s": 2.0, "out_s": 30.0, "visual": {"label": "faces", "screen_share": 0.0, "frames": 0},
            "status": "new", "warnings": 0
        }]);
        db.conn
            .execute(
                "INSERT INTO clip_sets(id, project_id, brief, created_at, options_json, items_json, log_json)
                 VALUES ('cs1', ?1, '', 0, '{\"min_s\": 20.0}', ?2, '{}')",
                rusqlite::params![pid, item.to_string()],
            )
            .unwrap();
        let cfg = ScriptConfig::default();
        // Made before the faces were indexed: the stored count says 0, the index says otherwise.
        let mut set = crate::clips::load_set(&db, "cs1").unwrap();
        assert_eq!(set.items[0].warnings, 0);
        crate::clips::with_live_checks(&db, &cfg, &mut set);
        assert!(
            set.items[0].issues.iter().any(|i| i.message.contains("aren't indexed yet")),
            "{:?}",
            set.items[0].issues
        );
        assert_eq!(set.items[0].warnings, 1);
        // The faces arrive: the same set, shown again, has nothing stale to say.
        podcast(&db);
        let mut set = crate::clips::load_set(&db, "cs1").unwrap();
        crate::clips::with_live_checks(&db, &cfg, &mut set);
        assert!(!set.items[0].issues.iter().any(|i| i.message.contains("indexed")), "{:?}", set.items[0].issues);
        assert_eq!(set.items[0].warnings, 0, "{:?}", set.items[0].issues);
        // Nothing live was written back.
        let stored: String =
            db.conn.query_row("SELECT items_json FROM clip_sets WHERE id = 'cs1'", [], |r| r.get(0)).unwrap();
        assert!(!stored.contains("issues"));
    }

    fn make_script(clips: Vec<ScriptClip>) -> Script {
        Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "Test Script".to_string(),
            target_duration_s: None,
            fps: Some(Fps::new(25, 1)),
            width: Some(1920),
            height: Some(1080),
            beats: vec![Beat {
                id: "b1".to_string(),
                purpose: "test".to_string(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                clips,
                notes: None,
                bed: None,
                card: None,
            }],
            card_style: None,
        }
    }

    #[test]
    fn test_mid_sentence_warning_vs_edge_ok() {
        let (db, project_id, _dir) = setup_test_db();
        // One sentence from 10.0 to 20.0
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 10.0, 20.0, 'sentence 1')",
                [],
            )
            .unwrap();

        let cfg = ScriptConfig::default();
        let opts = CheckOptions::default();

        // 1. Clip starts mid-sentence (15.0 s), ends outside (25.0 s) -> Warning
        let script_mid = make_script(vec![ScriptClip { exact: false,
            sentences: None,
            framing: None,
            video_id: 1,
            in_s: 15.0,
            out_s: 25.0,
            audio: Audio::Source,
            why: None,
            tracks: None,
        }]);
        let issues = check(&db, project_id, &script_mid, &cfg, &opts).unwrap();
        let sentence_warns: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning && i.message.contains("inside a sentence"))
            .collect();
        assert_eq!(sentence_warns.len(), 1, "expected mid-sentence warning");
        assert!(sentence_warns[0].message.contains("video #1"));
        assert!(sentence_warns[0].message.contains("15.0"));
        assert!(sentence_warns[0].message.contains("10.0"));

        // 2. Clip starts within 0.1 s of sentence start (10.1 s) and ends on edge (20.0 s) -> OK
        let script_edge = make_script(vec![ScriptClip { exact: false,
            sentences: None,
            framing: None,
            video_id: 1,
            in_s: 10.1,
            out_s: 20.0,
            audio: Audio::Source,
            why: None,
            tracks: None,
        }]);
        let issues_edge = check(&db, project_id, &script_edge, &cfg, &opts).unwrap();
        assert!(
            !issues_edge.iter().any(|i| i.message.contains("inside a sentence")),
            "cut near edge should not generate sentence warning"
        );
    }

    #[test]
    fn test_clip_length_warnings() {
        let (db, project_id, _dir) = setup_test_db();
        let cfg = ScriptConfig::default();
        let opts = CheckOptions { clip_min_s: Some(2.0), clip_max_s: Some(5.0), ..Default::default() };

        let script = make_script(vec![
            // Too short: 1.0 s < 2.0 s
            ScriptClip { exact: false,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 0.0,
                out_s: 1.0,
                audio: Audio::Mute,
                why: None,
                tracks: None,
            },
            // Just right: 3.0 s
            ScriptClip { exact: false,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 2.0,
                out_s: 5.0,
                audio: Audio::Mute,
                why: None,
                tracks: None,
            },
            // Too long: 6.0 s > 5.0 s
            ScriptClip { exact: false,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 10.0,
                out_s: 16.0,
                audio: Audio::Mute,
                why: None,
                tracks: None,
            },
        ]);

        let issues = check(&db, project_id, &script, &cfg, &opts).unwrap();
        assert!(
            issues
                .iter()
                .any(|i| i.severity == IssueSeverity::Warning && i.message.contains("shorter than minimum 2.0s")),
            "should warn on short clip"
        );
        assert!(
            issues
                .iter()
                .any(|i| i.severity == IssueSeverity::Warning && i.message.contains("longer than maximum 5.0s")),
            "should warn on long clip"
        );
    }

    #[test]
    fn test_overlap_error() {
        let (db, project_id, _dir) = setup_test_db();
        let cfg = ScriptConfig::default();

        let script_overlap = make_script(vec![
            ScriptClip { exact: false,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 0.0,
                out_s: 5.0,
                audio: Audio::Mute,
                why: None,
                tracks: None,
            },
            ScriptClip { exact: false,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 4.0,
                out_s: 8.0,
                audio: Audio::Mute,
                why: None,
                tracks: None,
            },
        ]);

        // When forbid_overlap = false: no error
        let opts_allow = CheckOptions::default();
        let issues = check(&db, project_id, &script_overlap, &cfg, &opts_allow).unwrap();
        assert!(!issues.iter().any(|i| i.severity == IssueSeverity::Error && i.message.contains("overlap")));

        // When forbid_overlap = true: error because overlap is 1.0 s > 0.1 s
        let opts_forbid = CheckOptions { forbid_overlap: true, ..Default::default() };
        let issues = check(&db, project_id, &script_overlap, &cfg, &opts_forbid).unwrap();
        let overlap_errors: Vec<_> =
            issues.iter().filter(|i| i.severity == IssueSeverity::Error && i.message.contains("overlap")).collect();
        assert_eq!(overlap_errors.len(), 1, "expected 1 overlap error");
        assert!(overlap_errors[0].message.contains("overlap by 1.0s"));

        // Touching / tiny overlap <= 0.1 s: no error
        let script_touch = make_script(vec![
            ScriptClip { exact: false,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 0.0,
                out_s: 5.0,
                audio: Audio::Mute,
                why: None,
                tracks: None,
            },
            ScriptClip { exact: false,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 4.95,
                out_s: 8.0,
                audio: Audio::Mute,
                why: None,
                tracks: None,
            },
        ]);
        let issues_touch = check(&db, project_id, &script_touch, &cfg, &opts_forbid).unwrap();
        assert!(!issues_touch.iter().any(|i| i.severity == IssueSeverity::Error && i.message.contains("overlap")));
    }

    /// Measured speech decides the cut edges: the transcript's "cut lands on the last word" hint
    /// stands aside when the cut is in the silence after the word, and only a clipped word or a
    /// leak of the next speaker is flagged, in the words `read_cut` uses.
    #[test]
    fn measured_speech_decides_the_cut_edge_issues() {
        use crate::speech::SpeechRegion;
        let (db, project_id, _dir) = setup_test_db();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 10.0, 20.0, 'sentence 1')",
                [],
            )
            .unwrap();
        let cfg = ScriptConfig::default();
        let opts = CheckOptions::default();
        let clip = |out_s| {
            make_script(vec![ScriptClip { exact: false,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 9.85,
                out_s,
                audio: Audio::Source,
                why: None,
                tracks: None,
            }])
        };
        let edge_issues = |issues: Vec<Issue>| -> Vec<String> {
            issues
                .into_iter()
                .filter(|i| {
                    i.message.contains("last word") || i.message.contains("leaks") || i.message.contains("mid-word")
                })
                .map(|i| i.message)
                .collect()
        };

        // Not measured: the transcript's hint, as it always was.
        let hint = edge_issues(check(&db, project_id, &clip(20.1), &cfg, &opts).unwrap());
        assert_eq!(hint, vec!["cut lands on the last word; hold ~1 s".to_string()]);

        // Measured: the voice ends at 19.80 and nobody speaks until 25. The same cut, 0.3 s after
        // the last word, is clean: no hint, no issue.
        db.set_speech_regions(
            1,
            &[SpeechRegion { start_s: 10.0, end_s: 19.80, quiet_after_s: Some(22.0) }, SpeechRegion::new(25.0, 30.0)],
        )
        .unwrap();
        assert!(edge_issues(check(&db, project_id, &clip(20.1), &cfg, &opts).unwrap()).is_empty());
        // A cut on the segment's end that is still in the silence is clean too (the old hint fired).
        assert!(edge_issues(check(&db, project_id, &clip(20.0), &cfg, &opts).unwrap()).is_empty());

        // Inside the last word: flagged, with the fix.
        let issues = check(&db, project_id, &clip(19.6), &cfg, &opts).unwrap();
        let clipped: Vec<&Issue> = issues.iter().filter(|i| i.message.contains("clips the last word")).collect();
        assert_eq!(clipped.len(), 1, "{issues:?}");
        assert_eq!(clipped[0].severity, IssueSeverity::Warning);
        assert!(clipped[0].message.contains("0.20 s past the cut"), "{}", clipped[0].message);
        assert!(
            clipped[0]
                .fixes
                .iter()
                .any(|f| matches!(&f.action, crate::script::FixAction::Trim { out_s, .. } if *out_s > 19.8)),
            "the fix places the cut after the word: {:?}",
            clipped[0].fixes
        );

        // The next voice comes in at 20.05: a cut at 20.1 has it leaking in.
        db.set_speech_regions(
            1,
            &[SpeechRegion { start_s: 10.0, end_s: 19.80, quiet_after_s: Some(19.9) }, SpeechRegion::new(20.05, 30.0)],
        )
        .unwrap();
        let leaks = edge_issues(check(&db, project_id, &clip(20.1), &cfg, &opts).unwrap());
        assert_eq!(leaks.len(), 1, "{leaks:?}");
        assert!(leaks[0].contains("next speaker leaks in"), "{}", leaks[0]);
        // One that stays in the 0.25 s pause is fine, however short.
        assert!(edge_issues(check(&db, project_id, &clip(19.93), &cfg, &opts).unwrap()).is_empty());

        // A start inside the first word is flagged as such; one a little before it is not.
        let mut start_script = clip(19.9);
        start_script.beats[0].clips[0].in_s = 10.1;
        let mid = edge_issues(check(&db, project_id, &start_script, &cfg, &opts).unwrap());
        assert!(mid.iter().any(|m| m.contains("starts mid-word")), "{mid:?}");
        start_script.beats[0].clips[0].in_s = 9.85;
        assert!(
            !edge_issues(check(&db, project_id, &start_script, &cfg, &opts).unwrap())
                .iter()
                .any(|m| m.contains("mid-word"))
        );
    }

    #[test]
    fn test_muted_clip_gets_no_sentence_warning() {
        let (db, project_id, _dir) = setup_test_db();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 10.0, 20.0, 'sentence 1')",
                [],
            )
            .unwrap();

        let cfg = ScriptConfig::default();
        let opts = CheckOptions::default();

        let script = make_script(vec![ScriptClip { exact: false,
            sentences: None,
            framing: None,
            video_id: 1,
            in_s: 15.0,
            out_s: 25.0,
            audio: Audio::Mute,
            why: None,
            tracks: None,
        }]);

        let issues = check(&db, project_id, &script, &cfg, &opts).unwrap();
        assert!(
            !issues.iter().any(|i| i.message.contains("inside a sentence")),
            "muted clip should not produce sentence warning"
        );
    }

    #[test]
    fn test_breath_info_cut_lands_on_last_word() {
        let (db, project_id, _dir) = setup_test_db();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 10.0, 20.0, 'sentence 1')",
                [],
            )
            .unwrap();

        let cfg = ScriptConfig::default();
        let opts = CheckOptions::default();

        // 1. Cut lands 0.1 s after sentence end (20.1 s) -> Info issue
        let script_breath = make_script(vec![ScriptClip { exact: false,
            sentences: None,
            framing: None,
            video_id: 1,
            in_s: 10.0,
            out_s: 20.1,
            audio: Audio::Source,
            why: None,
            tracks: None,
        }]);
        let issues = check(&db, project_id, &script_breath, &cfg, &opts).unwrap();
        assert!(
            issues
                .iter()
                .any(|i| i.severity == IssueSeverity::Info && i.message == "cut lands on the last word; hold ~1 s"),
            "should emit breath Info issue when cut lands right on last word"
        );

        // 2. Cut held 1.0 s after sentence end (21.0 s) -> No Info issue
        let script_held = make_script(vec![ScriptClip { exact: false,
            sentences: None,
            framing: None,
            video_id: 1,
            in_s: 10.0,
            out_s: 21.0,
            audio: Audio::Source,
            why: None,
            tracks: None,
        }]);
        let issues_held = check(&db, project_id, &script_held, &cfg, &opts).unwrap();
        assert!(
            !issues_held.iter().any(|i| i.message == "cut lands on the last word; hold ~1 s"),
            "should not emit breath Info issue when clip holds enough after last word"
        );
    }
}
