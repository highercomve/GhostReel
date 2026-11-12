//! Script schema and editing operations (plan §4a, M8).

use rusqlite::{OptionalExtension, params};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::Error;
use crate::db::Db;
use crate::projects::Project;

/// Frame rate representation as a rational number (num / den).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Fps {
    pub num: i64,
    pub den: i64,
}

impl Fps {
    pub fn new(num: i64, den: i64) -> Self {
        Self { num, den }
    }

    pub fn as_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Rational rate mapping per plan/spec:
    /// values within 0.01 of 23.976/29.97/59.94/119.88 -> N*1000/1001, else round to integer/1.
    pub fn from_f64(rate: f64) -> Self {
        if (rate - 24000.0 / 1001.0).abs() < 0.01 || (rate - 23.976).abs() < 0.01 {
            Self { num: 24000, den: 1001 }
        } else if (rate - 30000.0 / 1001.0).abs() < 0.01 || (rate - 29.97).abs() < 0.01 {
            Self { num: 30000, den: 1001 }
        } else if (rate - 60000.0 / 1001.0).abs() < 0.01 || (rate - 59.94).abs() < 0.01 {
            Self { num: 60000, den: 1001 }
        } else if (rate - 120000.0 / 1001.0).abs() < 0.01 || (rate - 119.88).abs() < 0.01 {
            Self { num: 120000, den: 1001 }
        } else {
            let rounded = rate.round() as i64;
            let num = if rounded <= 0 { 25 } else { rounded };
            Self { num, den: 1 }
        }
    }
}

impl<'de> Deserialize<'de> for Fps {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FpsVisitor;

        impl<'de> Visitor<'de> for FpsVisitor {
            type Value = Fps;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a number (e.g. 25, 29.97) or an object with num and den")
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Fps { num: v, den: 1 })
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Fps { num: v as i64, den: 1 })
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Fps::from_f64(v))
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut num = None;
                let mut den = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "num" => num = Some(map.next_value()?),
                        "den" => den = Some(map.next_value()?),
                        _ => {
                            let _ = map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }
                let num = num.ok_or_else(|| de::Error::missing_field("num"))?;
                let den = den.ok_or_else(|| de::Error::missing_field("den"))?;
                Ok(Fps { num, den })
            }
        }

        deserializer.deserialize_any(FpsVisitor)
    }
}

/// How a chat builds its cuts, chosen per chat and kept for every turn in it.
///
/// The default is the interview-led documentary the pipeline was built around. `broll` is the
/// other kind of piece: a montage on a theme, no one speaking, no voice-over — every speech pass
/// (sentence endings, padding, audio beds, narration) stands aside for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ChatStyle {
    /// Pictures only: no interviews, no narration.
    pub broll: bool,
    /// With `broll`, play each clip's own ambient sound; otherwise every clip is muted, for music.
    pub natural_sound: bool,
}

/// Audio mode for a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Audio {
    #[default]
    Source,
    Mute,
}

/// A clip used in a script beat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptClip {
    pub video_id: i64,
    pub in_s: f64,
    pub out_s: f64,
    #[serde(default)]
    pub audio: Audio,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
    /// Source audio tracks to play (0-based among the file's audio streams). None = the file's
    /// automatic track (today's behaviour). Empty = silent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracks: Option<Vec<u32>>,
    /// How this clip is cut to the cut's shape when that differs from the footage's (a 9:16 clip
    /// from 16:9 footage). None = the cut's `framing`, else letterboxed as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub framing: Option<crate::framing::Framing>,
}

/// The sound that runs under a beat while its pictures play.
///
/// An interview answer does not have to stop when we cut away to what the speaker is describing;
/// in a cut edit that is the normal way round. A bed names the range whose audio carries the beat,
/// and the beat's own clips play silent under it — a J-cut, written down.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioBed {
    pub video_id: i64,
    pub in_s: f64,
    pub out_s: f64,
    /// Why this voice belongs under these pictures. The editor's own note, kept for the UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
    /// True when the pipeline laid this bed rather than the editor asking for it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub inferred: bool,
    /// Source audio tracks to play (0-based among the file's audio streams). None = the file's
    /// automatic track (today's behaviour). Empty = silent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracks: Option<Vec<u32>>,
}

impl AudioBed {
    pub fn duration_s(&self) -> f64 {
        (self.out_s - self.in_s).max(0.0)
    }
}

/// A story beat in a script.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Beat {
    pub id: String,
    pub purpose: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub narration: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_screen_text: Option<String>,
    /// When and where `on_screen_text` shows, and how it looks. None = the whole beat, at the
    /// default place, in the cut's title style.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title_layout: Option<TitleLayout>,
    #[serde(default)]
    pub clips: Vec<ScriptClip>,
    /// Sound that runs across the whole beat, from a clip we may never show.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bed: Option<AudioBed>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// How burnt-in text looks: titles and speech captions. Every field is optional; what is left out
/// falls back to the default for that kind of text (see `overlays::title_defaults` and
/// `overlays::caption_defaults`), so a style written as `{"size": 0.06}` changes only the size.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextStyle {
    /// Font family, e.g. "Arial". The live viewer and libass both look it up by name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font: Option<String>,
    /// Line height as a fraction of the output frame's height (libass font size / frame height).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bold: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uppercase: Option<bool>,
    /// Text colour, `#rrggbb`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Outline colour, `#rrggbb`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outline_color: Option<String>,
    /// Outline width as a fraction of the font size (0 = none).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outline: Option<f64>,
    /// Drop shadow offset as a fraction of the font size (0 = none).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow: Option<f64>,
    /// A box behind each line instead of an outline.
    #[serde(rename = "box", skip_serializing_if = "Option::is_none")]
    pub boxed: Option<bool>,
    /// Box colour, `#rrggbb`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub box_color: Option<String>,
    /// Box opacity, 0 (clear) to 1 (solid).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub box_opacity: Option<f64>,
    /// `left`, `center` or `right`. Also says what `x` points at: the text's left edge, its
    /// middle, or its right edge.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub align: Option<String>,
    /// Longest line as a fraction of the frame's width; longer text wraps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_width: Option<f64>,
}

impl TextStyle {
    /// `over`'s fields where it has them, else this style's.
    pub fn merged(&self, over: &TextStyle) -> TextStyle {
        TextStyle {
            font: over.font.clone().or_else(|| self.font.clone()),
            size: over.size.or(self.size),
            bold: over.bold.or(self.bold),
            uppercase: over.uppercase.or(self.uppercase),
            color: over.color.clone().or_else(|| self.color.clone()),
            outline_color: over.outline_color.clone().or_else(|| self.outline_color.clone()),
            outline: over.outline.or(self.outline),
            shadow: over.shadow.or(self.shadow),
            boxed: over.boxed.or(self.boxed),
            box_color: over.box_color.clone().or_else(|| self.box_color.clone()),
            box_opacity: over.box_opacity.or(self.box_opacity),
            align: over.align.clone().or_else(|| self.align.clone()),
            max_width: over.max_width.or(self.max_width),
        }
    }

    pub fn is_empty(&self) -> bool {
        *self == TextStyle::default()
    }
}

/// When, where and how a beat's on-screen title shows.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TitleLayout {
    /// Seconds after the beat starts. None = from the beat's start.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_s: Option<f64>,
    /// Seconds after the beat starts. None = until the beat ends.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_s: Option<f64>,
    /// Anchor in the output frame, 0..1 (see `TextStyle::align` for what x points at; y is the
    /// middle of the text). None = the default place.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
    /// This title's own style, over the cut's `title_style`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<TextStyle>,
}

/// Speech captions burnt into a cut's renders (what people say, from the transcript). Off when
/// a script has none — every script written before this existed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptionStyle {
    /// `bottom` or `center`: the default place when `y` is not set.
    pub position: String,
    /// Multiplier on the default size (1.0). `style.size`, when set, wins.
    pub size: f64,
    /// Look of every caption, over the caption defaults.
    #[serde(skip_serializing_if = "TextStyle::is_empty")]
    pub style: TextStyle,
    /// Anchor in the output frame, 0..1. x follows `style.align`; y is the bottom of the text, so
    /// a two-line caption grows upwards. None = the `position` preset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
    /// Longest caption line in characters when cues are built (default 42).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_chars: Option<usize>,
    /// Most lines per cue when cues are built (default 2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_lines: Option<usize>,
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self {
            position: "bottom".into(),
            size: 1.0,
            style: TextStyle::default(),
            x: None,
            y: None,
            max_chars: None,
            max_lines: None,
        }
    }
}

/// A hand edit of one speech caption. It is tied to the footage the cue was heard in (video and
/// source seconds), not to the cut's clock, so it follows the words when clips are moved, and is
/// dropped quietly when they are cut out.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CueEdit {
    pub video_id: i64,
    /// The generated cue's range in the source file — what identifies it.
    pub src_start_s: f64,
    pub src_end_s: f64,
    /// Replacement text (`\n` breaks lines).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// New start and end, in the source file's seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_s: Option<f64>,
    /// Leave this cue out.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hidden: bool,
}

/// Script schema v1 (plan §4a).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Script {
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_duration_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<Fps>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<i64>,
    /// Default framing for every clip (a clip's own `framing` wins).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub framing: Option<crate::framing::Framing>,
    #[serde(default)]
    pub beats: Vec<Beat>,
    /// Burn speech captions into previews and exports with this style. `None` = off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captions: Option<CaptionStyle>,
    /// Look of every on-screen title (a beat's `title_layout.style` wins).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title_style: Option<TextStyle>,
    /// Hand edits of speech captions: text, timing, hidden cues.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub caption_edits: Vec<CueEdit>,
    /// Which agents work on this cut (script chat turns, re-drafts). `None` = the app's choice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agents: Option<crate::agents::AgentSelection>,
}

impl Script {
    /// Fill missing fps, width, and height from the project sequence settings.
    pub fn fill_from_project(&mut self, project: &Project) {
        if self.fps.is_none() {
            self.fps = Some(Fps::new(project.fps_num, project.fps_den));
        }
        if self.width.is_none() {
            self.width = Some(project.width);
        }
        if self.height.is_none() {
            self.height = Some(project.height);
        }
    }

    /// Parse script JSON and fill missing fps, width, height from the project.
    pub fn parse_for_project(json: &str, project: &Project) -> Result<Self, serde_json::Error> {
        let mut script: Self = serde_json::from_str(json)?;
        script.fill_from_project(project);
        Ok(script)
    }

    /// Total duration of all clips across all beats.
    pub fn total_duration_s(&self) -> f64 {
        self.beats.iter().flat_map(|b| &b.clips).map(|c| (c.out_s - c.in_s).max(0.0)).sum()
    }

    /// Total number of clips across all beats.
    pub fn clip_count(&self) -> usize {
        self.beats.iter().map(|b| b.clips.len()).sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Issue {
    pub severity: IssueSeverity,
    pub beat_id: Option<String>,
    pub clip_index: Option<usize>,
    pub message: String,
    /// What would fix it, deterministically: the UI shows these as buttons, MCP's `apply_fix`
    /// and `ghostreel script fix` apply them. Applying one re-runs the checks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fixes: Vec<Fix>,
}

/// One way to fix an issue: a label for a button, and the action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fix {
    pub label: String,
    #[serde(flatten)]
    pub action: FixAction,
}

/// A deterministic change to a script (or to the project's index) that fixes an issue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FixAction {
    /// Move a clip's bounds (source seconds). A closing hold right after it moves along.
    Trim { beat_id: String, clip_index: usize, in_s: f64, out_s: f64 },
    /// Frame `from_s..to_s` of a clip (source seconds) with `layout`.
    SetLayout { beat_id: String, clip_index: usize, from_s: f64, to_s: f64, layout: crate::framing::Layout },
    /// Run an indexing stage for the project (turning it on first when `enable`).
    RunStage { stage: String, enable: bool },
}

impl Fix {
    pub fn new(label: impl Into<String>, action: FixAction) -> Self {
        Self { label: label.into(), action }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredScript {
    pub id: i64,
    pub project_id: i64,
    pub session_id: Option<i64>,
    pub title: String,
    pub version: i64,
    pub created_at: i64,
    pub script: Script,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptSummary {
    pub id: i64,
    pub title: String,
    pub version: i64,
    pub created_at: i64,
    pub beats: usize,
    pub clips: usize,
    pub duration_s: f64,
    pub session_id: Option<i64>,
    /// The clip set this script belongs to (`kind = "clip"`), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub set_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

/// Small tolerance for floating point boundary comparisons.
const EPSILON: f64 = 0.05;

/// Validate a script against project footage and boundaries.
pub fn validate(db: &Db, project_id: i64, script: &Script) -> Result<Vec<Issue>, Error> {
    let mut issues = Vec::new();

    if script.beats.is_empty() {
        issues.push(Issue {
            fixes: Vec::new(),
            severity: IssueSeverity::Error,
            beat_id: None,
            clip_index: None,
            message: "script has no beats".to_string(),
        });
    }

    let total_clips = script.clip_count();
    if total_clips == 0 {
        issues.push(Issue {
            fixes: Vec::new(),
            severity: IssueSeverity::Error,
            beat_id: None,
            clip_index: None,
            message: "script has no clips".to_string(),
        });
    }

    let mut seen_beat_ids = std::collections::HashSet::new();
    for beat in &script.beats {
        if !seen_beat_ids.insert(&beat.id) {
            issues.push(Issue {
                fixes: Vec::new(),
                severity: IssueSeverity::Warning,
                beat_id: Some(beat.id.clone()),
                clip_index: None,
                message: format!("duplicate beat id '{}'", beat.id),
            });
        }
    }

    for beat in &script.beats {
        for (clip_idx, clip) in beat.clips.iter().enumerate() {
            // Check if video exists and belongs to the project
            let video_row: Option<(Option<f64>, Option<i64>)> = db
                .conn
                .query_row(
                    "SELECT v.duration_s, v.vfr
                     FROM videos v
                     JOIN video_files vf ON vf.video_id = v.id
                     JOIN folders f ON f.id = vf.folder_id
                     JOIN project_folders pf ON pf.folder_id = f.id
                     WHERE pf.project_id = ?1 AND v.id = ?2
                       AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)
                     LIMIT 1",
                    params![project_id, clip.video_id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;

            match video_row {
                None => {
                    let exists: i64 =
                        db.conn
                            .query_row("SELECT count(*) FROM videos WHERE id = ?1", [clip.video_id], |r| r.get(0))?;
                    let msg = if exists > 0 {
                        format!("video #{} does not belong to project", clip.video_id)
                    } else {
                        format!("video #{} not found", clip.video_id)
                    };
                    issues.push(Issue {
                        fixes: Vec::new(),
                        severity: IssueSeverity::Error,
                        beat_id: Some(beat.id.clone()),
                        clip_index: Some(clip_idx),
                        message: msg,
                    });
                }
                Some((duration_s, vfr)) => {
                    if clip.in_s < -EPSILON {
                        issues.push(Issue {
                            fixes: Vec::new(),
                            severity: IssueSeverity::Error,
                            beat_id: Some(beat.id.clone()),
                            clip_index: Some(clip_idx),
                            message: format!("in_s ({:.2}s) is negative", clip.in_s),
                        });
                    }
                    if clip.in_s >= clip.out_s - EPSILON {
                        issues.push(Issue {
                            fixes: Vec::new(),
                            severity: IssueSeverity::Error,
                            beat_id: Some(beat.id.clone()),
                            clip_index: Some(clip_idx),
                            message: format!("in_s ({:.2}s) must be less than out_s ({:.2}s)", clip.in_s, clip.out_s),
                        });
                    }
                    if let Some(dur) = duration_s.filter(|&dur| clip.out_s > dur + EPSILON) {
                        issues.push(Issue {
                            fixes: Vec::new(),
                            severity: IssueSeverity::Error,
                            beat_id: Some(beat.id.clone()),
                            clip_index: Some(clip_idx),
                            message: format!("out_s ({:.2}s) exceeds video duration ({:.2}s)", clip.out_s, dur),
                        });
                    }
                    if vfr == Some(1) {
                        issues.push(Issue {
                            fixes: Vec::new(),
                            severity: IssueSeverity::Warning,
                            beat_id: Some(beat.id.clone()),
                            clip_index: Some(clip_idx),
                            message: format!(
                                "video #{} has variable frame rate (VFR); export timing may drift",
                                clip.video_id
                            ),
                        });
                    }
                }
            }
        }
    }

    let total_duration = script.total_duration_s();
    if let Some(target) = script.target_duration_s.filter(|&t| t > 0.0) {
        let diff = (total_duration - target).abs();
        let pct = (diff / target) * 100.0;
        if diff / target > 0.10 {
            issues.push(Issue {
                fixes: Vec::new(),
                severity: IssueSeverity::Warning,
                beat_id: None,
                clip_index: None,
                message: format!(
                    "total duration {:.1}s differs from target {:.1}s by {:.0}%",
                    total_duration, target, pct
                ),
            });
        } else {
            issues.push(Issue {
                fixes: Vec::new(),
                severity: IssueSeverity::Info,
                beat_id: None,
                clip_index: None,
                message: format!(
                    "total duration {:.1}s (target {:.1}s, difference {:.0}%)",
                    total_duration, target, pct
                ),
            });
        }
    }

    Ok(issues)
}

/// Snap clip in_s and out_s to nearest transcript segment boundaries within 0.75s.
/// Returns count of clips that were changed.
pub fn snap_to_segments(db: &Db, script: &mut Script) -> Result<usize, Error> {
    let mut clips_changed = 0;

    for beat in &mut script.beats {
        for clip in &mut beat.clips {
            let duration_s: Option<f64> = db
                .conn
                .query_row("SELECT duration_s FROM videos WHERE id = ?1", [clip.video_id], |r| r.get(0))
                .optional()?;
            let max_dur = duration_s.unwrap_or(f64::MAX);

            let mut st = db
                .conn
                .prepare("SELECT start_s, end_s FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")?;
            let rows = st.query_map([clip.video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?)))?;

            let mut boundaries: Vec<f64> = Vec::new();
            for r in rows {
                let (s, e) = r?;
                boundaries.push(s);
                boundaries.push(e);
            }

            if boundaries.is_empty() {
                continue;
            }

            let mut best_in = clip.in_s;
            let mut min_diff_in = 0.75;
            for &b in &boundaries {
                let diff = (b - clip.in_s).abs();
                if diff <= min_diff_in {
                    min_diff_in = diff;
                    best_in = b.clamp(0.0, max_dur);
                }
            }

            let mut best_out = clip.out_s;
            let mut min_diff_out = 0.75;
            for &b in &boundaries {
                let diff = (b - clip.out_s).abs();
                if diff <= min_diff_out {
                    min_diff_out = diff;
                    best_out = b.clamp(0.0, max_dur);
                }
            }

            // Never produce in >= out; skip snap if it would
            if best_in >= best_out {
                continue;
            }

            let changed = (best_in - clip.in_s).abs() > 1e-4 || (best_out - clip.out_s).abs() > 1e-4;
            if changed {
                clip.in_s = best_in;
                clip.out_s = best_out;
                clips_changed += 1;
            }
        }
    }

    Ok(clips_changed)
}

/// Save a new version of the script. Versions are numbered past trashed ones too, so a restore
/// never collides with what was saved meanwhile. A new version of a clip stays in its clip set
/// and keeps its kind, taken from the newest live version whose set is live (a deleted set does
/// not claim a new cut of the same title). The numbering, the insert and the inheritance are one
/// write transaction, so the app and the CLI cannot take the same number.
pub fn save_version(db: &Db, project_id: i64, script: &Script, session_id: Option<i64>) -> Result<i64, Error> {
    let script_json = serde_json::to_string_pretty(script).map_err(|e| Error::Invalid(e.to_string()))?;
    let now = crate::projects::now();
    let tx = rusqlite::Transaction::new_unchecked(&db.conn, rusqlite::TransactionBehavior::Immediate)?;
    let next_version: i64 = tx.query_row(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM scripts WHERE project_id = ?1 AND title = ?2",
        params![project_id, &script.title],
        |r| r.get(0),
    )?;
    let (set_id, kind): (Option<String>, Option<String>) = tx
        .query_row(
            "SELECT s.set_id, s.kind FROM scripts s LEFT JOIN clip_sets c ON c.id = s.set_id
             WHERE s.project_id = ?1 AND s.title = ?2 AND s.trash_id IS NULL
               AND (s.set_id IS NULL OR c.trash_id IS NULL)
             ORDER BY s.version DESC LIMIT 1",
            params![project_id, &script.title],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?
        .unwrap_or_default();
    tx.execute(
        "INSERT INTO scripts(project_id, session_id, title, version, script_json, created_at, set_id, kind)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![project_id, session_id, &script.title, next_version, script_json, now, set_id, kind],
    )?;
    let id = tx.last_insert_rowid();
    tx.commit()?;
    Ok(id)
}

/// Load a script by ID.
pub fn load(db: &Db, script_id: i64) -> Result<StoredScript, Error> {
    let row = db
        .conn
        .query_row(
            "SELECT id, project_id, session_id, title, version, script_json, created_at FROM scripts
             WHERE id = ?1 AND trash_id IS NULL",
            [script_id],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, i64>(6)?,
                ))
            },
        )
        .optional()?
        .ok_or_else(|| Error::NotFound(format!("script #{script_id}")))?;

    let script: Script =
        serde_json::from_str(&row.5).map_err(|e| Error::Invalid(format!("corrupt script JSON: {e}")))?;

    Ok(StoredScript {
        id: row.0,
        project_id: row.1,
        session_id: row.2,
        title: row.3,
        version: row.4,
        created_at: row.6,
        script,
    })
}

/// List all scripts for a project.
pub fn list(db: &Db, project_id: i64) -> Result<Vec<ScriptSummary>, Error> {
    let mut st = db.conn.prepare(
        "SELECT id, title, version, script_json, created_at, session_id, set_id, kind FROM scripts
         WHERE project_id = ?1 AND trash_id IS NULL ORDER BY id DESC",
    )?;
    let rows = st.query_map([project_id], |r| {
        Ok((
            (r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?, r.get::<_, String>(3)?),
            (r.get::<_, i64>(4)?, r.get::<_, Option<i64>>(5)?, r.get::<_, Option<String>>(6)?, r.get::<_, Option<String>>(7)?),
        ))
    })?;

    let mut result = Vec::new();
    for row in rows {
        let ((id, title, version, json_str), (created_at, session_id, set_id, kind)) = row?;
        let (beats, clips, duration_s) = if let Ok(s) = serde_json::from_str::<Script>(&json_str) {
            (s.beats.len(), s.clip_count(), s.total_duration_s())
        } else {
            (0, 0, 0.0)
        };
        result.push(ScriptSummary {
            id,
            title,
            version,
            created_at,
            beats,
            clips,
            duration_s,
            session_id,
            set_id,
            kind,
        });
    }
    Ok(result)
}

/// Delete a single script version: it goes to the trash (`trash.rs`), restorable for
/// `trash::KEEP_DAYS` days, and the delete is recorded as a run from `source`.
pub fn delete(db: &Db, script_id: i64, source: &str) -> Result<crate::trash::TrashItem, Error> {
    crate::trash::trash_script(db, script_id, source)
}

/// Delete every version of a cut (by title): one trash entry, restorable as a whole.
pub fn delete_cut(db: &Db, project_id: i64, title: &str, source: &str) -> Result<crate::trash::TrashItem, Error> {
    crate::trash::trash_cut(db, project_id, title, source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    /// Old scripts have no `tracks`: they load as None and save without the key; a choice,
    /// including "no tracks" (silent), survives the round trip.
    #[test]
    fn tracks_round_trip_and_absent_stays_absent() {
        let clip: ScriptClip = serde_json::from_str(r#"{"video_id":1,"in_s":0,"out_s":2}"#).unwrap();
        assert_eq!(clip.tracks, None);
        assert!(!serde_json::to_string(&clip).unwrap().contains("tracks"));
        let bed: AudioBed = serde_json::from_str(r#"{"video_id":1,"in_s":0,"out_s":2}"#).unwrap();
        assert_eq!(bed.tracks, None);
        assert!(!serde_json::to_string(&bed).unwrap().contains("tracks"));

        let clip: ScriptClip = serde_json::from_str(r#"{"video_id":1,"in_s":0,"out_s":2,"tracks":[0,1]}"#).unwrap();
        assert_eq!(clip.tracks, Some(vec![0, 1]));
        let back: ScriptClip = serde_json::from_str(&serde_json::to_string(&clip).unwrap()).unwrap();
        assert_eq!(back, clip);
        let bed: AudioBed = serde_json::from_str(r#"{"video_id":1,"in_s":0,"out_s":2,"tracks":[]}"#).unwrap();
        assert_eq!(bed.tracks, Some(vec![]));
        assert!(serde_json::to_string(&bed).unwrap().contains(r#""tracks":[]"#));
    }

    #[test]
    fn fps_deserialization_and_rational_rates() {
        // Plain integer
        let s: Script = serde_json::from_str(r#"{"title":"t","fps":25}"#).unwrap();
        assert_eq!(s.fps, Some(Fps::new(25, 1)));

        // Float 29.97 -> 30000/1001
        let s: Script = serde_json::from_str(r#"{"title":"t","fps":29.97}"#).unwrap();
        assert_eq!(s.fps, Some(Fps::new(30000, 1001)));

        // Float 23.976 -> 24000/1001
        let s: Script = serde_json::from_str(r#"{"title":"t","fps":23.976}"#).unwrap();
        assert_eq!(s.fps, Some(Fps::new(24000, 1001)));

        // Float 59.94 -> 60000/1001
        let s: Script = serde_json::from_str(r#"{"title":"t","fps":59.94}"#).unwrap();
        assert_eq!(s.fps, Some(Fps::new(60000, 1001)));

        // Integer 60 -> 60/1
        let s: Script = serde_json::from_str(r#"{"title":"t","fps":60}"#).unwrap();
        assert_eq!(s.fps, Some(Fps::new(60, 1)));

        // Object {"num": 30000, "den": 1001}
        let s: Script = serde_json::from_str(r#"{"title":"t","fps":{"num":30000,"den":1001}}"#).unwrap();
        assert_eq!(s.fps, Some(Fps::new(30000, 1001)));
    }

    #[test]
    fn script_crud_validate_snap_in_memory() {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("CM5")).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let folder = db.add_folder(project.id, dir.path(), true).unwrap();
        let clip_path = dir.path().join("intro.mp4");
        std::fs::write(&clip_path, b"test").unwrap();

        // Add video
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, vfr)
                 VALUES (10, 'h10', 5000, 20.0, 25.0, 0)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (10, ?1, ?2, 5000, 0, 0)",
                params![folder.id, clip_path.to_str().unwrap()],
            )
            .unwrap();

        // Add transcript segments: 0.0 - 5.2, 5.2 - 10.5
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
                 VALUES (10, 0.0, 5.2, 'intro speech'), (10, 5.2, 10.5, 'second sentence')",
                [],
            )
            .unwrap();

        let json = r#"{
            "title": "CM5 promo",
            "target_duration_s": 10.0,
            "beats": [
                {
                    "id": "b1",
                    "purpose": "hook",
                    "narration": "Hello world",
                    "on_screen_text": "CM5 Board",
                    "clips": [
                        {
                            "video_id": 10,
                            "in_s": 0.3,
                            "out_s": 5.4,
                            "audio": "source",
                            "why": "nice shot"
                        }
                    ]
                }
            ]
        }"#;

        let mut script = Script::parse_for_project(json, &project).unwrap();
        assert_eq!(script.fps, Some(Fps::new(25, 1)));
        assert_eq!(script.width, Some(1920));
        assert_eq!(script.height, Some(1080));

        // Snap to segments: 0.3 is within 0.75 of 0.0 (snaps to 0.0), 5.4 is within 0.75 of 5.2 (snaps to 5.2)
        let changed = snap_to_segments(&db, &mut script).unwrap();
        assert_eq!(changed, 1);
        assert!((script.beats[0].clips[0].in_s - 0.0).abs() < 1e-4);
        assert!((script.beats[0].clips[0].out_s - 5.2).abs() < 1e-4);

        // Validate
        let issues = validate(&db, project.id, &script).unwrap();
        assert!(issues.iter().all(|i| i.severity != IssueSeverity::Error));

        // Save versions
        let id1 = save_version(&db, project.id, &script, None).unwrap();
        let id2 = save_version(&db, project.id, &script, None).unwrap();

        let s1 = load(&db, id1).unwrap();
        assert_eq!(s1.version, 1);
        let s2 = load(&db, id2).unwrap();
        assert_eq!(s2.version, 2);

        let list = list(&db, project.id).unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, id2);
        assert_eq!(list[0].version, 2);
        assert_eq!(list[0].clips, 1);
    }

    #[test]
    fn validate_issues_and_edge_cases() {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("P1")).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let folder = db.add_folder(project.id, dir.path(), true).unwrap();
        let clip_path = dir.path().join("v1.mp4");
        std::fs::write(&clip_path, b"test").unwrap();

        // Add video with vfr=1, duration 10.0
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, vfr)
                 VALUES (1, 'h1', 100, 10.0, 25.0, 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (1, ?1, ?2, 100, 0, 0)",
                params![folder.id, clip_path.to_str().unwrap()],
            )
            .unwrap();

        // 1. Empty beats
        let empty_script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "Empty".into(),
            target_duration_s: None,
            fps: None,
            width: None,
            height: None,
            beats: vec![],
        };
        let issues = validate(&db, project.id, &empty_script).unwrap();
        assert!(issues.iter().any(|i| i.severity == IssueSeverity::Error && i.message.contains("no beats")));

        // 2. Duplicate beat IDs, invalid ranges, VFR warning, missing video
        let bad_script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "Bad".into(),
            target_duration_s: Some(20.0), // actual will be ~15s, diff > 10%
            fps: Some(Fps::new(25, 1)),
            width: Some(1920),
            height: Some(1080),
            beats: vec![
                Beat {
                    id: "b1".into(),
                    purpose: "hook".into(),
                    narration: None,
                    on_screen_text: None,
                    title_layout: None,
                    clips: vec![
                        ScriptClip {
                            framing: None,
                            video_id: 1,
                            in_s: 5.0,
                            out_s: 4.0, // in >= out: Error
                            audio: Audio::Source,
                            why: None,
                            tracks: None,
                        },
                        ScriptClip {
                            framing: None,
                            video_id: 999, // not found: Error
                            in_s: 0.0,
                            out_s: 5.0,
                            audio: Audio::Source,
                            why: None,
                            tracks: None,
                        },
                    ],
                    notes: None,
                    bed: None,
                },
                Beat {
                    id: "b1".into(), // duplicate beat id: Warning
                    purpose: "hook 2".into(),
                    narration: None,
                    on_screen_text: None,
                    title_layout: None,
                    clips: vec![ScriptClip {
                        framing: None,
                        video_id: 1,
                        in_s: 0.0,
                        out_s: 15.0, // exceeds duration (10s): Error
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    }],
                    notes: None,
                    bed: None,
                },
            ],
        };

        let issues = validate(&db, project.id, &bad_script).unwrap();
        assert!(issues.iter().any(|i| i.severity == IssueSeverity::Warning && i.message.contains("duplicate beat id")));
        assert!(
            issues.iter().any(|i| i.severity == IssueSeverity::Error && i.message.contains("must be less than out_s"))
        );
        assert!(
            issues.iter().any(|i| i.severity == IssueSeverity::Error && i.message.contains("video #999 not found"))
        );
        assert!(
            issues.iter().any(|i| i.severity == IssueSeverity::Error && i.message.contains("exceeds video duration"))
        );
        assert!(issues.iter().any(|i| i.severity == IssueSeverity::Warning && i.message.contains("VFR")));
    }

    #[test]
    fn multi_script_cut_versioning_and_deletion() {
        let db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("Multi-Script Demo")).unwrap();

        // 1. Cut A: Social Reel (sessionless starter)
        let cut_a = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "Social 30s Promo".into(),
            target_duration_s: Some(30.0),
            beats: vec![],
            fps: None,
            width: None,
            height: None,
        };
        let a1 = save_version(&db, project.id, &cut_a, None).unwrap();
        let s_a1 = load(&db, a1).unwrap();
        assert_eq!(s_a1.version, 1);
        assert_eq!(s_a1.session_id, None);

        let a2 = save_version(&db, project.id, &cut_a, None).unwrap();
        let s_a2 = load(&db, a2).unwrap();
        assert_eq!(s_a2.version, 2);

        // 2. Cut B: Documentary Cut
        let cut_b = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "Documentary 15m".into(),
            target_duration_s: Some(900.0),
            beats: vec![],
            fps: None,
            width: None,
            height: None,
        };
        let b1 = save_version(&db, project.id, &cut_b, None).unwrap();
        let s_b1 = load(&db, b1).unwrap();
        assert_eq!(s_b1.version, 1);

        let scripts = list(&db, project.id).unwrap();
        assert_eq!(scripts.len(), 3);

        // 3. Delete single version of Cut A (version 1)
        let deleted_a1 = delete(&db, a1, "app").unwrap();
        assert_eq!(deleted_a1.kind, crate::trash::TrashKind::ScriptVersion);
        assert!(load(&db, a1).is_err());
        assert!(load(&db, a2).is_ok());

        // 4. Delete entire cut B
        let deleted_b = delete_cut(&db, project.id, "Documentary 15m", "app").unwrap();
        assert_eq!(deleted_b.scripts, 1);
        assert!(load(&db, b1).is_err());

        let remaining = list(&db, project.id).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].title, "Social 30s Promo");
        assert_eq!(remaining[0].version, 2);
    }

    fn titled(title: &str) -> Script {
        serde_json::from_value(serde_json::json!({ "title": title, "beats": [] })).unwrap()
    }

    fn set_of(db: &Db, id: i64) -> (Option<String>, Option<String>) {
        db.conn.query_row("SELECT set_id, kind FROM scripts WHERE id = ?1", [id], |r| Ok((r.get(0)?, r.get(1)?))).unwrap()
    }

    #[test]
    fn a_new_version_joins_its_live_clip_set_but_not_a_deleted_one() {
        let db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("Clips")).unwrap();
        db.conn
            .execute("INSERT INTO clip_sets(id, project_id, brief, created_at) VALUES ('cs1', ?1, 'Launch', 5)", [project.id])
            .unwrap();
        let v1 = save_version(&db, project.id, &titled("Clip A"), None).unwrap();
        db.conn.execute("UPDATE scripts SET set_id = 'cs1', kind = 'clip' WHERE id = ?1", [v1]).unwrap();

        // An edit of a clip in a live set stays in the set.
        let v2 = save_version(&db, project.id, &titled("Clip A"), None).unwrap();
        assert_eq!(set_of(&db, v2), (Some("cs1".into()), Some("clip".into())));

        // Delete the set, then save a cut with the same title: a plain cut, listed, numbered past.
        crate::trash::trash_clip_set(&db, "cs1", "app").unwrap();
        let cut = save_version(&db, project.id, &titled("Clip A"), None).unwrap();
        assert_eq!(set_of(&db, cut), (None, None));
        assert_eq!(load(&db, cut).unwrap().version, 3);
        let listed = list(&db, project.id).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!((listed[0].id, listed[0].set_id.as_deref()), (cut, None));

        // Restoring the set brings back its clips only, not the cut saved meanwhile.
        let entry = crate::trash::list(&db, Some(project.id)).unwrap().remove(0);
        crate::trash::restore(&db, &entry.id, "app").unwrap();
        assert_eq!(set_of(&db, cut), (None, None));
        assert_eq!(set_of(&db, v2).0.as_deref(), Some("cs1"));
    }

    #[test]
    fn concurrent_saves_of_one_title_never_share_a_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ghostreel.db");
        let project = Db::open(&path).unwrap().create_project(&NewProject::named("Race")).unwrap();
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || {
                    let db = Db::open(&path).unwrap();
                    (0..8).map(|_| save_version(&db, project.id, &titled("Same"), None).unwrap()).count()
                })
            })
            .collect();
        let saved: usize = workers.into_iter().map(|w| w.join().unwrap()).sum();
        assert_eq!(saved, 32);
        let db = Db::open(&path).unwrap();
        let mut versions: Vec<i64> = list(&db, project.id).unwrap().into_iter().map(|s| s.version).collect();
        versions.sort();
        assert_eq!(versions, (1..=32).collect::<Vec<_>>());
    }

    #[test]
    fn scripts_without_text_layout_parse_and_save_unchanged() {
        let old = r#"{"title":"Clip","fps":{"num":25,"den":1},"width":1080,"height":1920,"framing":{"layout":"follow_speaker"},"beats":[{"id":"b1","purpose":"p","on_screen_text":"Hola","clips":[{"video_id":1,"in_s":1.0,"out_s":2.0,"audio":"source"}]}],"captions":{"position":"bottom","size":1.0}}"#;
        let s: Script = serde_json::from_str(old).unwrap();
        assert!(s.title_style.is_none() && s.caption_edits.is_empty());
        assert!(s.beats[0].title_layout.is_none());
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back, serde_json::from_str::<serde_json::Value>(old).unwrap());
    }

    #[test]
    fn text_layout_round_trips() {
        let json = r##"{"title":"T","beats":[{"id":"b1","purpose":"p","on_screen_text":"Hola",
            "title_layout":{"start_s":1.0,"end_s":3.5,"x":0.1,"y":0.75,"style":{"align":"left","box":false,"size":0.05}},
            "clips":[]}],
            "captions":{"position":"bottom","size":1.0,"style":{"color":"#ffcc00","bold":true},"x":0.5,"y":0.3,"max_chars":30,"max_lines":1},
            "title_style":{"font":"Georgia","uppercase":true},
            "caption_edits":[{"video_id":1,"src_start_s":3.0,"src_end_s":5.0,"text":"Fixed","start_s":3.2},{"video_id":1,"src_start_s":6.0,"src_end_s":7.0,"hidden":true}]}"##;
        let s: Script = serde_json::from_str(json).unwrap();
        let lay = s.beats[0].title_layout.as_ref().unwrap();
        assert_eq!((lay.start_s, lay.end_s, lay.x, lay.y), (Some(1.0), Some(3.5), Some(0.1), Some(0.75)));
        assert_eq!(lay.style.as_ref().unwrap().boxed, Some(false));
        let cap = s.captions.as_ref().unwrap();
        assert_eq!((cap.y, cap.max_chars, cap.max_lines), (Some(0.3), Some(30), Some(1)));
        assert_eq!(cap.style.color.as_deref(), Some("#ffcc00"));
        assert_eq!(s.title_style.as_ref().unwrap().font.as_deref(), Some("Georgia"));
        assert_eq!(s.caption_edits.len(), 2);
        assert!(s.caption_edits[1].hidden && s.caption_edits[1].text.is_none());
        let again: Script = serde_json::from_value(serde_json::to_value(&s).unwrap()).unwrap();
        assert_eq!(again, s);
        // `box` keeps its JSON name.
        assert!(serde_json::to_string(&s).unwrap().contains(r#""box":false"#));
    }

    #[test]
    fn a_style_override_only_changes_what_it_names() {
        let base = TextStyle { font: Some("Arial".into()), size: Some(0.05), bold: Some(true), ..TextStyle::default() };
        let m = base.merged(&TextStyle { size: Some(0.08), ..TextStyle::default() });
        assert_eq!((m.font.as_deref(), m.size, m.bold), (Some("Arial"), Some(0.08), Some(true)));
    }
}
