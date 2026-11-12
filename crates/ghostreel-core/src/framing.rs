//! Vertical framing for cuts (plan Milestone 3): a crop that follows the speaker, or two crops
//! stacked (speaker on top, listener below), cut from a wider picture.
//!
//! One crop path drives both the ffmpeg render (`ffmpeg_filter`) and the live player (which
//! applies the same keyframes as a CSS crop), so what plays live is what renders.
//!
//! The path is planned from the face boxes found at indexing (about 5 per second) and, when the
//! speakers stage has run, the speaker turns. Without turns it holds on the most prominent face
//! rather than guessing; without faces it holds the centre.
//!
//! A clip is first cut into stretches by what the picture is ([`crate::scenes`]): a two-shot or a
//! single face is framed with the layout asked for, a screen share shows the screen on top and
//! the speaker's camera tile below, and a stretch with nobody in it is fitted whole over a blurred
//! copy of itself. Each stretch has its own crop size; they cut hard into each other.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;
use crate::scenes::{FaceObs, Scene, SceneSpan};
use crate::script::{Script, ScriptClip};

/// Face height as a share of the output height (follow the speaker, fixed).
pub const FACE_FRAC: f64 = 0.24;
/// The same for each half of a stacked layout: a half is a smaller picture, so the face fills
/// more of it (about the same share of the whole screen).
pub const FACE_FRAC_STACKED: f64 = 0.40;
/// Where the eyes sit, as a share of the crop height from its top: about the upper third.
pub const EYE_LINE: f64 = 0.36;
pub const EYE_LINE_STACKED: f64 = 0.40;
/// The smallest crop, as a share of the source height. Past this the upscale is mush.
pub const MIN_CROP_FRAC: f64 = 0.30;
/// A crop shrinks to stay inside one person's tile only while their face stays under this
/// share of it; past that, showing a sliver of the neighbour is the lesser evil.
pub const MAX_FACE_FRAC: f64 = 0.60;
/// The face may drift this far from where the crop put it (share of the crop) before it moves.
pub const DEAD_ZONE_X: f64 = 0.10;
pub const DEAD_ZONE_Y: f64 = 0.08;
/// Eased moves: speed in crop widths per second, bounded in time.
pub const EASE_SPEED: f64 = 0.6;
pub const EASE_MIN_S: f64 = 0.5;
pub const EASE_MAX_S: f64 = 1.2;
/// Shortest shot: turns shorter than this ("sí", "claro") do not cut, and no cut comes sooner
/// than this after the last one or this close to the end.
pub const MIN_SHOT_S: f64 = 1.5;
/// Turns of one speaker separated by less than this are one turn.
pub const TURN_MERGE_GAP_S: f64 = 0.8;
/// Face positions are smoothed with a running median over this many seconds either side.
pub const SMOOTH_HALF_S: f64 = 0.5;
/// Unlabelled face tracks closer than this (normalised centre distance) and not on screen at the
/// same time are one "slot": the same seat in the composite.
pub const SLOT_GAP: f64 = 0.12;
/// Faces smaller than this share of the largest one are not people we frame (faces on a shared
/// screen, a poster in the background).
pub const MIN_SUBJECT_SIZE: f64 = 0.5;
/// Faces on screen for less than this share of the clip (or less than half as long as the
/// steadiest face) are passers-by or pictures on a shared screen.
pub const MIN_PRESENCE: f64 = 0.15;
/// Face detection rate at indexing.
pub const SAMPLE_HZ: f64 = 5.0;
/// A crop is at least this many face widths wide, so the face is never cut at the sides.
pub const FACE_MARGIN_W: f64 = 1.3;

/// How a clip is framed when the cut's shape differs from the footage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// One crop on whoever is talking; hard cut when the other person answers.
    #[default]
    FollowSpeaker,
    /// Two crops, one above the other: the speaker on top, the listener below.
    Stacked,
    /// One crop that does not move: on `box`, on `person_id`, or on the most prominent face.
    Fixed,
    /// For a screen share: the screen on top, the speaker's camera tile below.
    Screen,
    /// The whole picture, fitted to the width over a blurred copy of itself.
    Fit,
}

impl Layout {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FollowSpeaker => "follow_speaker",
            Self::Stacked => "stacked",
            Self::Fixed => "fixed",
            Self::Screen => "screen",
            Self::Fit => "fit",
        }
    }

    /// Words for people: "follow the speaker", "screen over speaker".
    pub fn label(&self) -> &'static str {
        match self {
            Self::FollowSpeaker => "follow the speaker",
            Self::Stacked => "speaker over listener",
            Self::Fixed => "fixed crop",
            Self::Screen => "screen over speaker",
            Self::Fit => "whole picture over a blur",
        }
    }
}

impl std::str::FromStr for Layout {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "follow_speaker" | "follow" | "speaker" => Ok(Self::FollowSpeaker),
            "stacked" | "stack" => Ok(Self::Stacked),
            "fixed" | "static" => Ok(Self::Fixed),
            "screen" | "screen_share" | "screen_over_speaker" => Ok(Self::Screen),
            "fit" | "letterbox" | "blur" => Ok(Self::Fit),
            other => {
                Err(Error::Invalid(format!("unknown layout '{other}' (follow_speaker, stacked, fixed, screen or fit)")))
            }
        }
    }
}

/// A rectangle in normalised source coordinates (0..1 of width and height).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Framing for a clip (`ScriptClip::framing`) or the whole cut (`Script::framing`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Framing {
    #[serde(default)]
    pub layout: Layout,
    /// Follow (or hold on) this person rather than whoever is speaking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub person_id: Option<i64>,
    /// For `fixed`: the part of the picture to show.
    #[serde(rename = "box", default, skip_serializing_if = "Option::is_none")]
    pub box_: Option<NormBox>,
    /// Stretches framed with another layout than the one their picture would get (source
    /// seconds): an editor's choice, or a fix a check offered.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<LayoutOverride>,
}

/// One stretch of a clip framed with a chosen layout, in source seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LayoutOverride {
    pub from_s: f64,
    pub to_s: f64,
    pub layout: Layout,
}

impl Framing {
    /// This framing with `layout` over `from_s..to_s` (source seconds), replacing whatever
    /// overrides that stretch had.
    pub fn with_override(&self, from_s: f64, to_s: f64, layout: Layout) -> Framing {
        let mut f = self.clone();
        let mut kept = Vec::new();
        for o in f.overrides.drain(..) {
            // Keep the parts of older overrides outside the new stretch.
            if o.from_s < from_s {
                kept.push(LayoutOverride { from_s: o.from_s, to_s: o.to_s.min(from_s), layout: o.layout });
            }
            if o.to_s > to_s {
                kept.push(LayoutOverride { from_s: o.from_s.max(to_s), to_s: o.to_s, layout: o.layout });
            }
        }
        kept.push(LayoutOverride { from_s, to_s, layout });
        kept.retain(|o| o.to_s - o.from_s > 0.05);
        kept.sort_by(|a, b| a.from_s.total_cmp(&b.from_s));
        f.overrides = kept;
        f
    }
}

/// The framing a clip plays with: its own, else the cut's.
pub fn effective<'a>(script: &'a Script, clip: &'a ScriptClip) -> Option<&'a Framing> {
    clip.framing.as_ref().or(script.framing.as_ref())
}

/// Whether the cut's shape differs enough from the footage's that it needs a crop.
pub fn needs_crop(src_w: f64, src_h: f64, out_w: f64, out_h: f64) -> bool {
    if src_w <= 0.0 || src_h <= 0.0 || out_w <= 0.0 || out_h <= 0.0 {
        return false;
    }
    ((src_w / src_h) / (out_w / out_h) - 1.0).abs() > 0.02
}

/// A rectangle in source pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    fn right(&self) -> f64 {
        self.x + self.w
    }
    fn bottom(&self) -> f64 {
        self.y + self.h
    }
    fn union(&self, o: &Rect) -> Rect {
        let x = self.x.min(o.x);
        let y = self.y.min(o.y);
        Rect { x, y, w: self.right().max(o.right()) - x, h: self.bottom().max(o.bottom()) - y }
    }
}

/// One detected face, normalised to the source frame; `t` is seconds from the clip's start.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct FaceSample {
    pub t: f64,
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    /// Height of the eye line (mean of the two eye landmarks).
    pub eye_y: f64,
    pub cut_off: bool,
}

/// A person on screen: by `person_id` when the people stage named them, else a "slot" (a seat
/// in the composite, found from where their face tracks sit).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Subject {
    pub person_id: Option<i64>,
    pub samples: Vec<FaceSample>,
}

impl Subject {
    fn med(&self, f: impl Fn(&FaceSample) -> f64) -> f64 {
        median(self.samples.iter().map(f).collect())
    }
    pub fn median_h(&self) -> f64 {
        self.med(|s| s.h)
    }
    pub fn median_cx(&self) -> f64 {
        self.med(|s| s.cx)
    }
    pub fn median_cy(&self) -> f64 {
        self.med(|s| s.cy)
    }
    pub fn presence(&self, duration: f64) -> f64 {
        (self.samples.len() as f64 / (duration * SAMPLE_HZ).max(1.0)).min(1.0)
    }
    fn samples_in(&self, a: f64, b: f64) -> impl Iterator<Item = &FaceSample> {
        self.samples.iter().filter(move |s| s.t >= a && s.t < b)
    }
}

/// Someone talking, in seconds from the clip's start; `subject` indexes `FramingInput::subjects`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub subject: usize,
}

/// Everything the planner needs about one clip.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FramingInput {
    pub src_w: f64,
    pub src_h: f64,
    pub duration: f64,
    pub subjects: Vec<Subject>,
    pub turns: Vec<Turn>,
    /// The part of the frame with picture in it (black bars excluded).
    pub content: Rect,
    /// The source second of `t = 0` (layout overrides are in source seconds).
    pub origin_s: f64,
}

/// A point of the crop path: the crop's top-left corner in source pixels, `t` seconds into the
/// clip. Between keyframes the crop moves in a straight line, unless the later one is a `cut`,
/// where it jumps.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cut: bool,
}

/// A stretch of the clip held on one subject.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Shot {
    pub start: f64,
    pub end: f64,
    /// Index into the plan's subjects; None = no one (centre).
    pub subject: Option<usize>,
}

/// One crop moving through the clip: a fixed size, a moving corner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CropPath {
    pub w: f64,
    pub h: f64,
    pub keyframes: Vec<Keyframe>,
    pub shots: Vec<Shot>,
}

impl CropPath {
    /// The crop's corner at `t` seconds into the clip.
    pub fn at(&self, t: f64) -> (f64, f64) {
        value_at(&self.keyframes, t)
    }
}

/// How well the faces fit the crop, for checks and the harness.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Quality {
    /// Longest stretch the framed face is not seen, and where it starts (clip seconds).
    pub longest_no_face_s: f64,
    pub no_face_at: Option<f64>,
    /// Share of the framed face's detections that are cut off by the frame or the crop.
    pub cut_off_share: f64,
    /// Median face height as a share of the crop height.
    pub face_frac: f64,
}

/// Summary of a subject for the harness.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SubjectSummary {
    pub index: usize,
    pub person_id: Option<i64>,
    pub median_cx: f64,
    pub median_cy: f64,
    pub median_h: f64,
    pub presence: f64,
}

/// One stretch of a clip with its own layout and crop size. Stretches cut hard into each other.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanSegment {
    /// Clip seconds.
    pub start: f64,
    pub end: f64,
    /// What the picture is here.
    pub scene: Scene,
    /// How it is framed (stacked falls back to follow with one person on screen; screen falls
    /// back to fit without a camera tile).
    pub layout: Layout,
    /// One path; two for stacked and screen (top, bottom); none for fit. Keyframe times are clip
    /// seconds.
    pub paths: Vec<CropPath>,
    pub quality: Quality,
}

/// The planned framing of one clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FramePlan {
    /// The layout of the longest stretch.
    pub layout: Layout,
    pub src_w: f64,
    pub src_h: f64,
    pub out_w: f64,
    pub out_h: f64,
    pub content: Rect,
    /// The clip's stretches, in order, covering it end to end.
    pub segments: Vec<PlanSegment>,
    pub subjects: Vec<SubjectSummary>,
    /// Whether speaker turns steered the cuts (false: held on the most prominent face).
    pub used_turns: bool,
    /// The worst of the stretches.
    pub quality: Quality,
}

/// One frame of a [`FramePlan`] ([`FramePlan::view_at`]): the source rectangles (source pixels)
/// that fill the `out_w`×`out_h` frame in equal bands top to bottom, or none to fit the whole
/// `content` picture inside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameView {
    pub src_w: f64,
    pub src_h: f64,
    pub out_w: f64,
    pub out_h: f64,
    pub content: Rect,
    pub crops: Vec<Rect>,
}

impl FramePlan {
    /// The stretch playing `t` seconds into the clip (the last one past the end).
    pub fn segment_at(&self, t: f64) -> Option<&PlanSegment> {
        self.segments.iter().find(|s| t < s.end).or(self.segments.last())
    }

    /// The ffmpeg filter chain that frames the source as planned and scales it to
    /// `out_w`×`out_h` (one simple filtergraph: one input, one output). Several stretches are
    /// trimmed out of one split, framed each on its own, and joined back with concat.
    pub fn ffmpeg_filter(&self, out_w: u32, out_h: u32) -> String {
        match self.segments.as_slice() {
            [] => format!("scale={out_w}:{out_h},setsar=1"),
            [one] => self.segment_filter(one, out_w, out_h, 0.0, ""),
            segs => {
                let n = segs.len();
                let mut f = format!("split={n}");
                for k in 0..n {
                    f.push_str(&format!("[s{k}]"));
                }
                for (k, seg) in segs.iter().enumerate() {
                    let trim = if k == 0 {
                        format!("trim=end={:.4}", seg.end)
                    } else if k == n - 1 {
                        format!("trim=start={:.4}", seg.start)
                    } else {
                        format!("trim=start={:.4}:end={:.4}", seg.start, seg.end)
                    };
                    let shift = if k == 0 { 0.0 } else { seg.start };
                    f.push_str(&format!(
                        ";[s{k}]{trim},setpts=PTS-STARTPTS,{}[v{k}]",
                        self.segment_filter(seg, out_w, out_h, shift, &k.to_string())
                    ));
                }
                f.push(';');
                for k in 0..n {
                    f.push_str(&format!("[v{k}]"));
                }
                f.push_str(&format!("concat=n={n}:v=1:a=0,setsar=1"));
                f
            }
        }
    }

    /// One stretch's chain; its paths' times moved back by `shift` (the stretch starts at 0 once
    /// trimmed). `tag` keeps link labels apart between stretches.
    fn segment_filter(&self, seg: &PlanSegment, out_w: u32, out_h: u32, shift: f64, tag: &str) -> String {
        let crop = |p: &CropPath| {
            let kfs: Vec<Keyframe> = p.keyframes.iter().map(|k| Keyframe { t: (k.t - shift).max(0.0), ..*k }).collect();
            format!(
                "crop=w={}:h={}:x='{}':y='{}':exact=1",
                p.w as i64,
                p.h as i64,
                ffmpeg_expr(&kfs, |k| k.x),
                ffmpeg_expr(&kfs, |k| k.y)
            )
        };
        match (seg.layout, seg.paths.as_slice()) {
            (Layout::Fit, _) | (_, []) => self.fit_filter(out_w, out_h, tag),
            (_, [top, bottom]) => {
                let top_h = (out_h / 2) & !1;
                let bottom_h = out_h - top_h;
                format!(
                    "split=2[fa{tag}][fb{tag}];[fa{tag}]{},scale={out_w}:{top_h},setsar=1[ft{tag}];[fb{tag}]{},scale={out_w}:{bottom_h},setsar=1[fl{tag}];[ft{tag}][fl{tag}]vstack=inputs=2,setsar=1",
                    crop(top),
                    crop(bottom)
                )
            }
            (_, [one, ..]) => format!("{},scale={out_w}:{out_h},setsar=1", crop(one)),
        }
    }

    /// The whole picture (black bars left out) fitted inside the frame, over a blurred copy of
    /// itself filling it. The blur runs on an eighth-size copy: cheap, and softer.
    fn fit_filter(&self, out_w: u32, out_h: u32, tag: &str) -> String {
        let c = self.content;
        let full = c.x <= 0.5 && c.y <= 0.5 && c.w >= self.src_w - 1.0 && c.h >= self.src_h - 1.0;
        let pre = if full {
            String::new()
        } else {
            format!("crop={}:{}:{}:{},", even(c.w) as i64, even(c.h) as i64, c.x.round() as i64, c.y.round() as i64)
        };
        let (bw, bh) = (even(out_w as f64 / 8.0) as u32, even(out_h as f64 / 8.0) as u32);
        // The picture's size inside the frame, worked out here so it is even and exact.
        let scale = (out_w as f64 / c.w).min(out_h as f64 / c.h);
        let (fw, fh) = (even(c.w * scale).min(out_w as f64) as u32, even(c.h * scale).min(out_h as f64) as u32);
        format!(
            "{pre}split=2[ba{tag}][bf{tag}];[ba{tag}]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},boxblur=3:2,scale={out_w}:{out_h},setsar=1[bb{tag}];[bf{tag}]scale={fw}:{fh},setsar=1[bg{tag}];[bb{tag}][bg{tag}]overlay=x={}:y={},setsar=1",
            (out_w - fw) / 2,
            (out_h - fh) / 2
        )
    }

    /// What one frame of the plan shows at `t` seconds into the clip, as the render frames it
    /// (see `segment_filter`): one crop filling the frame, two stacked top and bottom, or none
    /// (fit: the whole `content` picture inside the frame). For a picture of a framed clip
    /// without rendering one, e.g. a keyframe thumbnail cropped the same way.
    pub fn view_at(&self, t: f64) -> FrameView {
        let crop = |p: &CropPath| {
            let (x, y) = p.at(t);
            Rect { x, y, w: p.w, h: p.h }
        };
        let crops = match self.segment_at(t) {
            Some(seg) if seg.layout != Layout::Fit => match seg.paths.as_slice() {
                [top, bottom] => vec![crop(top), crop(bottom)],
                [one, ..] => vec![crop(one)],
                [] => Vec::new(),
            },
            _ => Vec::new(),
        };
        FrameView { src_w: self.src_w, src_h: self.src_h, out_w: self.out_w, out_h: self.out_h, content: self.content, crops }
    }

    /// This plan with every path held still where it is at `t` (one stretch), for a still frame.
    pub fn frozen_at(&self, t: f64) -> FramePlan {
        let mut out = self.clone();
        out.segments = self
            .segment_at(t)
            .map(|seg| {
                let mut seg = seg.clone();
                for p in &mut seg.paths {
                    let (x, y) = p.at(t);
                    p.keyframes = vec![Keyframe { t: 0.0, x, y, cut: false }];
                }
                seg.start = 0.0;
                seg.end = f64::MAX;
                vec![seg]
            })
            .unwrap_or_default();
        out
    }

    /// A short stable digest of the plan, for proxy cache names.
    pub fn digest(&self) -> String {
        let s = serde_json::to_string(&self.segments).unwrap_or_default();
        let h = blake3::hash(s.as_bytes()).to_hex();
        h[..12].to_string()
    }
}

/// Piecewise-linear value of the path at `t` (holds before the first and after the last keyframe).
pub fn value_at(kfs: &[Keyframe], t: f64) -> (f64, f64) {
    let Some(first) = kfs.first() else { return (0.0, 0.0) };
    if t <= first.t {
        return (first.x, first.y);
    }
    let idx = kfs.partition_point(|k| k.t <= t).saturating_sub(1);
    let a = kfs[idx];
    match kfs.get(idx + 1) {
        Some(b) if !b.cut && b.t > a.t => {
            let u = ((t - a.t) / (b.t - a.t)).clamp(0.0, 1.0);
            (a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u)
        }
        _ => (a.x, a.y),
    }
}

/// The path as one flat ffmpeg expression in `t`: the first value plus a ramp per move
/// (`slope*(clip(t,a,b)-a)`) and a step per cut (`d*gte(t,c)`). No nesting, so long clips with
/// many moves parse and evaluate fine.
pub fn ffmpeg_expr(kfs: &[Keyframe], axis: impl Fn(&Keyframe) -> f64) -> String {
    let Some(first) = kfs.first() else { return "0".into() };
    let mut s = format!("{:.1}", axis(first));
    let sign = |v: f64| if v < 0.0 { '-' } else { '+' };
    for pair in kfs.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let d = axis(&b) - axis(&a);
        if d.abs() < 0.05 {
            continue;
        }
        if b.cut || b.t - a.t < 1e-3 {
            s.push_str(&format!("{}{:.2}*gte(t,{:.4})", sign(d), d.abs(), b.t));
        } else {
            let slope = d / (b.t - a.t);
            s.push_str(&format!("{}{:.4}*(clip(t,{:.4},{:.4})-{:.4})", sign(slope), slope.abs(), a.t, b.t, a.t));
        }
    }
    s
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn smoothstep(u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

fn even(v: f64) -> f64 {
    ((v / 2.0).floor() * 2.0).max(2.0)
}

/// Clamp a crop coordinate into `[lo, hi - size]` when the crop fits there, else into the
/// fallback range `[clo, chi - size]`.
fn clamp_axis(v: f64, size: f64, lo: f64, hi: f64, clo: f64, chi: f64) -> f64 {
    if hi - lo >= size - 0.5 { v.clamp(lo, (hi - size).max(lo)) } else { v.clamp(clo, (chi - size).max(clo)) }
}

/// Subjects worth framing: big enough next to the largest face and on screen long enough.
pub fn kept_subjects(input: &FramingInput) -> Vec<usize> {
    let sizes: Vec<f64> = input.subjects.iter().map(Subject::median_h).collect();
    // The people of a clip are there most of the time; faces on a shared screen (a thumbnail, a
    // slide) come and go with it. Keep faces seen at least half as often as the steadiest one.
    let presence: Vec<f64> = input.subjects.iter().map(|s| s.presence(input.duration)).collect();
    let steadiest = presence.iter().copied().fold(0.0, f64::max);
    let min_presence = MIN_PRESENCE.max(steadiest * 0.5);
    let present: Vec<usize> = (0..input.subjects.len())
        .filter(|&i| {
            !input.subjects[i].samples.is_empty() && (presence[i] >= min_presence || input.duration * SAMPLE_HZ < 5.0)
        })
        .collect();
    // Size is judged among the faces that are there most of the time: a big face seen for a
    // moment (a layout change at the clip's edge) must not push out the people who stay.
    let largest = present.iter().map(|&i| sizes[i]).fold(0.0, f64::max);
    let mut kept: Vec<usize> = present.into_iter().filter(|&i| sizes[i] >= largest * MIN_SUBJECT_SIZE).collect();
    if kept.is_empty()
        && let Some(i) = (0..sizes.len())
            .filter(|&i| !input.subjects[i].samples.is_empty())
            .max_by_key(|&i| input.subjects[i].samples.len())
    {
        kept.push(i);
    }
    kept
}

/// The most prominent of the kept subjects: big and on screen most of the time.
fn dominant(input: &FramingInput, kept: &[usize]) -> Option<usize> {
    kept.iter().copied().max_by(|&a, &b| {
        let score = |i: usize| input.subjects[i].median_h() * input.subjects[i].presence(input.duration);
        score(a).total_cmp(&score(b))
    })
}

/// Each subject's own part of the frame (its "tile"): split from each neighbour halfway between
/// the two faces, sideways or up/down, whichever way they are further apart.
fn regions(input: &FramingInput, kept: &[usize]) -> HashMap<usize, Rect> {
    let c = input.content;
    let centre = |i: usize| (input.subjects[i].median_cx() * input.src_w, input.subjects[i].median_cy() * input.src_h);
    let mut out = HashMap::new();
    for &i in kept {
        let (cx, cy) = centre(i);
        let (mut l, mut r, mut t, mut b) = (c.x, c.right(), c.y, c.bottom());
        for &j in kept {
            if i == j {
                continue;
            }
            let (ox, oy) = centre(j);
            let (dx, dy) = (ox - cx, oy - cy);
            if dx.abs() >= dy.abs() {
                let mid = (cx + ox) / 2.0;
                if dx > 0.0 { r = r.min(mid) } else { l = l.max(mid) }
            } else {
                let mid = (cy + oy) / 2.0;
                if dy > 0.0 { b = b.min(mid) } else { t = t.max(mid) }
            }
        }
        out.insert(i, Rect { x: l, y: t, w: (r - l).max(1.0), h: (b - t).max(1.0) });
    }
    out
}

/// Turn speaker turns into shots: merge one speaker's turns across short pauses, ignore turns
/// shorter than `MIN_SHOT_S` (backchannels), never cut sooner than `MIN_SHOT_S` after the last
/// cut or closer than that to the end. Pauses keep the current shot.
pub fn build_shots(turns: &[Turn], duration: f64, fallback: Option<usize>) -> Vec<Shot> {
    let mut ts: Vec<Turn> = turns
        .iter()
        .map(|t| Turn { start: t.start.max(0.0), end: t.end.min(duration), subject: t.subject })
        .filter(|t| t.end > t.start)
        .collect();
    ts.sort_by(|a, b| a.start.total_cmp(&b.start));
    // Merge a speaker's turns across short pauses, and across another speaker's backchannel.
    let mut merged: Vec<Turn> = Vec::new();
    for t in ts {
        if t.end - t.start < MIN_SHOT_S
            && merged.last().is_some_and(|m: &Turn| m.subject != t.subject && t.start - m.end < TURN_MERGE_GAP_S)
        {
            continue; // a backchannel inside someone else's turn
        }
        match merged.last_mut() {
            Some(m) if m.subject == t.subject && t.start - m.end < TURN_MERGE_GAP_S => m.end = m.end.max(t.end),
            _ => merged.push(t),
        }
    }
    let long: Vec<Turn> = merged.into_iter().filter(|t| t.end - t.start >= MIN_SHOT_S).collect();

    let Some(first) = long.first() else {
        return vec![Shot { start: 0.0, end: duration, subject: fallback }];
    };
    let mut shots = vec![Shot { start: 0.0, end: duration, subject: Some(first.subject) }];
    for t in &long[1..] {
        let cur = shots.last().unwrap();
        if Some(t.subject) == cur.subject {
            continue;
        }
        let cut = t.start.max(cur.start + MIN_SHOT_S);
        if t.end - cut < MIN_SHOT_S.min(t.end - t.start) * 0.5 || duration - cut < MIN_SHOT_S {
            continue;
        }
        shots.last_mut().unwrap().end = cut;
        shots.push(Shot { start: cut, end: duration, subject: Some(t.subject) });
    }
    shots
}

struct Geometry {
    w: f64,
    h: f64,
    eye_line: f64,
}

/// The crop's corner that puts this face centred with the eyes on the eye line, kept inside the
/// subject's tile when the crop fits in it, else inside the picture.
fn target(input: &FramingInput, g: &Geometry, region: Option<&Rect>, s: &FaceSample) -> (f64, f64) {
    let c = input.content;
    let r = region.copied().unwrap_or(c);
    let x = s.cx * input.src_w - g.w / 2.0;
    let y = s.eye_y * input.src_h - g.eye_line * g.h;
    (clamp_axis(x, g.w, r.x, r.right(), c.x, c.right()), clamp_axis(y, g.h, r.y, r.bottom(), c.y, c.bottom()))
}

fn centre_target(input: &FramingInput, g: &Geometry) -> (f64, f64) {
    let c = input.content;
    (c.x + (c.w - g.w) / 2.0, c.y + (c.h - g.h) / 2.0)
}

/// Where to hold a subject when it is not seen during a shot: its usual place.
fn resting_target(input: &FramingInput, g: &Geometry, region: Option<&Rect>, subject: Option<usize>) -> (f64, f64) {
    let Some(sub) = subject.map(|i| &input.subjects[i]).filter(|s| !s.samples.is_empty()) else {
        return centre_target(input, g);
    };
    let pts: Vec<(f64, f64)> = sub.samples.iter().map(|s| target(input, g, region, s)).collect();
    (median(pts.iter().map(|p| p.0).collect()), median(pts.iter().map(|p| p.1).collect()))
}

/// The path through one shot: snap to the face at the start, then hold while it stays inside the
/// dead zone, ease to re-centre it when it drifts out, and hold through missed detections.
fn shot_keyframes(input: &FramingInput, g: &Geometry, regions: &HashMap<usize, Rect>, shot: &Shot) -> Vec<Keyframe> {
    let region = shot.subject.and_then(|i| regions.get(&i));
    let raw: Vec<(f64, f64, f64)> = match shot.subject {
        Some(i) => input.subjects[i]
            .samples_in(shot.start, shot.end)
            .map(|s| {
                let (x, y) = target(input, g, region, s);
                (s.t, x, y)
            })
            .collect(),
        None => Vec::new(),
    };
    if raw.is_empty() {
        let (x, y) = resting_target(input, g, region, shot.subject);
        return vec![Keyframe { t: shot.start, x, y, cut: false }];
    }
    // Running median: one bad detection does not move the crop.
    let smoothed: Vec<(f64, f64, f64)> = raw
        .iter()
        .map(|&(t, _, _)| {
            let win: Vec<&(f64, f64, f64)> = raw.iter().filter(|p| (p.0 - t).abs() <= SMOOTH_HALF_S).collect();
            (t, median(win.iter().map(|p| p.1).collect()), median(win.iter().map(|p| p.2).collect()))
        })
        .collect();

    let mut pos = (smoothed[0].1, smoothed[0].2);
    let mut out = vec![Keyframe { t: shot.start, x: pos.0, y: pos.1, cut: false }];
    let mut busy_until = shot.start;
    let end_limit = shot.end - 0.05;
    for &(t, tx, ty) in &smoothed {
        if t < busy_until {
            continue;
        }
        let (dx, dy) = (tx - pos.0, ty - pos.1);
        if dx.abs() <= DEAD_ZONE_X * g.w && dy.abs() <= DEAD_ZONE_Y * g.h {
            continue;
        }
        let dist = (dx / g.w).hypot(dy / g.h);
        let mut dur = (dist / EASE_SPEED).clamp(EASE_MIN_S, EASE_MAX_S);
        if t + dur > end_limit {
            dur = end_limit - t;
            if dur < EASE_MIN_S * 0.6 {
                continue;
            }
        }
        for k in 0..=4 {
            let u = k as f64 / 4.0;
            let e = smoothstep(u);
            out.push(Keyframe { t: t + u * dur, x: pos.0 + dx * e, y: pos.1 + dy * e, cut: false });
        }
        pos = (tx, ty);
        busy_until = t + dur;
    }
    // Two keyframes at one instant (a move starting on the shot's first sample) add nothing.
    out.dedup_by(|b, a| (b.t - a.t).abs() < 1e-6 && (b.x - a.x).abs() < 0.05 && (b.y - a.y).abs() < 0.05);
    out
}

fn path_for_shots(input: &FramingInput, g: &Geometry, regions: &HashMap<usize, Rect>, shots: &[Shot]) -> CropPath {
    let mut kfs: Vec<Keyframe> = Vec::new();
    for (i, shot) in shots.iter().enumerate() {
        let mut part = shot_keyframes(input, g, regions, shot);
        if i > 0
            && let Some(first) = part.first_mut()
        {
            first.cut = true;
        }
        kfs.extend(part);
    }
    for k in &mut kfs {
        k.x = (k.x * 10.0).round() / 10.0;
        k.y = (k.y * 10.0).round() / 10.0;
        k.t = (k.t * 10000.0).round() / 10000.0;
    }
    CropPath { w: g.w, h: g.h, keyframes: kfs, shots: shots.to_vec() }
}

/// One crop size for the clip (ffmpeg's crop size is fixed): face about `face_frac` of the crop
/// height, no smaller than `MIN_CROP_FRAC` of the source, inside the picture, and shrunk to stay
/// inside a person's tile while their face still fits.
fn crop_size(
    input: &FramingInput,
    aspect: f64,
    face_frac: f64,
    faces: &[usize],
    regions: &HashMap<usize, Rect>,
) -> (f64, f64) {
    let c = input.content;
    let fit = |h: f64| {
        let mut h = h.min(c.h);
        let mut w = h * aspect;
        if w > c.w {
            w = c.w;
            h = w / aspect;
        }
        (w, h)
    };
    let face_px = median(faces.iter().map(|&i| input.subjects[i].median_h() * input.src_h).collect());
    if faces.is_empty() || face_px <= 0.0 {
        let (w, h) = fit(c.h);
        return (even(w), even(h));
    }
    let want_h = (face_px / face_frac).max(MIN_CROP_FRAC * input.src_h);
    let (mut w, mut h) = fit(want_h);
    // Shrink into the narrowest tile among the people framed, while the face still fits.
    let tile = faces
        .iter()
        .filter_map(|i| regions.get(i))
        .fold((f64::MAX, f64::MAX), |acc, r| (acc.0.min(r.w), acc.1.min(r.h)));
    if tile.0 < f64::MAX {
        let tw = w.min(tile.0).min(tile.1 * aspect);
        let th = tw / aspect;
        if tw < w && face_px / th <= MAX_FACE_FRAC {
            w = tw;
            h = th;
        }
    }
    // Never narrower than the widest face framed, with a margin: a face cut off at the sides is
    // worse than a little of the neighbour.
    let face_w = faces.iter().map(|&i| input.subjects[i].med(|s| s.w) * input.src_w).fold(0.0, f64::max);
    if w < face_w * FACE_MARGIN_W {
        (w, h) = fit((face_w * FACE_MARGIN_W) / aspect);
    }
    (even(w), even(h))
}

/// The eye line for this crop: `base`, lowered when the face is so big that the top of the head
/// would be cut (the head reaches about 0.55 face heights above the eyes).
fn eye_line_for(input: &FramingInput, faces: &[usize], crop_h: f64, base: f64) -> f64 {
    let face_px = median(faces.iter().map(|&i| input.subjects[i].median_h() * input.src_h).collect());
    base.max((0.55 * face_px / crop_h + 0.10).min(0.5))
}

/// The listener for each speaker shot: the other subject seen most during it.
fn listener_shots(input: &FramingInput, kept: &[usize], speaker: &[Shot]) -> Vec<Shot> {
    let overall = |exclude: Option<usize>| {
        kept.iter().copied().filter(|&i| Some(i) != exclude).max_by_key(|&i| input.subjects[i].samples.len())
    };
    let mut out: Vec<Shot> = Vec::new();
    for s in speaker {
        let other = kept
            .iter()
            .copied()
            .filter(|&i| Some(i) != s.subject)
            .map(|i| (i, input.subjects[i].samples_in(s.start, s.end).count()))
            .filter(|&(_, n)| n > 0)
            .max_by_key(|&(_, n)| n)
            .map(|(i, _)| i)
            .or_else(|| overall(s.subject));
        match out.last_mut() {
            Some(prev) if prev.subject == other => prev.end = s.end,
            _ => out.push(Shot { start: s.start, end: s.end, subject: other }),
        }
    }
    out
}

/// How well the faces of these shots fit their crop.
fn quality(input: &FramingInput, path: &CropPath) -> Quality {
    let mut longest = 0.0;
    let mut at = None;
    let mut seen = 0usize;
    let mut cut = 0usize;
    let mut heights = Vec::new();
    for shot in &path.shots {
        let times: Vec<f64> = match shot.subject {
            Some(i) => input.subjects[i].samples_in(shot.start, shot.end).map(|s| s.t).collect(),
            None => Vec::new(),
        };
        // Gaps between detections, counting from the shot's start and to its end.
        let mut prev = shot.start;
        for &t in times.iter().chain(std::iter::once(&shot.end)) {
            // A detection every 1/SAMPLE_HZ is continuous; anything longer is a gap.
            let gap = t - prev - 1.0 / SAMPLE_HZ;
            if gap > longest {
                longest = gap;
                at = Some(prev);
            }
            prev = t;
        }
        if let Some(i) = shot.subject {
            for s in input.subjects[i].samples_in(shot.start, shot.end) {
                seen += 1;
                heights.push(s.h * input.src_h);
                let (x, y) = path.at(s.t);
                let fx0 = (s.cx - s.w / 2.0) * input.src_w;
                let fx1 = (s.cx + s.w / 2.0) * input.src_w;
                let fy0 = (s.cy - s.h / 2.0) * input.src_h;
                let fy1 = (s.cy + s.h / 2.0) * input.src_h;
                let tol = 0.15 * s.w * input.src_w;
                let outside = fx0 < x - tol || fx1 > x + path.w + tol || fy0 < y - tol || fy1 > y + path.h + tol;
                if s.cut_off || outside {
                    cut += 1;
                }
            }
        }
    }
    Quality {
        longest_no_face_s: (longest.max(0.0) * 100.0).round() / 100.0,
        no_face_at: at.filter(|_| longest > 0.0).map(|a| (a * 100.0).round() / 100.0),
        cut_off_share: if seen > 0 { cut as f64 / seen as f64 } else { 0.0 },
        face_frac: if heights.is_empty() { 0.0 } else { median(heights) / path.h },
    }
}

fn worse(a: Quality, b: Quality) -> Quality {
    let (longest_no_face_s, no_face_at) = if b.longest_no_face_s > a.longest_no_face_s {
        (b.longest_no_face_s, b.no_face_at)
    } else {
        (a.longest_no_face_s, a.no_face_at)
    };
    Quality {
        longest_no_face_s,
        no_face_at,
        cut_off_share: a.cut_off_share.max(b.cut_off_share),
        face_frac: if a.face_frac > 0.0 && b.face_frac > 0.0 {
            a.face_frac.min(b.face_frac)
        } else {
            a.face_frac.max(b.face_frac)
        },
    }
}

/// The smallest crop of a camera tile under a screen share, as a share of the source height:
/// tiles are small, so this zooms further than [`MIN_CROP_FRAC`] allows elsewhere.
pub const MIN_TILE_CROP_FRAC: f64 = 0.22;
/// A camera tile at least this tall (share of the source height) holds the whole crop of it.
pub const MIN_TILE_H_FRAC: f64 = 0.15;
/// A stretch with no face shorter than this, between two stretches of the same people, holds
/// the crop instead of changing layout.
pub const NO_FACE_HOLD_S: f64 = 4.0;
/// Pieces of a clip shorter than this (where an override meets a scene change) join the stretch
/// before them.
const MIN_PIECE_S: f64 = 0.5;

/// Every face of the input as the scene classifier sees it: relative to the picture (black bars
/// left out).
fn scene_obs(input: &FramingInput) -> Vec<FaceObs> {
    let c = input.content;
    let (cw, ch) = (c.w.max(1.0), c.h.max(1.0));
    input
        .subjects
        .iter()
        .flat_map(|s| s.samples.iter())
        .map(|s| FaceObs {
            t: s.t,
            cx: ((s.cx * input.src_w - c.x) / cw).clamp(0.0, 1.0),
            cy: ((s.cy * input.src_h - c.y) / ch).clamp(0.0, 1.0),
            w: s.w * input.src_w / cw,
            h: s.h * input.src_h / ch,
        })
        .collect()
}

/// What a clip's picture is, stretch by stretch (clip seconds).
pub fn scenes_of(input: &FramingInput) -> Vec<SceneSpan> {
    crate::scenes::segment(&scene_obs(input), 0.0, input.duration)
}

/// The layout a stretch gets: an override wins; otherwise people are framed as asked, a screen
/// share as screen over speaker, and an empty picture is fitted whole.
pub fn layout_for(scene: Scene, requested: Layout, chosen: Option<Layout>) -> Layout {
    if let Some(l) = chosen {
        return l;
    }
    match (scene, requested) {
        (_, Layout::Fit) => Layout::Fit,
        (Scene::NoFace, _) => Layout::Fit,
        (Scene::ScreenShare, _) => Layout::Screen,
        (_, Layout::Screen) => Layout::FollowSpeaker,
        (_, l) => l,
    }
}

/// The part `a..b` of an input, on its own clock (`t = 0` at `a`). Subject indices are kept.
fn sub_input(input: &FramingInput, a: f64, b: f64) -> FramingInput {
    FramingInput {
        src_w: input.src_w,
        src_h: input.src_h,
        duration: (b - a).max(0.0),
        subjects: input
            .subjects
            .iter()
            .map(|s| Subject {
                person_id: s.person_id,
                samples: s.samples_in(a, b).map(|x| FaceSample { t: x.t - a, ..*x }).collect(),
            })
            .collect(),
        turns: input
            .turns
            .iter()
            .filter_map(|t| {
                let (s, e) = (t.start.max(a), t.end.min(b));
                (e > s).then_some(Turn { start: s - a, end: e - a, subject: t.subject })
            })
            .collect(),
        content: input.content,
        origin_s: input.origin_s + a,
    }
}

fn shift_path(p: &mut CropPath, by: f64) {
    for k in &mut p.keyframes {
        k.t = ((k.t + by) * 10000.0).round() / 10000.0;
    }
    for s in &mut p.shots {
        s.start += by;
        s.end += by;
    }
}

/// The clip cut into stretches of one scene and one layout: scene changes, and the edges of
/// layout overrides, cut it.
fn pieces(input: &FramingInput, framing: &Framing, scenes: &[SceneSpan]) -> Vec<(f64, f64, Scene, Layout)> {
    let dur = input.duration;
    let rel: Vec<(f64, f64, Layout)> = framing
        .overrides
        .iter()
        .map(|o| ((o.from_s - input.origin_s).max(0.0), (o.to_s - input.origin_s).min(dur), o.layout))
        .filter(|o| o.1 - o.0 > 0.05)
        .collect();
    let mut cuts: Vec<f64> = vec![0.0, dur];
    cuts.extend(scenes.iter().map(|s| s.start));
    for o in &rel {
        cuts.push(o.0);
        cuts.push(o.1);
    }
    cuts.retain(|&c| (0.0..=dur).contains(&c));
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 0.05);
    // A short gap in the faces between two stretches of people is a face looking away or a
    // missed detection, not a new shot: the crop holds through it.
    let scene_at = |t: f64| -> Scene {
        let Some(k) = scenes.iter().position(|s| t >= s.start && t < s.end).or(scenes.len().checked_sub(1)) else {
            return Scene::NoFace;
        };
        let s = scenes[k];
        if s.scene == Scene::NoFace
            && s.len() < NO_FACE_HOLD_S
            && k > 0
            && scenes[k - 1].scene.has_people()
            && scenes.get(k + 1).is_some_and(|n| n.scene == scenes[k - 1].scene)
        {
            return scenes[k - 1].scene;
        }
        s.scene
    };
    let mut out: Vec<(f64, f64, Scene, Layout)> = Vec::new();
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mid = (a + b) / 2.0;
        let scene = scene_at(mid);
        let chosen = rel.iter().rev().find(|o| mid >= o.0 && mid < o.1).map(|o| o.2);
        let layout = layout_for(scene, framing.layout, chosen);
        match out.last_mut() {
            Some(prev) if (prev.2 == scene && prev.3 == layout) || b - a < MIN_PIECE_S => prev.1 = b,
            _ => out.push((a, b, scene, layout)),
        }
    }
    if out.is_empty() {
        out.push((0.0, dur, Scene::NoFace, layout_for(Scene::NoFace, framing.layout, None)));
    }
    out
}

/// Plan the framing of one clip for an output of `out_w`×`out_h`.
pub fn plan(input: &FramingInput, framing: &Framing, out_w: f64, out_h: f64) -> FramePlan {
    let summaries: Vec<SubjectSummary> = (0..input.subjects.len())
        .map(|i| {
            let s = &input.subjects[i];
            SubjectSummary {
                index: i,
                person_id: s.person_id,
                median_cx: (s.median_cx() * 1000.0).round() / 1000.0,
                median_cy: (s.median_cy() * 1000.0).round() / 1000.0,
                median_h: (s.median_h() * 1000.0).round() / 1000.0,
                presence: (s.presence(input.duration) * 100.0).round() / 100.0,
            }
        })
        .collect();
    let scenes = scenes_of(input);
    let mut used_turns = false;
    let segments: Vec<PlanSegment> = if framing.layout == Layout::Fixed && framing.box_.is_some() {
        // A box is the editor's exact choice: one still crop, whatever the picture does.
        let scene = scenes.iter().max_by(|a, b| a.len().total_cmp(&b.len())).map(|s| s.scene).unwrap_or(Scene::NoFace);
        let (layout, paths, quality, turns) = plan_people(input, framing, out_w, out_h);
        used_turns |= turns;
        vec![PlanSegment { start: 0.0, end: input.duration, scene, layout, paths, quality }]
    } else {
        pieces(input, framing, &scenes)
            .into_iter()
            .map(|(a, b, scene, layout)| {
                let sub = sub_input(input, a, b);
                let (layout, mut paths, quality) = match layout {
                    Layout::Fit => (Layout::Fit, Vec::new(), Quality::default()),
                    Layout::Screen => match plan_screen(&sub, out_w, out_h) {
                        Some((paths, q, turns)) => {
                            used_turns |= turns;
                            (Layout::Screen, paths, q)
                        }
                        None => (Layout::Fit, Vec::new(), Quality::default()),
                    },
                    l => {
                        let f = Framing { layout: l, overrides: Vec::new(), ..framing.clone() };
                        let (l, paths, q, turns) = plan_people(&sub, &f, out_w, out_h);
                        used_turns |= turns;
                        (l, paths, q)
                    }
                };
                for p in &mut paths {
                    shift_path(p, a);
                }
                let mut quality = quality;
                quality.no_face_at = quality.no_face_at.map(|t| ((t + a) * 100.0).round() / 100.0);
                PlanSegment { start: a, end: b, scene, layout, paths, quality }
            })
            .collect()
    };
    let quality = segments.iter().filter(|s| !s.paths.is_empty()).map(|s| s.quality).reduce(worse).unwrap_or_default();
    let layout = segments
        .iter()
        .max_by(|a, b| (a.end - a.start).total_cmp(&(b.end - b.start)))
        .map(|s| s.layout)
        .unwrap_or(framing.layout);
    FramePlan {
        layout,
        src_w: input.src_w,
        src_h: input.src_h,
        out_w,
        out_h,
        content: input.content,
        segments,
        subjects: summaries,
        used_turns,
        quality,
    }
}

/// A screen share: the screen (the part of the picture clear of the camera tiles) on top, the
/// speaker's tile below, following whoever talks. None without a camera tile to show.
fn plan_screen(input: &FramingInput, out_w: f64, out_h: f64) -> Option<(Vec<CropPath>, Quality, bool)> {
    let kept = kept_subjects(input);
    if kept.is_empty() {
        return None;
    }
    let c = input.content;
    let half_aspect = out_w / (out_h / 2.0);
    // Each camera tile: a 16:9 picture about 2.4 face heights tall around the face.
    let tiles: HashMap<usize, Rect> = kept
        .iter()
        .map(|&i| {
            let s = &input.subjects[i];
            let fh = s.median_h() * input.src_h;
            let th = 2.4 * fh;
            let tw = th * 16.0 / 9.0;
            let (cx, cy) = (s.median_cx() * input.src_w, s.median_cy() * input.src_h + 0.15 * fh);
            (i, Rect { x: cx - tw / 2.0, y: cy - th / 2.0, w: tw, h: th })
        })
        .collect();
    let all = tiles.values().copied().reduce(|a, b| a.union(&b))?;
    // The screen: above the tiles when they sit low, else beside them.
    let region = if all.y > c.y + c.h * 0.5 {
        Rect { x: c.x, y: c.y, w: c.w, h: (all.y - c.y).max(2.0) }
    } else if all.x + all.w / 2.0 > c.x + c.w / 2.0 {
        Rect { x: c.x, y: c.y, w: (all.x - c.x).max(2.0), h: c.h }
    } else {
        Rect { x: all.right(), y: c.y, w: (c.right() - all.right()).max(2.0), h: c.h }
    };
    let mut h = region.h;
    let mut w = h * half_aspect;
    if w > region.w {
        w = region.w;
        h = w / half_aspect;
    }
    let (w, h) = (even(w), even(h));
    // Centred on the picture (a shared window usually is), kept clear of the tiles.
    let x = (c.x + (c.w - w) / 2.0).clamp(region.x, (region.right() - w).max(region.x));
    let y = (region.y + (region.h - h) / 2.0).max(c.y);
    let top = CropPath {
        w,
        h,
        keyframes: vec![Keyframe { t: 0.0, x: x.round(), y: y.round(), cut: false }],
        shots: vec![Shot { start: 0.0, end: input.duration, subject: None }],
    };
    // The speaker's tile, zoomed in as far as the upscale allows.
    let main = dominant(input, &kept);
    let turns: Vec<Turn> = input.turns.iter().copied().filter(|t| kept.contains(&t.subject)).collect();
    let used_turns = !turns.is_empty();
    let shots = if used_turns {
        build_shots(&turns, input.duration, main)
    } else {
        vec![Shot { start: 0.0, end: input.duration, subject: main }]
    };
    let face_px = median(kept.iter().map(|&i| input.subjects[i].median_h() * input.src_h).collect());
    // Inside the tile when it is big enough to upscale: past its edge is the screen, or black.
    let tile_h = main.and_then(|i| tiles.get(&i)).map(|t| t.h).unwrap_or(f64::MAX);
    let mut bh = (face_px / FACE_FRAC_STACKED).max(MIN_TILE_CROP_FRAC * input.src_h).min(c.h);
    if tile_h >= MIN_TILE_H_FRAC * input.src_h {
        bh = bh.min(tile_h);
    }
    let mut bw = bh * half_aspect;
    if bw > c.w {
        bw = c.w;
        bh = bw / half_aspect;
    }
    let g = Geometry { w: even(bw), h: even(bh), eye_line: EYE_LINE_STACKED };
    let bottom = path_for_shots(input, &g, &tiles, &shots);
    let q = quality(input, &bottom);
    Some((vec![top, bottom], q, used_turns))
}

/// Frame people with `framing`'s layout (follow the speaker, stacked, fixed): the paths, the
/// layout used, how well the faces fit, and whether speaker turns steered it.
fn plan_people(
    input: &FramingInput,
    framing: &Framing,
    out_w: f64,
    out_h: f64,
) -> (Layout, Vec<CropPath>, Quality, bool) {
    let kept = kept_subjects(input);
    let regions = regions(input, &kept);
    let pinned = framing
        .person_id
        .and_then(|p| kept.iter().copied().find(|&i| input.subjects[i].person_id == Some(p)))
        .or_else(|| {
            framing.person_id.and_then(|p| (0..input.subjects.len()).find(|&i| input.subjects[i].person_id == Some(p)))
        });
    let main = pinned.or_else(|| dominant(input, &kept));
    let turns: Vec<Turn> = input.turns.iter().copied().filter(|t| kept.contains(&t.subject)).collect();
    let used_turns = framing.layout != Layout::Fixed && pinned.is_none() && !turns.is_empty();
    let speaker_shots = if used_turns {
        build_shots(&turns, input.duration, main)
    } else {
        vec![Shot { start: 0.0, end: input.duration, subject: main }]
    };
    let aspect = out_w / out_h;

    // Fixed on a box: the output-shaped rectangle covering it, centred on it.
    if framing.layout == Layout::Fixed
        && let Some(b) = framing.box_
    {
        let c = input.content;
        let bw = b.w * input.src_w;
        let bh = b.h * input.src_h;
        let mut w = bw.max(bh * aspect).min(c.w);
        let mut h = w / aspect;
        if h > c.h {
            h = c.h;
            w = h * aspect;
        }
        let (w, h) = (even(w), even(h));
        let x = ((b.x + b.w / 2.0) * input.src_w - w / 2.0).clamp(c.x, (c.right() - w).max(c.x));
        let y = ((b.y + b.h / 2.0) * input.src_h - h / 2.0).clamp(c.y, (c.bottom() - h).max(c.y));
        let path = CropPath {
            w,
            h,
            keyframes: vec![Keyframe { t: 0.0, x: x.round(), y: y.round(), cut: false }],
            shots: vec![Shot { start: 0.0, end: input.duration, subject: None }],
        };
        let q = quality(input, &path);
        return (Layout::Fixed, vec![path], q, false);
    }

    let stacked = framing.layout == Layout::Stacked && kept.len() >= 2;
    if stacked {
        let half_aspect = out_w / (out_h / 2.0);
        let (w, h) = crop_size(input, half_aspect, FACE_FRAC_STACKED, &kept, &regions);
        let g = Geometry { w, h, eye_line: eye_line_for(input, &kept, h, EYE_LINE_STACKED) };
        let top = path_for_shots(input, &g, &regions, &speaker_shots);
        let listener = listener_shots(input, &kept, &speaker_shots);
        let bottom = path_for_shots(input, &g, &regions, &listener);
        let q = worse(quality(input, &top), quality(input, &bottom));
        return (Layout::Stacked, vec![top, bottom], q, used_turns);
    }

    let faces: Vec<usize> = {
        let mut f: Vec<usize> = speaker_shots.iter().filter_map(|s| s.subject).collect();
        f.sort_unstable();
        f.dedup();
        f
    };
    let (w, h) = crop_size(input, aspect, FACE_FRAC, &faces, &regions);
    let g = Geometry { w, h, eye_line: eye_line_for(input, &faces, h, EYE_LINE) };
    let path = if framing.layout == Layout::Fixed {
        let region = main.and_then(|i| regions.get(&i));
        let (x, y) = resting_target(input, &g, region, main);
        CropPath {
            w,
            h,
            keyframes: vec![Keyframe { t: 0.0, x: x.round(), y: y.round(), cut: false }],
            shots: speaker_shots.clone(),
        }
    } else {
        path_for_shots(input, &g, &regions, &speaker_shots)
    };
    let q = quality(input, &path);
    let layout = if framing.layout == Layout::Fixed { Layout::Fixed } else { Layout::FollowSpeaker };
    (layout, vec![path], q, used_turns)
}

// ---------------------------------------------------------------------------------------------
// Loading from the index

#[derive(Debug, Clone)]
struct RawBox {
    track_id: i64,
    person_id: Option<i64>,
    sample: FaceSample,
}

fn eye_line(landmarks_json: &str, y: f64, h: f64) -> f64 {
    // YuNet landmarks: right eye, left eye, nose, mouth right, mouth left (normalised).
    serde_json::from_str::<Vec<(f64, f64)>>(landmarks_json)
        .ok()
        .filter(|l| l.len() >= 2 && l[0].1 > 0.0 && l[1].1 > 0.0)
        .map(|l| (l[0].1 + l[1].1) / 2.0)
        .unwrap_or(y + 0.4 * h)
}

fn load_boxes(db: &Db, video_id: i64, from_s: f64, to_s: f64) -> Result<Vec<RawBox>, Error> {
    let map = |r: &rusqlite::Row<'_>, with_person: bool| -> rusqlite::Result<RawBox> {
        let (x, y, w, h): (f64, f64, f64, f64) = (r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?);
        let lm: String = r.get(6)?;
        Ok(RawBox {
            track_id: r.get(0)?,
            person_id: if with_person { r.get(8)? } else { None },
            sample: FaceSample {
                t: r.get::<_, f64>(1)? - from_s,
                cx: x + w / 2.0,
                cy: y + h / 2.0,
                w,
                h,
                eye_y: eye_line(&lm, y, h),
                cut_off: r.get::<_, i64>(7)? != 0,
            },
        })
    };
    let base = "SELECT fb.track_id, fb.t_s, fb.x, fb.y, fb.w, fb.h, fb.landmarks_json, fb.cut_off";
    let tail = "FROM face_boxes fb JOIN face_tracks ft ON ft.id = fb.track_id
                WHERE ft.video_id = ?1 AND fb.t_s >= ?2 AND fb.t_s < ?3 ORDER BY fb.t_s, fb.track_id";
    // `person_id` arrives with the people stage; before it, face tracks group by position.
    if let Ok(mut st) = db.conn.prepare(&format!("{base}, ft.person_id {tail}")) {
        let rows = st.query_map(rusqlite::params![video_id, from_s, to_s], |r| map(r, true))?;
        return Ok(rows.filter_map(Result::ok).collect());
    }
    let mut st = db.conn.prepare(&format!("{base} {tail}"))?;
    let rows = st.query_map(rusqlite::params![video_id, from_s, to_s], |r| map(r, false))?;
    Ok(rows.filter_map(Result::ok).collect())
}

/// Group face boxes into subjects: by person when the people stage named the tracks, else by
/// seat: tracks whose faces sit in the same place and are never on screen together.
pub fn group_subjects(boxes: Vec<(i64, Option<i64>, FaceSample)>) -> Vec<Subject> {
    struct Track {
        person: Option<i64>,
        cx: f64,
        cy: f64,
        start: f64,
        end: f64,
        samples: Vec<FaceSample>,
    }
    let mut by_track: HashMap<i64, Track> = HashMap::new();
    let mut order = Vec::new();
    for (track, person, s) in boxes {
        let e = by_track.entry(track).or_insert_with(|| {
            order.push(track);
            Track { person, cx: 0.0, cy: 0.0, start: s.t, end: s.t, samples: Vec::new() }
        });
        e.start = e.start.min(s.t);
        e.end = e.end.max(s.t);
        e.samples.push(s);
    }
    let mut tracks: Vec<Track> = order.into_iter().filter_map(|id| by_track.remove(&id)).collect();
    for t in &mut tracks {
        t.cx = median(t.samples.iter().map(|s| s.cx).collect());
        t.cy = median(t.samples.iter().map(|s| s.cy).collect());
    }
    // Longest tracks first: they set where the seats are.
    tracks.sort_by_key(|t| std::cmp::Reverse(t.samples.len()));

    struct Group {
        person: Option<i64>,
        cx: f64,
        cy: f64,
        spans: Vec<(f64, f64)>,
        samples: Vec<FaceSample>,
    }
    let mut groups: Vec<Group> = Vec::new();
    for t in tracks {
        let found = match t.person {
            Some(p) => groups.iter().position(|g| g.person == Some(p)),
            None => groups.iter().position(|g| {
                let close = (g.cx - t.cx).hypot(g.cy - t.cy) < SLOT_GAP;
                let together = g.spans.iter().any(|&(a, b)| t.start.max(a) < t.end.min(b) - 0.5);
                close && !together
            }),
        };
        match found {
            Some(i) => {
                let g = &mut groups[i];
                g.spans.push((t.start, t.end));
                g.samples.extend(t.samples);
            }
            None => groups.push(Group {
                person: t.person,
                cx: t.cx,
                cy: t.cy,
                spans: vec![(t.start, t.end)],
                samples: t.samples,
            }),
        }
    }
    let mut subjects: Vec<Subject> = groups
        .into_iter()
        .map(|mut g| {
            g.samples.sort_by(|a, b| a.t.total_cmp(&b.t));
            // One face per instant per subject: keep the bigger one.
            g.samples.dedup_by(|b, a| {
                if (a.t - b.t).abs() < 1e-3 {
                    if b.h > a.h {
                        std::mem::swap(a, b);
                    }
                    true
                } else {
                    false
                }
            });
            Subject { person_id: g.person, samples: g.samples }
        })
        .collect();
    // Left to right, so slot numbers read like the picture.
    subjects.sort_by(|a, b| a.median_cx().total_cmp(&b.median_cx()));
    subjects
}

fn load_turns(db: &Db, video_id: i64, from_s: f64, to_s: f64, subjects: &[Subject]) -> Vec<Turn> {
    let Ok(mut st) = db.conn.prepare(
        "SELECT start_s, end_s, person_id FROM speaker_turns
         WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3 AND person_id IS NOT NULL ORDER BY start_s",
    ) else {
        return Vec::new();
    };
    let Ok(rows) = st.query_map(rusqlite::params![video_id, from_s, to_s], |r| {
        Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?, r.get::<_, i64>(2)?))
    }) else {
        return Vec::new();
    };
    rows.filter_map(Result::ok)
        .filter_map(|(a, b, p)| {
            let subject = subjects.iter().position(|s| s.person_id == Some(p))?;
            Some(Turn { start: (a - from_s).max(0.0), end: (b - from_s).min(to_s - from_s), subject })
        })
        .collect()
}

/// The source's display size (rotation applied).
pub fn video_size(db: &Db, video_id: i64) -> Result<Option<(f64, f64)>, Error> {
    let row: Option<(Option<i64>, Option<i64>, Option<i64>)> = db
        .conn
        .query_row("SELECT width, height, rotation FROM videos WHERE id = ?1", [video_id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .optional()?;
    Ok(row.and_then(|(w, h, rot)| {
        let (w, h) = (w? as f64, h? as f64);
        if rot.unwrap_or(0).rem_euclid(180) == 90 { Some((h, w)) } else { Some((w, h)) }
    }))
}

/// Load what the planner needs for `from_s..to_s` of a video. `content` defaults to the whole
/// frame; pass what `detect_content` found to keep black bars out.
pub fn load_input(
    db: &Db,
    video_id: i64,
    from_s: f64,
    to_s: f64,
    content: Option<Rect>,
) -> Result<Option<FramingInput>, Error> {
    let Some((src_w, src_h)) = video_size(db, video_id)? else { return Ok(None) };
    let boxes = load_boxes(db, video_id, from_s, to_s)?;
    let subjects = group_subjects(boxes.into_iter().map(|b| (b.track_id, b.person_id, b.sample)).collect());
    let turns = load_turns(db, video_id, from_s, to_s, &subjects);
    let full = Rect { x: 0.0, y: 0.0, w: src_w, h: src_h };
    let mut content = content.unwrap_or(full);
    // Never leave a face we might frame outside the "picture".
    let mut input =
        FramingInput { src_w, src_h, duration: (to_s - from_s).max(0.0), subjects, turns, content, origin_s: from_s };
    for &i in &kept_subjects(&input) {
        for s in &input.subjects[i].samples {
            let r =
                Rect { x: (s.cx - s.w / 2.0) * src_w, y: (s.cy - s.h / 2.0) * src_h, w: s.w * src_w, h: s.h * src_h };
            content = content.union(&r);
        }
    }
    input.content = Rect {
        x: content.x.max(0.0),
        y: content.y.max(0.0),
        w: content.w.min(src_w - content.x.max(0.0)),
        h: content.h.min(src_h - content.y.max(0.0)),
    };
    Ok(Some(input))
}

type ContentKey = (PathBuf, i64, i64);
static CONTENT_CACHE: OnceLock<Mutex<HashMap<ContentKey, Option<Rect>>>> = OnceLock::new();

/// The part of the frame that has picture in it, from ffmpeg's `cropdetect` at three points of
/// the range (letterbox bars must be black at all three). None when nothing reliable was found.
pub fn detect_content(ffmpeg: &Path, path: &Path, from_s: f64, to_s: f64, src_w: f64, src_h: f64) -> Option<Rect> {
    let key = (path.to_path_buf(), (from_s * 1000.0).round() as i64, (to_s * 1000.0).round() as i64);
    let cache = CONTENT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&key).copied()) {
        return hit;
    }
    let dur = (to_s - from_s).max(0.0);
    let mut found: Option<Rect> = None;
    for frac in [0.2, 0.5, 0.8] {
        let at = from_s + dur * frac;
        let out = crate::proc::std_command(ffmpeg)
            .args(["-hide_banner", "-nostats", "-ss", &format!("{at:.3}"), "-t", "0.6", "-i"])
            .arg(path)
            .args(["-an", "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"])
            .output();
        let Ok(out) = out else { return None };
        let text = String::from_utf8_lossy(&out.stderr);
        let Some(r) = text.lines().rev().find_map(|l| {
            let c = l.split("crop=").nth(1)?;
            let v: Vec<f64> = c.trim().split(':').filter_map(|p| p.parse().ok()).collect();
            (v.len() == 4).then(|| Rect { x: v[2], y: v[3], w: v[0], h: v[1] })
        }) else {
            continue;
        };
        found = Some(found.map_or(r, |f| f.union(&r)));
    }
    // A dark scene can look like bars; only trust a result that keeps most of the frame.
    let found = found.filter(|r| r.w >= 0.6 * src_w && r.h >= 0.6 * src_h && r.w > 0.0 && r.h > 0.0);
    if let Ok(mut c) = cache.lock() {
        c.insert(key, found);
    }
    found
}

/// Plan one clip's framing, or None when it needs no crop (same shape as the footage) or the
/// video is unknown. With `ffmpeg` and `path`, black bars are detected and kept out.
#[allow(clippy::too_many_arguments)]
pub fn plan_clip(
    db: &Db,
    ffmpeg: Option<&Path>,
    path: Option<&Path>,
    video_id: i64,
    in_s: f64,
    out_s: f64,
    framing: &Framing,
    out_w: f64,
    out_h: f64,
) -> Result<Option<FramePlan>, Error> {
    let Some((src_w, src_h)) = video_size(db, video_id)? else { return Ok(None) };
    if !needs_crop(src_w, src_h, out_w, out_h) {
        return Ok(None);
    }
    let content = match (ffmpeg, path) {
        (Some(f), Some(p)) if p.is_file() => detect_content(f, p, in_s, out_s, src_w, src_h),
        _ => None,
    };
    let Some(input) = load_input(db, video_id, in_s, out_s, content)? else { return Ok(None) };
    Ok(Some(plan(&input, framing, out_w, out_h)))
}

/// Times worth looking at in a planned clip: a little into it, just after each change of
/// stretch, either side of the first cuts, and near the end. At most `max` times, in clip seconds.
pub fn still_times(plan: &FramePlan, duration: f64, max: usize) -> Vec<f64> {
    let mut times = vec![(duration * 0.1).min(1.0)];
    for seg in plan.segments.iter().skip(1).take(3) {
        times.push((seg.start + 0.3).min(duration));
    }
    let cuts: Vec<f64> = plan
        .segments
        .iter()
        .filter_map(|s| s.paths.first())
        .flat_map(|p| p.keyframes.iter().filter(|k| k.cut).map(|k| k.t))
        .collect();
    for c in cuts.iter().take(2) {
        times.push((c - 0.3).max(0.0));
        times.push((c + 0.3).min(duration));
    }
    if cuts.is_empty() && plan.segments.len() <= 1 {
        times.push(duration * 0.5);
    }
    let end = (duration - 0.5).max(0.0);
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| (*a - *b).abs() < 0.2);
    // The end always makes the cut: it is what the closing hold shows.
    times.retain(|t| (t - end).abs() >= 0.2);
    times.truncate(max.max(2) - 1);
    times.push(end);
    times
}

/// Render single frames of the source cropped as the plan has it at each time (clip seconds),
/// to JPEGs in `dir`. Returns the files written.
#[allow(clippy::too_many_arguments)]
pub fn render_stills(
    ffmpeg: &Path,
    source: &Path,
    plan: &FramePlan,
    in_s: f64,
    times: &[f64],
    out_w: u32,
    out_h: u32,
    dir: &Path,
    stem: &str,
) -> Result<Vec<PathBuf>, Error> {
    std::fs::create_dir_all(dir).map_err(|e| Error::Io(dir.to_path_buf(), e))?;
    let mut out = Vec::new();
    for &t in times {
        // The same filter the render uses, with each path frozen at `t`.
        let still = plan.frozen_at(t);
        let file = dir.join(format!("{stem}_{:07.2}.jpg", in_s + t));
        let res = crate::proc::std_command(ffmpeg)
            .args(["-v", "error", "-y", "-ss", &format!("{:.3}", in_s + t), "-i"])
            .arg(source)
            .args(["-frames:v", "1", "-vf", &still.ffmpeg_filter(out_w, out_h), "-q:v", "3"])
            .arg(&file)
            .output()
            .map_err(|e| Error::Io(file.clone(), e))?;
        if !res.status.success() {
            return Err(Error::Preview(format!(
                "ffmpeg could not render a framed still at {:.2}s: {}",
                in_s + t,
                String::from_utf8_lossy(&res.stderr).trim()
            )));
        }
        out.push(file);
    }
    Ok(out)
}

/// A compact account of a plan for people and agents: stretches, subjects, shots, crop sizes,
/// quality.
pub fn summary(plan: &FramePlan) -> serde_json::Value {
    let r2 = |v: f64| (v * 100.0).round() / 100.0;
    let segments: Vec<serde_json::Value> = plan
        .segments
        .iter()
        .map(|seg| {
            let paths: Vec<serde_json::Value> = seg
                .paths
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let role = match (seg.layout, seg.paths.len(), i) {
                        (Layout::Screen, _, 0) => "screen (top)",
                        (Layout::Screen, _, _) => "speaker tile (bottom)",
                        (_, 2, 0) => "speaker (top)",
                        (_, 2, _) => "listener (bottom)",
                        _ => "main",
                    };
                    serde_json::json!({
                        "role": role,
                        "crop": { "w": p.w, "h": p.h },
                        "shots": p.shots.iter().map(|s| serde_json::json!({
                            "start_s": r2(s.start),
                            "end_s": r2(s.end),
                            "subject": s.subject,
                            "person_id": s.subject.and_then(|i| plan.subjects.get(i)).and_then(|x| x.person_id),
                        })).collect::<Vec<_>>(),
                        "cuts": p.keyframes.iter().filter(|k| k.cut).count(),
                        "keyframes": p.keyframes.len(),
                    })
                })
                .collect();
            serde_json::json!({
                "start_s": r2(seg.start),
                "end_s": r2(seg.end),
                "scene": seg.scene.as_str(),
                "layout": seg.layout.as_str(),
                "paths": paths,
                "quality": seg.quality,
            })
        })
        .collect();
    serde_json::json!({
        "layout": plan.layout.as_str(),
        "source": { "w": plan.src_w, "h": plan.src_h },
        "output": { "w": plan.out_w, "h": plan.out_h },
        "content": plan.content,
        "subjects": plan.subjects,
        "used_speaker_turns": plan.used_turns,
        "segments": segments,
        "quality": plan.quality,
    })
}

/// A reachable file of the video, if any.
pub fn source_path(db: &Db, video_id: i64) -> Result<Option<PathBuf>, Error> {
    let mut st = db.conn.prepare("SELECT path FROM video_files WHERE video_id = ?1 ORDER BY path")?;
    let paths: Vec<String> = st.query_map([video_id], |r| r.get(0))?.filter_map(Result::ok).collect();
    Ok(paths.iter().map(PathBuf::from).find(|p| p.is_file()).or_else(|| paths.first().map(PathBuf::from)))
}

/// The harness view of a framing: plan `from_s..to_s` of a video for `out_w`×`out_h` and render
/// a few stills with the crop (at the times `still_times` picks) into `dir`.
/// Returns the plan and (clip seconds, file) per still.
#[allow(clippy::too_many_arguments)]
pub fn preview_crop(
    db: &Db,
    ffmpeg: &Path,
    video_id: i64,
    from_s: f64,
    to_s: f64,
    framing: &Framing,
    out_w: u32,
    out_h: u32,
    dir: &Path,
    max_stills: usize,
) -> Result<(FramePlan, Vec<(f64, PathBuf)>), Error> {
    if to_s <= from_s {
        return Err(Error::Invalid(format!("to_s ({to_s:.2}) must be after from_s ({from_s:.2})")));
    }
    let source = source_path(db, video_id)?.ok_or_else(|| Error::NotFound(format!("a file for video #{video_id}")))?;
    let plan = plan_clip(db, Some(ffmpeg), Some(&source), video_id, from_s, to_s, framing, out_w as f64, out_h as f64)?
        .ok_or_else(|| {
            Error::Invalid(format!(
                "video #{video_id} already has the shape of {out_w}x{out_h} (or no size): nothing to crop"
            ))
        })?;
    // Stills at a viewable size: 540 px tall, same shape as the output.
    let sh = 540u32;
    let sw = (((sh as f64) * out_w as f64 / out_h as f64 / 2.0).round() as u32 * 2).max(2);
    let times = still_times(&plan, to_s - from_s, max_stills);
    let stem = format!("crop_v{video_id}_{}", plan.layout.as_str());
    let files = render_stills(ffmpeg, &source, &plan, from_s, &times, sw, sh, dir, &stem)?;
    Ok((plan, times.into_iter().zip(files).collect()))
}

/// A clip's planned framing, addressed the way the timeline addresses clips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipFraming {
    pub beat_index: usize,
    pub clip_index: usize,
    pub video_id: i64,
    pub in_s: f64,
    pub plan: FramePlan,
}

/// The framing of every framed clip in a script (clips without framing, or the same shape as
/// their footage, are left out). Needs the script's width and height.
pub fn plans_for_script(db: &Db, ffmpeg: Option<&Path>, script: &Script) -> Result<Vec<ClipFraming>, Error> {
    let (Some(out_w), Some(out_h)) = (script.width, script.height) else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for (bi, beat) in script.beats.iter().enumerate() {
        for (ci, clip) in beat.clips.iter().enumerate() {
            let Some(fr) = effective(script, clip) else { continue };
            let path: Option<String> = db
                .conn
                .query_row("SELECT MIN(path) FROM video_files WHERE video_id = ?1", [clip.video_id], |r| r.get(0))
                .optional()?
                .flatten();
            let path = path.map(PathBuf::from);
            if let Some(plan) = plan_clip(
                db,
                ffmpeg,
                path.as_deref(),
                clip.video_id,
                clip.in_s,
                clip.out_s,
                fr,
                out_w as f64,
                out_h as f64,
            )? {
                out.push(ClipFraming {
                    beat_index: bi,
                    clip_index: ci,
                    video_id: clip.video_id,
                    in_s: clip.in_s,
                    plan,
                });
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = 1280.0;
    const H: f64 = 720.0;

    fn face(t: f64, cx: f64, cy: f64, h: f64) -> FaceSample {
        FaceSample { t, cx, cy, w: h * 0.75 * H / W, h, eye_y: cy - h * 0.1, cut_off: false }
    }

    /// A face sampled at 5 fps over `a..b`, at `cx(t)`.
    fn track(a: f64, b: f64, cy: f64, h: f64, cx: impl Fn(f64) -> f64) -> Vec<FaceSample> {
        let n = ((b - a) * SAMPLE_HZ).round() as usize;
        (0..n).map(|i| a + i as f64 / SAMPLE_HZ).map(|t| face(t, cx(t), cy, h)).collect()
    }

    fn input(duration: f64, subjects: Vec<Subject>, turns: Vec<Turn>) -> FramingInput {
        FramingInput {
            src_w: W,
            src_h: H,
            duration,
            subjects,
            turns,
            content: Rect { x: 0.0, y: 0.0, w: W, h: H },
            origin_s: 0.0,
        }
    }

    fn two_hosts(duration: f64) -> Vec<Subject> {
        vec![
            Subject { person_id: Some(1), samples: track(0.0, duration, 0.4, 0.2, |_| 0.25) },
            Subject { person_id: Some(2), samples: track(0.0, duration, 0.4, 0.2, |_| 0.75) },
        ]
    }

    const FOLLOW: Framing =
        Framing { layout: Layout::FollowSpeaker, person_id: None, box_: None, overrides: Vec::new() };

    #[test]
    fn old_scripts_parse_and_framing_round_trips() {
        let old = r#"{"title":"t","beats":[{"id":"a","purpose":"p","clips":[{"video_id":1,"in_s":0,"out_s":2}]}]}"#;
        let s: Script = serde_json::from_str(old).unwrap();
        assert!(s.framing.is_none() && s.beats[0].clips[0].framing.is_none());
        let back = serde_json::to_string(&s).unwrap();
        assert!(!back.contains("framing"), "no framing keys written for old scripts: {back}");

        let new = r#"{"title":"t","framing":{"layout":"stacked"},"beats":[{"id":"a","purpose":"p",
            "clips":[{"video_id":1,"in_s":0,"out_s":2,"framing":{"layout":"fixed","box":{"x":0.1,"y":0.1,"w":0.3,"h":0.5}}}]}]}"#;
        let s: Script = serde_json::from_str(new).unwrap();
        assert_eq!(s.framing.as_ref().unwrap().layout, Layout::Stacked);
        let clip = &s.beats[0].clips[0];
        assert_eq!(effective(&s, clip).unwrap().layout, Layout::Fixed, "clip overrides the script");
        assert_eq!(clip.framing.as_ref().unwrap().box_.unwrap().w, 0.3);
        let json = serde_json::to_value(&s).unwrap();
        assert_eq!(json["beats"][0]["clips"][0]["framing"]["box"]["h"], 0.5);
    }

    #[test]
    fn same_shape_needs_no_crop() {
        assert!(!needs_crop(1920.0, 1080.0, 1280.0, 720.0));
        assert!(needs_crop(1920.0, 1080.0, 1080.0, 1920.0));
        assert!(needs_crop(1920.0, 1080.0, 1080.0, 1080.0));
    }

    #[test]
    fn crop_has_output_shape_face_share_and_stays_inside() {
        // A small face (0.08 of the height) in the right half.
        let subj = vec![Subject { person_id: None, samples: track(0.0, 6.0, 0.6, 0.08, |_| 0.8) }];
        let p = plan(&input(6.0, subj, vec![]), &FOLLOW, 1080.0, 1920.0);
        let c = &p.segments[0].paths[0];
        assert!((c.w / c.h - 1080.0 / 1920.0).abs() < 0.02, "9:16 crop: {}x{}", c.w, c.h);
        // Face 57.6 px; MIN_CROP_FRAC keeps the crop at 216 px or more: face share ≤ 0.28.
        assert!(p.quality.face_frac >= 0.2 && p.quality.face_frac <= 0.28, "face share {}", p.quality.face_frac);
        for k in &c.keyframes {
            assert!(k.x >= 0.0 && k.x + c.w <= W + 0.01 && k.y >= 0.0 && k.y + c.h <= H + 0.01, "{k:?}");
        }
        // Face centred horizontally, eyes near the upper third.
        let (x, y) = c.at(3.0);
        let s = face(0.0, 0.8, 0.6, 0.08);
        assert!(((x + c.w / 2.0) - 0.8 * W).abs() < 2.0);
        assert!(((s.eye_y * H - y) / c.h - EYE_LINE).abs() < 0.02);
    }

    #[test]
    fn big_face_uses_full_height_and_stays_in_its_tile() {
        // Side by side tiles; faces 0.44 of the height: the crop can't zoom out past the frame.
        let subjects = vec![
            Subject { person_id: None, samples: track(0.0, 4.0, 0.45, 0.44, |_| 0.27) },
            Subject { person_id: None, samples: track(0.0, 4.0, 0.4, 0.46, |_| 0.73) },
        ];
        let p = plan(&input(4.0, subjects, vec![]), &FOLLOW, 1080.0, 1920.0);
        let c = &p.segments[0].paths[0];
        assert_eq!(c.h, H);
        let (x, _) = c.at(1.0);
        let followed = p.segments[0].paths[0].shots[0].subject.unwrap();
        let mid = 0.5 * W;
        if p.subjects[followed].median_cx < 0.5 {
            assert!(x + c.w <= mid + 1.0, "crop {x}+{} leaks into the right tile", c.w);
        } else {
            assert!(x >= mid - 1.0, "crop at {x} leaks into the left tile");
        }
    }

    #[test]
    fn content_rect_keeps_black_bars_out() {
        let mut inp =
            input(4.0, vec![Subject { person_id: None, samples: track(0.0, 4.0, 0.5, 0.3, |_| 0.3) }], vec![]);
        inp.content = Rect { x: 0.0, y: 90.0, w: W, h: 540.0 };
        let p = plan(&inp, &FOLLOW, 1080.0, 1920.0);
        let c = &p.segments[0].paths[0];
        assert!(c.h <= 540.0);
        let (_, y) = c.at(1.0);
        assert!(y >= 90.0 && y + c.h <= 630.0 + 0.01, "y {y} h {}", c.h);
    }

    #[test]
    fn dead_zone_ignores_small_drift() {
        // Jitter of ±1% of the width: well inside the dead zone.
        let samples = track(0.0, 8.0, 0.5, 0.2, |t| 0.5 + 0.01 * (t * 7.0).sin());
        let p = plan(&input(8.0, vec![Subject { person_id: None, samples }], vec![]), &FOLLOW, 1080.0, 1920.0);
        assert_eq!(p.segments[0].paths[0].keyframes.len(), 1, "{:?}", p.segments[0].paths[0].keyframes);
    }

    #[test]
    fn drift_eases_without_jumps() {
        // The face walks from 0.4 to 0.6 of the width between 2 s and 4 s.
        let samples = track(0.0, 10.0, 0.5, 0.2, |t| 0.4 + 0.2 * ((t - 2.0) / 2.0).clamp(0.0, 1.0));
        let p = plan(&input(10.0, vec![Subject { person_id: None, samples }], vec![]), &FOLLOW, 1080.0, 1920.0);
        let c = &p.segments[0].paths[0];
        assert!(c.keyframes.len() > 1, "it moves");
        assert!(c.keyframes.iter().all(|k| !k.cut));
        // No step bigger than an eased move allows between two frames at 30 fps.
        let mut prev = c.at(0.0).0;
        let mut t = 0.0;
        while t < 10.0 {
            t += 1.0 / 30.0;
            let x = c.at(t).0;
            assert!((x - prev).abs() < 0.12 * c.w, "jump at {t}: {prev} -> {x}");
            prev = x;
        }
        // Ends with the face re-centred.
        let end = c.at(9.9).0 + c.w / 2.0;
        assert!((end - 0.6 * W).abs() < DEAD_ZONE_X * c.w + 1.0, "end centre {end}");
    }

    #[test]
    fn hard_cut_on_speaker_change() {
        let turns = vec![Turn { start: 0.0, end: 5.0, subject: 0 }, Turn { start: 5.2, end: 12.0, subject: 1 }];
        let p = plan(&input(12.0, two_hosts(12.0), turns), &FOLLOW, 1080.0, 1920.0);
        assert!(p.used_turns);
        let c = &p.segments[0].paths[0];
        let cuts: Vec<&Keyframe> = c.keyframes.iter().filter(|k| k.cut).collect();
        assert_eq!(cuts.len(), 1, "{:?}", c.keyframes);
        assert!((cuts[0].t - 5.2).abs() < 1e-6);
        assert!(c.at(5.1).0 + c.w / 2.0 < 0.5 * W && c.at(5.2).0 + c.w / 2.0 > 0.5 * W);
        // The expression steps at the cut.
        assert!(ffmpeg_expr(&c.keyframes, |k| k.x).contains("gte(t,5.2000)"));
    }

    #[test]
    fn backchannels_and_short_shots_do_not_cut() {
        let turns = vec![
            Turn { start: 0.0, end: 4.0, subject: 0 },
            Turn { start: 4.1, end: 4.6, subject: 1 }, // "sí"
            Turn { start: 4.7, end: 9.0, subject: 0 },
            Turn { start: 9.0, end: 9.9, subject: 1 }, // "claro"
            Turn { start: 10.0, end: 11.0, subject: 0 },
        ];
        let shots = build_shots(&turns, 11.0, Some(0));
        assert_eq!(shots.len(), 1, "{shots:?}");
        // A real answer cuts, but not within MIN_SHOT_S of the last cut or the end.
        let turns = vec![
            Turn { start: 0.0, end: 0.8, subject: 0 },
            Turn { start: 0.8, end: 6.0, subject: 1 },
            Turn { start: 6.0, end: 7.0, subject: 0 },
            Turn { start: 7.0, end: 9.2, subject: 1 },
            Turn { start: 9.2, end: 10.0, subject: 0 },
        ];
        let shots = build_shots(&turns, 10.0, Some(0));
        assert_eq!(shots.len(), 1, "opening backchannel and closing words don't cut: {shots:?}");
        assert_eq!(shots[0].subject, Some(1));
        let turns = vec![
            Turn { start: 0.0, end: 3.0, subject: 0 },
            Turn { start: 3.0, end: 6.0, subject: 1 },
            Turn { start: 6.0, end: 9.0, subject: 0 },
        ];
        let shots = build_shots(&turns, 9.5, Some(0));
        assert_eq!(shots.iter().map(|s| s.subject.unwrap()).collect::<Vec<_>>(), vec![0, 1, 0]);
        assert!(shots.windows(2).all(|w| w[1].start - w[0].start >= MIN_SHOT_S));
    }

    #[test]
    fn holds_through_missing_detections() {
        // Seen 0–3 s, lost 3–6 s, back 6–9 s at the same place.
        let mut samples = track(0.0, 3.0, 0.5, 0.2, |_| 0.3);
        samples.extend(track(6.0, 9.0, 0.5, 0.2, |_| 0.3));
        let p = plan(&input(9.0, vec![Subject { person_id: None, samples }], vec![]), &FOLLOW, 1080.0, 1920.0);
        let c = &p.segments[0].paths[0];
        assert_eq!(c.keyframes.len(), 1, "held: {:?}", c.keyframes);
        assert!((p.quality.longest_no_face_s - 3.0).abs() < 0.05, "{:?}", p.quality);
        assert_eq!(p.quality.no_face_at, Some(2.8));
    }

    #[test]
    fn no_faces_fit_the_whole_picture_over_a_blur() {
        let p = plan(&input(5.0, vec![], vec![]), &FOLLOW, 1080.0, 1920.0);
        assert_eq!(p.segments.len(), 1);
        assert_eq!((p.segments[0].layout, p.segments[0].scene), (Layout::Fit, Scene::NoFace));
        assert!(p.segments[0].paths.is_empty());
        let f = p.ffmpeg_filter(1080, 1920);
        assert!(f.contains("boxblur") && f.contains("overlay=x=0:y=657"), "{f}");
        // Black bars stay out of the fitted picture.
        let mut inp = input(5.0, vec![], vec![]);
        inp.content = Rect { x: 0.0, y: 90.0, w: W, h: 540.0 };
        let f = plan(&inp, &FOLLOW, 1080.0, 1920.0).ffmpeg_filter(1080, 1920);
        assert!(f.starts_with("crop=1280:540:0:90,split=2"), "{f}");
    }

    #[test]
    fn without_turns_follows_the_most_prominent_face_and_never_switches() {
        let subjects = vec![
            Subject { person_id: None, samples: track(0.0, 10.0, 0.5, 0.18, |_| 0.25) },
            Subject { person_id: None, samples: track(0.0, 10.0, 0.5, 0.24, |_| 0.75) },
        ];
        let p = plan(&input(10.0, subjects, vec![]), &FOLLOW, 1080.0, 1920.0);
        assert!(!p.used_turns);
        assert_eq!(p.segments[0].paths[0].shots.len(), 1);
        assert_eq!(p.segments[0].paths[0].shots[0].subject, Some(1));
        assert!(p.segments[0].paths[0].keyframes.iter().all(|k| !k.cut));
    }

    #[test]
    fn pinned_person_is_followed_regardless_of_turns() {
        let turns = vec![Turn { start: 0.0, end: 5.0, subject: 0 }, Turn { start: 5.0, end: 10.0, subject: 1 }];
        let fr = Framing { layout: Layout::FollowSpeaker, person_id: Some(2), box_: None, overrides: Vec::new() };
        let p = plan(&input(10.0, two_hosts(10.0), turns), &fr, 1080.0, 1920.0);
        assert_eq!(p.segments[0].paths[0].shots.len(), 1);
        assert_eq!(p.subjects[p.segments[0].paths[0].shots[0].subject.unwrap()].person_id, Some(2));
    }

    #[test]
    fn slots_group_tracks_by_seat() {
        // Left seat: two tracks one after the other (detection lost in between); right seat: one.
        // Two faces stacked in one column (a cam above another) stay apart.
        let mut boxes = Vec::new();
        for s in track(0.0, 4.0, 0.4, 0.2, |_| 0.25) {
            boxes.push((1, None, s));
        }
        for s in track(5.0, 9.0, 0.42, 0.2, |_| 0.27) {
            boxes.push((2, None, s));
        }
        for s in track(0.0, 9.0, 0.4, 0.2, |_| 0.75) {
            boxes.push((3, None, s));
        }
        for s in track(0.0, 9.0, 0.8, 0.2, |_| 0.76) {
            boxes.push((4, None, s));
        }
        let subjects = group_subjects(boxes);
        assert_eq!(
            subjects.len(),
            3,
            "{:?}",
            subjects.iter().map(|s| (s.median_cx(), s.median_cy())).collect::<Vec<_>>()
        );
        assert!(subjects[0].median_cx() < 0.3 && subjects[0].samples.len() == 40);
        // By person when named, whatever the position.
        let mut boxes = Vec::new();
        for s in track(0.0, 2.0, 0.4, 0.2, |_| 0.25) {
            boxes.push((1, Some(7), s));
        }
        for s in track(2.0, 4.0, 0.4, 0.2, |_| 0.75) {
            boxes.push((2, Some(7), s));
        }
        let subjects = group_subjects(boxes);
        assert_eq!(subjects.len(), 1);
        assert_eq!(subjects[0].person_id, Some(7));
    }

    #[test]
    fn tiny_faces_and_passers_by_are_not_framed() {
        let subjects = vec![
            Subject { person_id: None, samples: track(0.0, 10.0, 0.3, 0.23, |_| 0.8) },
            Subject { person_id: None, samples: track(0.0, 10.0, 0.6, 0.08, |_| 0.3) }, // on a slide
            Subject { person_id: None, samples: track(4.0, 4.6, 0.5, 0.25, |_| 0.5) },  // walks past
        ];
        let inp = input(10.0, subjects, vec![]);
        assert_eq!(kept_subjects(&inp), vec![0]);
        // A big face seen only briefly (the layout changes at the end) doesn't push out the
        // small faces that are there all along.
        let subjects = vec![
            Subject { person_id: None, samples: track(0.0, 9.0, 0.85, 0.07, |_| 0.72) },
            Subject { person_id: None, samples: track(0.0, 9.0, 0.85, 0.07, |_| 0.9) },
            Subject { person_id: None, samples: track(9.0, 10.0, 0.4, 0.44, |_| 0.25) },
        ];
        assert_eq!(kept_subjects(&input(10.0, subjects, vec![])), vec![0, 1]);
        // Big faces in a thumbnail shown for a third of the clip lose to the cams that stay.
        let subjects = vec![
            Subject { person_id: None, samples: track(0.0, 10.0, 0.85, 0.07, |_| 0.72) },
            Subject { person_id: None, samples: track(3.0, 6.5, 0.5, 0.26, |_| 0.5) },
        ];
        assert_eq!(kept_subjects(&input(10.0, subjects, vec![])), vec![0]);
    }

    #[test]
    fn stacked_puts_speaker_on_top_and_listener_below() {
        let turns = vec![Turn { start: 0.0, end: 5.0, subject: 0 }, Turn { start: 5.0, end: 10.0, subject: 1 }];
        let fr = Framing { layout: Layout::Stacked, person_id: None, box_: None, overrides: Vec::new() };
        let p = plan(&input(10.0, two_hosts(10.0), turns), &fr, 1080.0, 1920.0);
        assert_eq!(p.layout, Layout::Stacked);
        let [top, bottom] = p.segments[0].paths.as_slice() else { panic!("two paths") };
        assert!((top.w / top.h - 1080.0 / 960.0).abs() < 0.03, "half-height aspect {}x{}", top.w, top.h);
        assert_eq!((top.w, top.h), (bottom.w, bottom.h));
        let left = |c: &CropPath, t: f64| c.at(t).0 + c.w / 2.0 < 0.5 * W;
        assert!(left(top, 2.0) && !left(bottom, 2.0));
        assert!(!left(top, 7.0) && left(bottom, 7.0));
        let f = p.ffmpeg_filter(304, 540);
        assert!(f.contains("vstack") && f.contains("scale=304:270"), "{f}");
        // One person on screen: stacked falls back to a single crop.
        let one = vec![Subject { person_id: Some(1), samples: track(0.0, 10.0, 0.4, 0.2, |_| 0.3) }];
        let p = plan(&input(10.0, one, vec![]), &fr, 1080.0, 1920.0);
        assert_eq!((p.layout, p.segments[0].paths.len()), (Layout::FollowSpeaker, 1));
    }

    /// Two hosts side by side 0–10 s, then a screen share with their camera tiles in the bottom
    /// right corner 10–20 s; the people stage named each host's tile and big face as one person.
    fn podcast() -> (Vec<Subject>, Vec<Turn>) {
        let mut a = track(0.0, 10.0, 0.42, 0.45, |_| 0.25);
        a.extend(track(10.0, 20.0, 0.88, 0.076, |_| 0.72));
        let mut b = track(0.0, 10.0, 0.42, 0.45, |_| 0.75);
        b.extend(track(10.0, 20.0, 0.88, 0.07, |_| 0.90));
        let subjects = vec![Subject { person_id: Some(1), samples: a }, Subject { person_id: Some(2), samples: b }];
        (subjects, vec![Turn { start: 0.0, end: 20.0, subject: 0 }])
    }

    #[test]
    fn a_screen_share_gets_its_own_stretch_layout_crop_and_filter_branch() {
        let (subjects, turns) = podcast();
        let p = plan(&input(20.0, subjects.clone(), turns.clone()), &FOLLOW, 1080.0, 1920.0);
        let kinds: Vec<(Scene, Layout)> = p.segments.iter().map(|s| (s.scene, s.layout)).collect();
        assert_eq!(kinds, vec![(Scene::TwoShot, Layout::FollowSpeaker), (Scene::ScreenShare, Layout::Screen)]);
        let (two, share) = (&p.segments[0], &p.segments[1]);
        assert!((two.end - 10.0).abs() < 0.41 && two.end == share.start && share.end == 20.0);
        // Each stretch sized for its own faces: full height for the big ones, a tight crop of
        // the speaker's tile under the screen, which stays clear of the tiles.
        assert_eq!(two.paths[0].h, H);
        let [screen, tile] = share.paths.as_slice() else { panic!("screen over speaker") };
        let (sx, sy) = screen.at(15.0);
        assert!(sy + screen.h <= 0.80 * H, "screen crop {sx},{sy} {}x{} reaches the tiles", screen.w, screen.h);
        assert!(tile.h < 0.3 * H);
        let (tx, _) = tile.at(15.0);
        assert!(((tx + tile.w / 2.0) / W - 0.72).abs() < 0.03, "the speaker's tile: {tx}");
        assert_eq!(p.quality.cut_off_share, 0.0, "no face cut off: {:?}", p.quality);
        assert!(std::ptr::eq(p.segment_at(15.0).unwrap(), share));
        // One branch per stretch, each on its own clock, joined back.
        let f = p.ffmpeg_filter(1080, 1920);
        assert!(f.starts_with("split=2[s0][s1];[s0]trim=end="), "{f}");
        assert!(f.contains(&format!("[s1]trim=start={:.4},setpts=PTS-STARTPTS,split=2[fa1][fb1]", share.start)), "{f}");
        assert!(f.ends_with("[v0][v1]concat=n=2:v=1:a=0,setsar=1"), "{f}");
        assert!(!f.contains(&format!("gte(t,{:.4})", share.start)), "the share branch starts at 0: {f}");
        // An override frames the share whole instead.
        let fr = FOLLOW.with_override(10.0, 20.0, Layout::Fit);
        let p = plan(&input(20.0, subjects, turns), &fr, 1080.0, 1920.0);
        assert_eq!(p.segments.last().map(|s| s.layout), Some(Layout::Fit));
        assert!(p.ffmpeg_filter(1080, 1920).contains("boxblur"));
    }

    #[test]
    fn stills_freeze_the_stretch_they_fall_in() {
        let (subjects, turns) = podcast();
        let p = plan(&input(20.0, subjects, turns), &FOLLOW, 1080.0, 1920.0);
        let still = p.frozen_at(15.0);
        assert_eq!(still.segments.len(), 1);
        assert_eq!(still.segments[0].layout, Layout::Screen);
        assert!(still.segments[0].paths.iter().all(|p| p.keyframes.len() == 1));
        let times = still_times(&p, 20.0, 4);
        assert!(times.iter().any(|t| (t - (p.segments[1].start + 0.3)).abs() < 1e-6), "{times:?}");
        assert_eq!(*times.last().unwrap(), 19.5);
    }

    #[test]
    fn a_frame_view_is_the_crops_the_render_uses_at_that_moment() {
        let (subjects, turns) = podcast();
        let p = plan(&input(20.0, subjects.clone(), turns.clone()), &FOLLOW, 1080.0, 1920.0);
        // The first stretch: one crop following the speaker, where the path is then.
        let first = &p.segments[0];
        let v = p.view_at(1.0);
        assert_eq!(v.crops.len(), 1, "{v:?}");
        let (x, y) = first.paths[0].at(1.0);
        assert_eq!(v.crops[0], Rect { x, y, w: first.paths[0].w, h: first.paths[0].h });
        assert_eq!((v.out_w, v.out_h, v.src_w, v.src_h), (1080.0, 1920.0, p.src_w, p.src_h));
        // The screen share: two bands, screen over the speaker's tile.
        assert_eq!(p.view_at(15.0).crops.len(), 2);
        // Fit: no crop, the whole picture.
        let fr = FOLLOW.with_override(10.0, 20.0, Layout::Fit);
        let fit = plan(&input(20.0, subjects, turns), &fr, 1080.0, 1920.0);
        assert!(fit.view_at(15.0).crops.is_empty());
    }

    #[test]
    fn fixed_box_and_fixed_person() {
        let fr = Framing {
            layout: Layout::Fixed,
            person_id: None,
            box_: Some(NormBox { x: 0.6, y: 0.1, w: 0.2, h: 0.6 }),
            overrides: Vec::new(),
        };
        let p = plan(&input(5.0, two_hosts(5.0), vec![]), &fr, 1080.0, 1080.0);
        let c = &p.segments[0].paths[0];
        assert_eq!(c.keyframes.len(), 1);
        assert_eq!(c.w, c.h, "1:1");
        assert!(c.h >= 0.6 * H - 2.0);
        let (x, _) = c.at(0.0);
        assert!(((x + c.w / 2.0) - 0.7 * W).abs() < 2.0);
        let fr = Framing { layout: Layout::Fixed, person_id: Some(1), box_: None, overrides: Vec::new() };
        let p = plan(&input(5.0, two_hosts(5.0), vec![]), &fr, 1080.0, 1920.0);
        assert_eq!(p.segments[0].paths[0].keyframes.len(), 1);
        assert!(p.segments[0].paths[0].at(0.0).0 + p.segments[0].paths[0].w / 2.0 < 0.5 * W);
    }

    #[test]
    fn expression_matches_value_at() {
        let kfs = vec![
            Keyframe { t: 0.0, x: 100.0, y: 10.0, cut: false },
            Keyframe { t: 1.0, x: 100.0, y: 10.0, cut: false },
            Keyframe { t: 2.0, x: 200.0, y: 30.0, cut: false },
            Keyframe { t: 4.0, x: 50.0, y: 0.0, cut: true },
            Keyframe { t: 5.0, x: 50.0, y: 0.0, cut: false },
            Keyframe { t: 6.0, x: 20.0, y: 5.0, cut: false },
        ];
        // Evaluate our own expression format by hand: x0 + ramps + steps.
        let eval = |t: f64| {
            let mut x = kfs[0].x;
            for p in kfs.windows(2) {
                let d = p[1].x - p[0].x;
                if p[1].cut {
                    x += if t >= p[1].t { d } else { 0.0 };
                } else if d != 0.0 {
                    x += d / (p[1].t - p[0].t) * (t.clamp(p[0].t, p[1].t) - p[0].t);
                }
            }
            x
        };
        for i in 0..70 {
            let t = i as f64 * 0.1;
            assert!((eval(t) - value_at(&kfs, t).0).abs() < 1e-6, "t={t}");
        }
        let e = ffmpeg_expr(&kfs, |k| k.x);
        assert!(e.starts_with("100.0+100.0000*(clip(t,1.0000,2.0000)-1.0000)-150.00*gte(t,4.0000)"), "{e}");
    }
}
