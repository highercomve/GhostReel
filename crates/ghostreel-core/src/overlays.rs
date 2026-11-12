//! Burnt-in text: on-screen titles and speech captions, laid out once for both the render (one
//! ASS file for libass) and the editor's live viewer, so the two put every line in the same place.
//!
//! Everything is in output-frame units: positions are 0..1 of the frame, sizes a fraction of its
//! height, so the same layout holds for the 540p preview, a full-size export and the viewer.

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::captions::Cue;
use crate::db::Db;
use crate::script::{Script, TextStyle};

/// Fonts the style menu offers. Each is on Windows and macOS, and fontconfig maps it to a
/// metric-compatible Liberation font on Linux, so lines break and sit the same everywhere.
pub const FONTS: &[&str] = &["Arial", "Times New Roman", "Courier New", "Verdana", "Georgia", "Impact"];

/// libass sizes a font so that its cell (ascent + descent) is the font size; CSS sizes the em.
/// Cell height in ems, per family: the browser sets `font-size = size / cell_em`.
pub fn cell_em(font: &str) -> f64 {
    match font.to_ascii_lowercase().as_str() {
        "arial" | "helvetica" | "liberation sans" | "arimo" => 1.117,
        "times new roman" | "liberation serif" | "tinos" => 1.107,
        "courier new" | "liberation mono" | "cousine" => 1.133,
        "verdana" => 1.215,
        "georgia" => 1.136,
        "impact" => 1.220,
        "dejavu sans" => 1.164,
        _ => 1.15,
    }
}

/// A text style with every value decided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedStyle {
    pub font: String,
    /// Line height / frame height.
    pub size: f64,
    pub bold: bool,
    pub uppercase: bool,
    pub color: String,
    pub outline_color: String,
    /// Fraction of the font size.
    pub outline: f64,
    /// Fraction of the font size.
    pub shadow: f64,
    #[serde(rename = "box")]
    pub boxed: bool,
    pub box_color: String,
    pub box_opacity: f64,
    pub align: String,
    pub max_width: f64,
    /// Padding around each line when `box` is on, as a fraction of the font size.
    pub box_pad: f64,
    /// Cell height in ems for this font (see [`cell_em`]).
    pub cell_em: f64,
}

/// Box padding, in font sizes: the old title box's 12 px around 45 px type.
const BOX_PAD: f64 = 0.25;

fn clean_color(c: Option<&str>, default: &str) -> String {
    match c.map(str::trim) {
        Some(c) if c.len() == 7 && c.starts_with('#') && c[1..].chars().all(|ch| ch.is_ascii_hexdigit()) => {
            c.to_ascii_lowercase()
        }
        _ => default.to_string(),
    }
}

fn clean_font(f: Option<&str>) -> String {
    // A comma would end the field in an ASS style line; braces and backslashes have no place
    // in a family name either.
    let f: String = f.unwrap_or("").chars().filter(|c| !matches!(c, ',' | '{' | '}' | '\\' | '\n' | '\r')).collect();
    let f = f.trim();
    if f.is_empty() { "Arial".to_string() } else { f.to_string() }
}

fn finite(v: Option<f64>) -> Option<f64> {
    v.filter(|v| v.is_finite())
}

/// Fill a style from defaults, keeping every value within sane bounds.
pub fn resolve(style: &TextStyle) -> ResolvedStyle {
    let font = clean_font(style.font.as_deref());
    let align = match style.align.as_deref() {
        Some("left") => "left",
        Some("right") => "right",
        _ => "center",
    };
    ResolvedStyle {
        cell_em: cell_em(&font),
        font,
        size: finite(style.size).unwrap_or(0.055).clamp(0.01, 0.4),
        bold: style.bold.unwrap_or(false),
        uppercase: style.uppercase.unwrap_or(false),
        color: clean_color(style.color.as_deref(), "#ffffff"),
        outline_color: clean_color(style.outline_color.as_deref(), "#000000"),
        outline: finite(style.outline).unwrap_or(0.0).clamp(0.0, 0.5),
        shadow: finite(style.shadow).unwrap_or(0.0).clamp(0.0, 0.5),
        boxed: style.boxed.unwrap_or(false),
        box_color: clean_color(style.box_color.as_deref(), "#000000"),
        box_opacity: finite(style.box_opacity).unwrap_or(0.5).clamp(0.0, 1.0),
        align: align.to_string(),
        max_width: finite(style.max_width).unwrap_or(0.9).clamp(0.1, 1.0),
        box_pad: BOX_PAD,
    }
}

/// What a title looks like when nothing says otherwise: the old drawtext title — white on a
/// half-black box, `min(h/12, w/14)` type, centred near the top.
pub fn title_defaults(width: u32, height: u32) -> TextStyle {
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let em = (1.0 / 12.0f64).min(w / (14.0 * h));
    TextStyle {
        font: Some("Arial".into()),
        size: Some(em * cell_em("Arial")),
        bold: Some(false),
        uppercase: Some(false),
        color: Some("#ffffff".into()),
        outline_color: Some("#000000".into()),
        outline: Some(0.0),
        shadow: Some(0.0),
        boxed: Some(true),
        box_color: Some("#000000".into()),
        box_opacity: Some(0.5),
        align: Some("center".into()),
        max_width: Some(0.9),
    }
}

/// What a caption looks like when nothing says otherwise: the old captions ASS — bold white
/// Arial, a 3 px outline and a soft shadow on 30 px type. A vertical frame sizes off its width.
pub fn caption_defaults(width: u32, height: u32, legacy_scale: f64) -> TextStyle {
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let scale = if legacy_scale > 0.2 && legacy_scale < 4.0 { legacy_scale } else { 1.0 };
    let basis = if h > w { w * 0.95 } else { h };
    TextStyle {
        font: Some("Arial".into()),
        size: Some(basis * 0.055 * scale / h),
        bold: Some(true),
        uppercase: Some(false),
        color: Some("#ffffff".into()),
        outline_color: Some("#000000".into()),
        outline: Some(0.1),
        shadow: Some(0.05),
        boxed: Some(false),
        box_color: Some("#000000".into()),
        box_opacity: Some(0.6),
        align: Some("center".into()),
        max_width: Some(0.9),
    }
}

/// Which point of the text block `x`,`y` name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VAnchor {
    Middle,
    Bottom,
}

/// One piece of text on screen, laid out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Overlay {
    /// `title` or `caption`.
    pub kind: String,
    /// The beat a title belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beat_index: Option<usize>,
    /// On the cut's clock.
    pub start_s: f64,
    pub end_s: f64,
    /// Lines as drawn: uppercased when asked, wrapped to the style's width.
    pub lines: Vec<String>,
    pub x: f64,
    pub y: f64,
    pub v_anchor: VAnchor,
    pub style: ResolvedStyle,
    /// A caption's cue (anchors, edit state).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cue: Option<Cue>,
}

/// All burnt-in text of a cut, in the frame it is laid out for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Overlays {
    pub width: u32,
    pub height: u32,
    pub titles: Vec<Overlay>,
    /// Every caption cue, hidden ones included (`cue.hidden`); renders skip those.
    pub captions: Vec<Overlay>,
    /// The caption style and place, also when there are no cues to show it on.
    pub caption_style: ResolvedStyle,
    pub caption_x: f64,
    pub caption_y: f64,
    pub caption_v_anchor: VAnchor,
}

/// Average glyph advance in ems, for wrapping without a font engine. Both sides use this one
/// estimate, so the viewer and the render break lines at the same words.
fn avg_advance(style: &ResolvedStyle) -> f64 {
    let base = match style.font.to_ascii_lowercase().as_str() {
        "courier new" | "liberation mono" => 0.6,
        "verdana" => 0.58,
        "impact" => 0.5,
        "georgia" => 0.53,
        "times new roman" => 0.45,
        _ => 0.5,
    };
    let bold = if style.bold { 1.06 } else { 1.0 };
    let upper = if style.uppercase { 1.2 } else { 1.0 };
    base * bold * upper
}

/// Most characters a line of this style holds in a frame of this shape.
pub fn max_line_chars(style: &ResolvedStyle, width: u32, height: u32) -> usize {
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let em_px = style.size * h / style.cell_em;
    ((style.max_width * w) / (em_px * avg_advance(style))).floor().max(6.0) as usize
}

fn greedy(words: &[&str], max: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut line = String::new();
    for word in words {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > max {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// Word wrap of each paragraph to at most `max` characters a line (a longer word keeps its own
/// line), balanced like libass's smart wrap: as many lines as greedy wrapping needs, but as even
/// as they can be, so no line is left with one word.
pub fn wrap_lines(text: &str, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.lines() {
        let words: Vec<&str> = para.split_whitespace().collect();
        let lines = greedy(&words, max);
        if lines.len() <= 1 {
            out.extend(lines);
            continue;
        }
        let n = lines.len();
        let longest = words.iter().map(|w| w.chars().count()).max().unwrap_or(1);
        let mut width = max;
        while width > longest && greedy(&words, width - 1).len() <= n {
            width -= 1;
        }
        out.extend(greedy(&words, width));
    }
    out
}

fn lines_for(text: &str, style: &ResolvedStyle, width: u32, height: u32) -> Vec<String> {
    let text = if style.uppercase { text.to_uppercase() } else { text.to_string() };
    wrap_lines(&text, max_line_chars(style, width, height))
}

/// The frame a script is laid out in: its own size, else 1920x1080.
pub fn frame_of(script: &Script) -> (u32, u32) {
    let w = script.width.filter(|w| *w > 0).unwrap_or(1920) as u32;
    let h = script.height.filter(|h| *h > 0).unwrap_or(1080) as u32;
    (w, h)
}

/// Where captions go and how they look, for a frame.
fn caption_placement(script: &Script, width: u32, height: u32) -> (ResolvedStyle, f64, f64, VAnchor) {
    let cap = script.captions.clone().unwrap_or_default();
    let style = resolve(&caption_defaults(width, height, cap.size).merged(&cap.style));
    let portrait = height > width;
    // A stacked cut puts its captions on the seam between the two faces.
    let at_seam =
        portrait && script.framing.as_ref().is_some_and(|f| f.layout == crate::framing::Layout::Stacked);
    let default_x = match style.align.as_str() {
        "left" => (1.0 - style.max_width) / 2.0,
        "right" => 1.0 - (1.0 - style.max_width) / 2.0,
        _ => 0.5,
    };
    let x = finite(cap.x).unwrap_or(default_x).clamp(0.0, 1.0);
    match finite(cap.y) {
        Some(y) => (style, x, y.clamp(0.0, 1.0), VAnchor::Bottom),
        None if at_seam || cap.position == "center" => (style, x, 0.5, VAnchor::Middle),
        // In 9:16 the bottom margin clears the platform's buttons (the lower fifth).
        None => (style, x, if portrait { 0.8 } else { 0.92 }, VAnchor::Bottom),
    }
}

/// Lay out a cut's titles (beat spans) and the given caption cues.
pub fn layout(script: &Script, cues: &[Cue]) -> Overlays {
    let (width, height) = frame_of(script);
    let base = title_defaults(width, height).merged(script.title_style.as_ref().unwrap_or(&TextStyle::default()));

    let mut titles = Vec::new();
    let mut t = 0.0;
    for (i, beat) in script.beats.iter().enumerate() {
        let dur: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();
        let (beat_start, beat_end) = (t, t + dur);
        t = beat_end;
        let Some(text) = beat.on_screen_text.as_deref().filter(|s| !s.trim().is_empty()) else { continue };
        if dur <= 0.0 {
            continue;
        }
        let lay = beat.title_layout.clone().unwrap_or_default();
        let style = resolve(&base.merged(lay.style.as_ref().unwrap_or(&TextStyle::default())));
        let rel_start = finite(lay.start_s).unwrap_or(0.0).clamp(0.0, (dur - 0.1).max(0.0));
        let rel_end = finite(lay.end_s).unwrap_or(dur).clamp(rel_start + 0.1, dur.max(rel_start + 0.1));
        let lines = lines_for(text, &style, width, height);
        let default_x = match style.align.as_str() {
            "left" => (1.0 - style.max_width) / 2.0,
            "right" => 1.0 - (1.0 - style.max_width) / 2.0,
            _ => 0.5,
        };
        // The old title's top sat at 8% of the height; centred on its first line.
        let default_y = 0.08 + style.size * lines.len().max(1) as f64 / 2.0;
        titles.push(Overlay {
            kind: "title".into(),
            beat_index: Some(i),
            start_s: beat_start + rel_start,
            end_s: (beat_start + rel_end).min(beat_end),
            lines,
            x: finite(lay.x).unwrap_or(default_x).clamp(0.0, 1.0),
            y: finite(lay.y).unwrap_or(default_y).clamp(0.0, 1.0),
            v_anchor: VAnchor::Middle,
            style,
            cue: None,
        });
    }

    let (caption_style, caption_x, caption_y, caption_v_anchor) = caption_placement(script, width, height);
    let max_chars = max_line_chars(&caption_style, width, height);
    let captions = cues
        .iter()
        .map(|cue| {
            let text = if caption_style.uppercase { cue.text.to_uppercase() } else { cue.text.clone() };
            // A cue's own line breaks stand unless a line is too wide for the frame; then the
            // whole cue is wrapped again to fit.
            let lines = if text.lines().any(|l| l.chars().count() > max_chars) {
                wrap_lines(&text.split_whitespace().collect::<Vec<_>>().join(" "), max_chars)
            } else {
                text.lines().map(str::to_string).collect()
            };
            Overlay {
                kind: "caption".into(),
                beat_index: None,
                start_s: cue.start_s,
                end_s: cue.end_s,
                lines,
                x: caption_x,
                y: caption_y,
                v_anchor: caption_v_anchor,
                style: caption_style.clone(),
                cue: Some(cue.clone()),
            }
        })
        .collect();

    Overlays { width, height, titles, captions, caption_style, caption_x, caption_y, caption_v_anchor }
}

/// Lay out a cut's titles and its speech captions (hidden cues included), read from the index.
pub fn for_script(db: &Db, script: &Script) -> Result<Overlays, Error> {
    let cues = crate::captions::all_cues_for_script(db, script)?;
    Ok(layout(script, &cues))
}

/// `#rrggbb` + opacity as an ASS colour, `&HAABBGGRR` (alpha 00 = solid).
fn ass_color(hex: &str, opacity: f64) -> String {
    let h = hex.trim_start_matches('#');
    let (r, g, b) = (&h[0..2], &h[2..4], &h[4..6]);
    let a = ((1.0 - opacity.clamp(0.0, 1.0)) * 255.0).round() as u8;
    format!("&H{a:02X}{}{}{}", b.to_ascii_uppercase(), g.to_ascii_uppercase(), r.to_ascii_uppercase())
}

/// ASS numpad alignment for an anchor: bottom row 1-3, middle row 4-6.
fn an(style: &ResolvedStyle, v: VAnchor) -> u8 {
    let col = match style.align.as_str() {
        "left" => 1,
        "right" => 3,
        _ => 2,
    };
    match v {
        VAnchor::Bottom => col,
        VAnchor::Middle => col + 3,
    }
}

/// Text made safe for an ASS event: `{` would open an override block and `\N`, `\n`, `\h` are
/// line codes, so braces are escaped and a backslash gets a word joiner after it.
pub fn escape_ass_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\u{2060}"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '\n' | '\r' => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

fn ass_style_line(name: &str, s: &ResolvedStyle, v: VAnchor, height: u32) -> String {
    let fs = s.size * height as f64;
    let (border_style, outline, shadow, outline_colour) = if s.boxed {
        (3, s.box_pad * fs, 0.0, ass_color(&s.box_color, s.box_opacity))
    } else {
        (1, s.outline * fs, s.shadow * fs, ass_color(&s.outline_color, 1.0))
    };
    format!(
        "Style: {name},{font},{fs:.2},{primary},&H000000FF,{outline_colour},&H80000000,{bold},0,0,0,100,100,0,0,{border_style},{outline:.2},{shadow:.2},{an},0,0,0,1",
        font = s.font,
        primary = ass_color(&s.color, 1.0),
        bold = if s.bold { -1 } else { 0 },
        an = an(s, v),
    )
}

/// One ASS file with every title and visible caption, each at an explicit `\pos`, for a frame of
/// `width` x `height` pixels. No fades: text is on for exactly its span.
pub fn to_ass(ov: &Overlays, width: u32, height: u32) -> String {
    use std::fmt::Write;
    let mut styles: Vec<(String, ResolvedStyle, VAnchor)> = Vec::new();
    let mut style_name = |s: &ResolvedStyle, v: VAnchor, prefix: &str| -> String {
        if let Some((n, _, _)) = styles.iter().find(|(n, st, va)| st == s && *va == v && n.starts_with(prefix)) {
            return n.clone();
        }
        let n = format!("{prefix}{}", styles.len());
        styles.push((n.clone(), s.clone(), v));
        n
    };
    let mut events = String::new();
    let visible_captions = ov.captions.iter().filter(|c| !c.cue.as_ref().is_some_and(|q| q.hidden));
    for (layer, o) in visible_captions.map(|o| (0, o)).chain(ov.titles.iter().map(|o| (1, o))) {
        if o.lines.is_empty() || o.end_s <= o.start_s {
            continue;
        }
        let name = style_name(&o.style, o.v_anchor, if o.kind == "title" { "Title" } else { "Caption" });
        let text = o.lines.iter().map(|l| escape_ass_text(l)).collect::<Vec<_>>().join("\\N");
        let (px, py) = (o.x * width as f64, o.y * height as f64);
        writeln!(
            &mut events,
            "Dialogue: {layer},{},{},{name},,0,0,0,,{{\\an{}\\pos({px:.1},{py:.1})}}{text}",
            crate::preview::format_ass_time(o.start_s),
            crate::preview::format_ass_time(o.end_s),
            an(&o.style, o.v_anchor),
        )
        .unwrap();
    }
    let mut ass = format!(
        "[Script Info]\n\
         ScriptType: v4.00+\n\
         PlayResX: {width}\n\
         PlayResY: {height}\n\
         WrapStyle: 2\n\
         ScaledBorderAndShadow: yes\n\n\
         [V4+ Styles]\n\
         Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    );
    for (name, s, v) in &styles {
        ass.push_str(&ass_style_line(name, s, *v, height));
        ass.push('\n');
    }
    ass.push_str("\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n");
    ass.push_str(&events);
    ass
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script::{Audio, Beat, CaptionStyle, ScriptClip, TitleLayout};

    fn script(w: i64, h: i64, text: Option<&str>) -> Script {
        Script {
            title: "t".into(),
            target_duration_s: None,
            fps: None,
            width: Some(w),
            height: Some(h),
            framing: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: text.map(Into::into),
                title_layout: None,
                clips: vec![ScriptClip {
                    video_id: 1,
                    in_s: 10.0,
                    out_s: 20.0,
                    audio: Audio::Source,
                    why: None,
                    tracks: None,
                    framing: None,
                }],
                bed: None,
                notes: None,
            }],
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
        }
    }

    fn cue(start: f64, end: f64, text: &str) -> Cue {
        Cue { start_s: start, end_s: end, text: text.into(), ..Cue::default() }
    }

    #[test]
    fn default_title_matches_the_old_drawtext_title() {
        let ov = layout(&script(1920, 1080, Some("Hola")), &[]);
        let t = &ov.titles[0];
        assert_eq!((t.start_s, t.end_s), (0.0, 10.0));
        assert!(t.style.boxed && t.style.box_opacity == 0.5);
        // min(h/12, w/14) em = 90 px at 1080 → a 100.5 px cell.
        assert!((t.style.size * 1080.0 - 90.0 * 1.117).abs() < 0.5, "{}", t.style.size);
        assert_eq!(t.x, 0.5);
        assert!(t.y > 0.08 && t.y < 0.2);
    }

    #[test]
    fn default_captions_match_the_old_captions_ass() {
        let ov = layout(&script(960, 540, None), &[cue(1.0, 2.0, "hola")]);
        let ass = to_ass(&ov, 960, 540);
        // 30 px bold white Arial, 3 px black outline, 1.5 px shadow, bottom centre at 92%.
        assert!(ass.contains("Style: Caption0,Arial,29.70,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2.97,1.49,2,"), "{ass}");
        assert!(ass.contains("Dialogue: 0,0:00:01.00,0:00:02.00,Caption0,,0,0,0,,{\\an2\\pos(480.0,496.8)}hola"), "{ass}");
        // A vertical frame keeps captions above the lower fifth.
        let v = layout(&script(1080, 1920, None), &[cue(1.0, 2.0, "hola")]);
        assert_eq!((v.captions[0].y, v.captions[0].v_anchor), (0.8, VAnchor::Bottom));
    }

    #[test]
    fn positions_styles_and_timing_reach_the_ass_file() {
        let mut s = script(1000, 1000, Some("Lower third"));
        s.beats[0].title_layout = Some(TitleLayout {
            start_s: Some(2.0),
            end_s: Some(5.0),
            x: Some(0.1),
            y: Some(0.75),
            style: Some(TextStyle {
                align: Some("left".into()),
                color: Some("#FFCC00".into()),
                boxed: Some(false),
                outline: Some(0.1),
                size: Some(0.05),
                bold: Some(true),
                uppercase: Some(true),
                ..TextStyle::default()
            }),
        });
        s.captions = Some(CaptionStyle {
            x: Some(0.5),
            y: Some(0.3),
            style: TextStyle { boxed: Some(true), box_color: Some("#112233".into()), box_opacity: Some(1.0), ..TextStyle::default() },
            ..CaptionStyle::default()
        });
        let ov = layout(&s, &[cue(3.0, 4.0, "dicho")]);
        let ass = to_ass(&ov, 1000, 1000);
        assert!(ass.contains("Dialogue: 1,0:00:02.00,0:00:05.00,Title"), "{ass}");
        assert!(ass.contains("{\\an4\\pos(100.0,750.0)}LOWER THIRD"), "{ass}");
        // Yellow (#ffcc00 → &H0000CCFF), bold, 50 px, 5 px outline, left-middle.
        assert!(ass.contains(",Arial,50.00,&H0000CCFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5.00,0.00,4,"), "{ass}");
        // Captions in a solid box (BorderStyle 3, box colour as outline colour), bottom at 30%.
        assert!(ass.contains(",&H00332211,&H80000000,-1,0,0,0,100,100,0,0,3,"), "{ass}");
        assert!(ass.contains("{\\an2\\pos(500.0,300.0)}dicho"), "{ass}");
    }

    #[test]
    fn title_times_stay_inside_their_beat() {
        let mut s = script(1920, 1080, Some("x"));
        s.beats[0].title_layout = Some(TitleLayout { start_s: Some(-3.0), end_s: Some(50.0), ..TitleLayout::default() });
        let t = &layout(&s, &[]).titles[0];
        assert_eq!((t.start_s, t.end_s), (0.0, 10.0));
    }

    #[test]
    fn text_is_escaped_for_ass() {
        assert_eq!(escape_ass_text("a {b} c"), "a \\{b\\} c");
        assert_eq!(escape_ass_text("C:\\Name"), "C:\\\u{2060}Name");
        let ov = layout(&script(1920, 1080, Some("{\\b1}bold?")), &[]);
        let ass = to_ass(&ov, 1920, 1080);
        assert!(!ass.contains("}{\\b1}"), "{ass}");
        assert!(ass.contains("\\{\\\u{2060}b1\\}bold?"), "{ass}");
    }

    #[test]
    fn long_titles_wrap_to_the_frame() {
        let ov = layout(&script(1080, 1920, Some("Mi tiempo vale más que diez dólares por hora")), &[]);
        assert!(ov.titles[0].lines.len() >= 2, "{:?}", ov.titles[0].lines);
        let ass = to_ass(&ov, 1080, 1920);
        assert!(ass.contains("\\N"));
        assert!(ass.contains("WrapStyle: 2"));
    }

    #[test]
    fn hidden_cues_are_laid_out_but_not_burnt() {
        let mut hidden = cue(1.0, 2.0, "secret");
        hidden.hidden = true;
        let ov = layout(&script(960, 540, None), &[hidden, cue(2.0, 3.0, "shown")]);
        assert_eq!(ov.captions.len(), 2);
        let ass = to_ass(&ov, 960, 540);
        assert!(!ass.contains("secret") && ass.contains("shown"));
    }

    #[test]
    fn wrapping_is_balanced() {
        assert_eq!(wrap_lines("Y entonces bueno, de ahí podemos hacer lo que sea", 36), vec![
            "Y entonces bueno, de ahí",
            "podemos hacer lo que sea"
        ]);
        assert_eq!(wrap_lines("corto", 16), vec!["corto"]);
        assert_eq!(wrap_lines("uno\ndos tres", 40), vec!["uno", "dos tres"]);
    }

    #[test]
    fn bad_style_values_fall_back() {
        let s = resolve(&TextStyle {
            color: Some("red".into()),
            size: Some(f64::NAN),
            font: Some("Ari,al".into()),
            box_opacity: Some(7.0),
            ..TextStyle::default()
        });
        assert_eq!(s.color, "#ffffff");
        assert_eq!(s.font, "Arial");
        assert_eq!(s.size, 0.055);
        assert_eq!(s.box_opacity, 1.0);
    }
}
