//! What a rendered cut actually plays: the words heard in its audio, the silence at each cut,
//! and a contact sheet of its pictures.
//!
//! Everything else an editor reads about a cut is worked out from the index (`read_cut`): the
//! sentences the clips were asked for, the speech the cuts were placed on. The render is what a
//! viewer gets, and it can differ — a cut inside a word the transcript timed wrongly, a co-host's
//! "sí" in a pause, a screen share where faces were expected. This reads the file itself, the way
//! a reviewer watching it would, and reports measurements: what was heard and when, how much
//! silence each cut sits in, what each frame shows. What to do about it is the editor's call.

use std::path::{Path, PathBuf};

use rusqlite::OptionalExtension;
use serde::Serialize;

use crate::Error;
use crate::db::Db;
use crate::script::Script;
use crate::sentences::clock;

/// A line heard in the render, on the render's clock, with the clip it falls in.
#[derive(Debug, Clone, Serialize)]
pub struct HeardLine {
    pub start_s: f64,
    pub end_s: f64,
    /// 1-based position of the clip (or section card) this line starts in.
    pub clip: usize,
    pub text: String,
    /// Share of the line's span (0–1) the render's measured sound has voice in. A line heard over
    /// silence (near 0) is the transcriber inventing words — "Gracias." over a trailing pause —
    /// not a word that leaked in.
    pub voiced: f64,
}

/// One join between two clips in the render, and the sound around it.
#[derive(Debug, Clone, Serialize)]
pub struct CutSound {
    /// Where the cut is on the render's clock.
    pub t_s: f64,
    /// Silence right before and right after the cut (s); 0 when the cut is inside sound.
    pub silence_before_s: f64,
    pub silence_after_s: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RenderReview {
    pub path: PathBuf,
    pub duration_s: f64,
    pub cuts: Vec<CutSound>,
    /// Silence from the first frame to the first voice, and from the last voice to the end (s).
    pub head_silence_s: f64,
    pub tail_silence_s: f64,
    pub heard: Vec<HeardLine>,
    /// Why nothing was heard when the speech engine could not run (the cuts and the picture are
    /// still measured).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript_error: Option<String>,
    #[serde(skip)]
    pub sheet_jpeg: Option<Vec<u8>>,
    /// The frames on both sides of every cut, and the first and last frame: a frame every 2 s
    /// misses a dissolve or a flash of another picture right at a cut.
    #[serde(skip)]
    pub cuts_jpeg: Option<Vec<u8>>,
}

/// The newest render of a script version on disk.
pub fn latest_render(db: &Db, script_id: i64) -> Option<PathBuf> {
    db.conn
        .query_row(
            "SELECT path FROM exports WHERE script_id = ?1 AND format = 'preview_mp4' ORDER BY created_at DESC, id DESC",
            [script_id],
            |r| r.get::<_, String>(0),
        )
        .optional()
        .ok()
        .flatten()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
}

/// Where each clip (and section card) starts on the render's clock, from the same plan the
/// renderer follows.
pub fn clip_starts(db: &Db, project_id: i64, script: &Script) -> Result<Vec<f64>, Error> {
    let mut s = script.clone();
    if let Ok(p) = db.project(project_id) {
        s.fill_from_project(&p);
    }
    let fps = s.fps.unwrap_or(crate::script::Fps::new(25, 1));
    let media = crate::otio::resolve_media_for_script(db, project_id, &s, fps)?;
    Ok(crate::preview::plan_segments(&s, &media)?.iter().map(|seg| seg.timeline_start_s).collect())
}

/// The silence around `t` in measured `regions`: (before, after).
fn silence_at(regions: &[crate::speech::SpeechRegion], t: f64, duration: f64) -> (f64, f64) {
    if regions.iter().any(|r| r.start_s < t && r.end_s > t) {
        return (0.0, 0.0);
    }
    let before = regions.iter().rev().find(|r| r.end_s <= t).map_or(t, |r| t - r.end_s);
    let after = regions.iter().find(|r| r.start_s >= t).map_or(duration - t, |r| r.start_s - t);
    (before.max(0.0), after.max(0.0))
}

/// Share of `start..end` covered by measured `regions`.
fn voiced_share(regions: &[crate::speech::SpeechRegion], start: f64, end: f64) -> f64 {
    let len = end - start;
    if len <= 0.0 {
        return 0.0;
    }
    let covered: f64 = regions.iter().map(|r| (r.end_s.min(end) - r.start_s.max(start)).max(0.0)).sum();
    (covered / len).clamp(0.0, 1.0)
}

/// The file's length, from ffprobe.
fn media_duration(video: &Path) -> Option<f64> {
    let ffprobe = crate::doctor::locate("ffprobe")?;
    let out = crate::proc::std_command(&ffprobe)
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(video)
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// A contact sheet of the render: a frame every `every_s`, timestamped, six to a row.
fn contact_sheet(ffmpeg: &Path, video: &Path, every_s: f64) -> Option<Vec<u8>> {
    let out = std::env::temp_dir().join(format!("ghostreel-sheet-{}-{}.jpg", std::process::id(), rand_suffix()));
    let vf = format!(
        "fps=1/{every_s},scale=320:-1,drawtext=text='%{{pts\\:hms}}':x=6:y=6:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6,tile=6x6"
    );
    let ok = crate::proc::std_command(ffmpeg)
        .args(["-v", "error", "-y", "-i"])
        .arg(video)
        .args(["-vf", &vf, "-frames:v", "1", "-q:v", "4"])
        .arg(&out)
        .status()
        .is_ok_and(|s| s.success());
    let jpeg = ok.then(|| std::fs::read(&out).ok()).flatten();
    let _ = std::fs::remove_file(&out);
    jpeg
}

/// When to look around each cut (s from the cut): the last frame before it, and three after it
/// across the first second, where a dissolve into the next shot would show.
const AROUND_CUT_S: [f64; 4] = [-0.1, 0.1, 0.5, 1.0];

/// A sheet of the frames at `marks` (time, label), labelled, four to a row.
fn frames_sheet(ffmpeg: &Path, video: &Path, marks: &[(f64, String)]) -> Option<Vec<u8>> {
    if marks.is_empty() {
        return None;
    }
    let dir = std::env::temp_dir().join(format!("ghostreel-cuts-{}-{}", std::process::id(), rand_suffix()));
    std::fs::create_dir_all(&dir).ok()?;
    let mut ok = true;
    for (i, (t, label)) in marks.iter().enumerate() {
        let label = label.replace(':', "\\:").replace('\'', "");
        let vf = format!(
            "scale=320:-1,drawtext=text='{label}':x=6:y=6:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6"
        );
        ok &= crate::proc::std_command(ffmpeg)
            .args(["-v", "error", "-y", "-ss", &format!("{:.3}", t.max(0.0)), "-i"])
            .arg(video)
            .args(["-frames:v", "1", "-vf", &vf, "-q:v", "4"])
            .arg(dir.join(format!("f{i:03}.jpg")))
            .status()
            .is_ok_and(|s| s.success());
    }
    let out = dir.join("sheet.jpg");
    let rows = marks.len().div_ceil(4);
    ok = ok
        && crate::proc::std_command(ffmpeg)
            .args(["-v", "error", "-y", "-framerate", "1", "-i"])
            .arg(dir.join("f%03d.jpg"))
            .args(["-vf", &format!("tile=4x{rows}"), "-frames:v", "1", "-q:v", "4"])
            .arg(&out)
            .status()
            .is_ok_and(|s| s.success());
    let jpeg = ok.then(|| std::fs::read(&out).ok()).flatten();
    let _ = std::fs::remove_dir_all(&dir);
    jpeg
}

fn rand_suffix() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
}

/// Review the render at `video`. `starts`: where each clip and card starts on the render's clock
/// ([`clip_starts`], read from the index before this runs: the review itself never touches it);
/// `length_s` stands in when the file's own length cannot be read.
pub async fn review(
    paths: &crate::paths::Paths,
    config: &crate::config::Config,
    starts: &[f64],
    length_s: f64,
    video: &Path,
) -> Result<RenderReview, Error> {
    let ffmpeg = crate::doctor::locate("ffmpeg").ok_or_else(|| Error::Invalid("ffmpeg not found".into()))?;
    let duration_s = media_duration(video).unwrap_or(length_s);

    // The sound, measured the way the speech stage measures footage.
    let levels = crate::speech::measure(&ffmpeg, video, 0, |_| {}).await?;
    let regions = crate::speech::detect(&levels, &crate::speech::VadParams::default());
    let cuts = starts
        .iter()
        .skip(1)
        .map(|&t| {
            let (before, after) = silence_at(&regions, t, duration_s);
            CutSound { t_s: r2(t), silence_before_s: r2(before), silence_after_s: r2(after) }
        })
        .collect();

    let head_silence_s = r2(regions.first().map_or(duration_s, |r| r.start_s).max(0.0));
    let tail_silence_s = r2(regions.last().map_or(duration_s, |r| duration_s - r.end_s).max(0.0));

    // The words, heard again from the render.
    let (heard, transcript_error) = match crate::runtime::resolve_stt(paths, config).await {
        crate::runtime::SttSetup::Ready(engine) => {
            match crate::stt::transcribe(&engine, &ffmpeg, video, duration_s, |_| {}, None).await {
                Ok(t) => (
                    t.segments
                        .into_iter()
                        .filter(|s| !s.text.trim().is_empty())
                        .map(|s| HeardLine {
                            start_s: r2(s.start),
                            end_s: r2(s.end),
                            clip: starts.iter().rposition(|&c| c <= s.start + 0.05).map_or(1, |i| i + 1),
                            text: s.text.trim().to_string(),
                            voiced: r2(voiced_share(&regions, s.start, s.end)),
                        })
                        .collect(),
                    None,
                ),
                Err(e) => (Vec::new(), Some(e.to_string())),
            }
        }
        other => (Vec::new(), Some(format!("no speech engine: {}", other.describe()))),
    };

    let mut marks: Vec<(f64, String)> = vec![(0.0, "start".to_string())];
    for (i, &t) in starts.iter().enumerate().skip(1) {
        for d in AROUND_CUT_S {
            marks.push((t + d, format!("cut {}-{} {:+.1}s", i, i + 1, d)));
        }
    }
    marks.push(((duration_s - 0.1).max(0.0), "end".to_string()));

    Ok(RenderReview {
        path: video.to_path_buf(),
        duration_s: r2(duration_s),
        cuts,
        head_silence_s,
        tail_silence_s,
        cuts_jpeg: frames_sheet(&ffmpeg, video, &marks),
        heard,
        transcript_error,
        sheet_jpeg: contact_sheet(&ffmpeg, video, 2.0),
    })
}

impl RenderReview {
    /// Compact text: the cuts with their silence, then the words heard, each with its clip.
    pub fn render(&self) -> String {
        let mut out = format!("render {} ({})\n", self.path.display(), clock(self.duration_s));
        out.push_str(&format!(
            "silence at the head {:.2} s, at the tail {:.2} s\n",
            self.head_silence_s, self.tail_silence_s
        ));
        out.push_str("cuts (silence before / after, s):\n");
        for (i, c) in self.cuts.iter().enumerate() {
            out.push_str(&format!(
                "  {} → {} at {}: {:.2} / {:.2}\n",
                i + 1,
                i + 2,
                clock(c.t_s),
                c.silence_before_s,
                c.silence_after_s
            ));
        }
        match &self.transcript_error {
            Some(e) => out.push_str(&format!("heard: not transcribed ({e})\n")),
            None => {
                out.push_str("heard (re-transcribed from the render):\n");
                for h in &self.heard {
                    let silent = if h.voiced < 0.2 { " (no voice measured here: likely invented by the transcriber)" } else { "" };
                    out.push_str(&format!("  [{} clip {}] {}{silent}\n", clock(h.start_s), h.clip, h.text));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::SpeechRegion;

    #[test]
    fn a_cut_reads_the_silence_around_it_or_none_inside_sound() {
        let regions = [SpeechRegion::new(0.5, 4.0), SpeechRegion::new(4.6, 9.0)];
        let at = |t| {
            let (a, b) = silence_at(&regions, t, 10.0);
            (r2(a), r2(b))
        };
        assert_eq!(at(4.2), (0.2, 0.4));
        assert_eq!(at(6.0), (0.0, 0.0), "inside sound");
        assert_eq!(at(9.5), (0.5, 0.5), "after the last voice, to the end of the file");
    }

    #[test]
    fn a_line_heard_over_silence_has_no_voice_under_it() {
        let regions = [SpeechRegion::new(0.5, 4.0), SpeechRegion::new(4.6, 9.0)];
        assert_eq!(r2(voiced_share(&regions, 1.0, 3.0)), 1.0);
        assert_eq!(r2(voiced_share(&regions, 3.0, 5.0)), 0.7, "1.0 of voice, 0.6 of pause, 0.4 of voice");
        assert_eq!(voiced_share(&regions, 9.2, 10.0), 0.0, "a 'Gracias.' after the last voice");
    }
}
