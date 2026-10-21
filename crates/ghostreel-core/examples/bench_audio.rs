//! Compare per-segment seeks against one-pass measurements on an existing video.
//! cargo run -p ghostreel-core --example bench_audio -- <manifest.json>
//! Manifest: {"video":"...","track":0,"segments":[[0.0,2.0], ...]}.
use ghostreel_core::audio;
use serde::Deserialize;
use std::{path::PathBuf, time::Instant};
#[derive(Deserialize)]
struct Input {
    video: PathBuf,
    track: u32,
    segments: Vec<(f64, f64)>,
}
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let path = std::env::args().nth(1).expect("manifest.json required");
    let input: Input = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let ffmpeg = ghostreel_core::doctor::locate("ffmpeg").unwrap();
    let started = Instant::now();
    let mut before = Vec::new();
    for &(start, end) in &input.segments {
        before.push(if end - start >= 0.4 {
            audio::mean_level(&ffmpeg, &input.video, input.track, start, end - start).await
        } else {
            None
        });
    }
    let before_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let after = audio::segment_levels(&ffmpeg, &input.video, input.track, &input.segments).await;
    let after_s = started.elapsed().as_secs_f64();
    let max_delta_db =
        before.iter().zip(&after).filter_map(|(a, b)| Some((a.as_ref()? - b.as_ref()?).abs())).fold(0.0_f64, f64::max);
    let flags_before = audio::off_mic_flags(&before, audio::OFF_MIC_MARGIN_DB);
    let flags_after = audio::off_mic_flags(&after, audio::OFF_MIC_MARGIN_DB);
    println!(
        "{}",
        serde_json::json!({"segments":input.segments.len(),"before_s":before_s,"after_s":after_s,"speedup":before_s/after_s,"max_delta_db":max_delta_db,"changed_flags":flags_before.iter().zip(flags_after).filter(|(a,b)|**a!=*b).count(),"missing_before":before.iter().filter(|v|v.is_none()).count(),"missing_after":after.iter().filter(|v|v.is_none()).count()})
    );
}
