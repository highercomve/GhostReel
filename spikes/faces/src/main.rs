mod cluster;
mod detector;
mod embedder;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use cluster::{cluster_faces, save_cluster_samples, write_faces_csv, DetectedFace};
use detector::{Detector, OrtDetector, TractDetector};
use embedder::{Embedder, OrtEmbedder, TractEmbedder};
use image::imageops::FilterType;
use image::RgbImage;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RuntimeChoice {
    Ort,
    Tract,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Face detection and recognition spike", long_about = None)]
pub struct Cli {
    /// Path to input video file
    pub video: PathBuf,

    /// Runtime engine to use: ort (ONNX Runtime) or tract (pure Rust)
    #[arg(short, long, value_enum, default_value_t = RuntimeChoice::Ort)]
    pub runtime: RuntimeChoice,

    /// Sample frame rate
    #[arg(long, default_value_t = 5.0)]
    pub fps: f64,

    /// Start timestamp in seconds
    #[arg(long)]
    pub from: Option<f64>,

    /// End timestamp in seconds
    #[arg(long)]
    pub to: Option<f64>,

    /// Path to YuNet ONNX model
    #[arg(long, default_value = "models/face_detection_yunet_2023mar.onnx")]
    pub yunet_path: PathBuf,

    /// Path to SFace ONNX model
    #[arg(long, default_value = "models/face_recognition_sface_2021dec.onnx")]
    pub sface_path: PathBuf,

    /// Output directory for artifacts (CSV and PNG crops)
    #[arg(long, default_value = "out")]
    pub out_dir: PathBuf,

    /// Custom CSV output path (defaults to <out_dir>/faces.csv)
    #[arg(long)]
    pub csv: Option<PathBuf>,

    /// Face detection confidence threshold
    #[arg(long, default_value_t = 0.8)]
    pub score_threshold: f32,

    /// NMS IoU threshold
    #[arg(long, default_value_t = 0.3)]
    pub nms_threshold: f32,

    /// Cosine similarity threshold for clustering
    #[arg(long, default_value_t = 0.45)]
    pub sim_threshold: f32,
}

fn stats(v: &mut [f64]) -> (f64, f64, f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let median = if v.len() % 2 == 0 {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    } else {
        v[v.len() / 2]
    };
    let min = v[0];
    let max = v[v.len() - 1];
    (mean, median, min, max)
}

fn main() -> Result<()> {
    let args = Cli::parse();

    println!("============================================================");
    println!("GhostReel Faces Spike");
    println!("Video:     {:?}", args.video);
    println!("Runtime:   {:?}", args.runtime);
    println!("FPS:       {:.2}", args.fps);
    println!("Range:     from={:?}s to={:?}s", args.from, args.to);
    println!("YuNet:     {:?}", args.yunet_path);
    println!("SFace:     {:?}", args.sface_path);
    println!("Output:    {:?}", args.out_dir);
    println!("============================================================");

    std::fs::create_dir_all(&args.out_dir)?;
    let csv_path = args
        .csv
        .clone()
        .unwrap_or_else(|| args.out_dir.join("faces.csv"));

    // Initialize models
    let (mut detector, mut embedder): (Box<dyn Detector>, Box<dyn Embedder>) = match args.runtime {
        RuntimeChoice::Ort => {
            println!("Initializing ORT (ONNX Runtime) engine...");
            let env = Arc::new(ort::init().build().context("Failed to init ort environment")?);
            let d = OrtDetector::new(&env, args.yunet_path.to_str().unwrap())?;
            let e = OrtEmbedder::new(&env, args.sface_path.to_str().unwrap())?;
            (Box::new(d), Box::new(e))
        }
        RuntimeChoice::Tract => {
            println!("Initializing Tract (pure Rust) engine...");
            let d = TractDetector::new(args.yunet_path.to_str().unwrap())?;
            let e = TractEmbedder::new(args.sface_path.to_str().unwrap())?;
            (Box::new(d), Box::new(e))
        }
    };

    println!("Engines loaded successfully. Spawning ffmpeg reader...");

    // Build ffmpeg process
    let mut cmd = std::process::Command::new("ffmpeg");
    cmd.arg("-hide_banner").arg("-loglevel").arg("error");
    if let Some(from_s) = args.from {
        cmd.arg("-ss").arg(format!("{from_s}"));
    }
    cmd.arg("-i").arg(&args.video);
    if let (Some(from_s), Some(to_s)) = (args.from, args.to) {
        let duration = (to_s - from_s).max(0.0);
        cmd.arg("-t").arg(format!("{duration}"));
    } else if let Some(to_s) = args.to {
        cmd.arg("-to").arg(format!("{to_s}"));
    }
    cmd.arg("-vf").arg(format!(
        "fps={},scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:0:0:black",
        args.fps
    ));
    cmd.arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("bgr24")
        .arg("pipe:1");
    cmd.stdout(std::process::Stdio::piped());

    let mut child = cmd.spawn().context("Failed to spawn ffmpeg")?;
    let stdout = child.stdout.take().context("Failed to capture ffmpeg stdout")?;
    let mut reader = std::io::BufReader::with_capacity(4 * 1024 * 1024, stdout);

    const FRAME_W: usize = 640;
    const FRAME_H: usize = 640;
    const ACTIVE_H: usize = 360; // 1280x720 scaled by 0.5 is 640x360
    const FRAME_BYTES: usize = FRAME_W * FRAME_H * 3;

    let mut frame_buf = vec![0u8; FRAME_BYTES];
    let mut bgr_nchw = vec![0.0f32; 1 * 3 * FRAME_W * FRAME_H];
    let mut rgb_sface_nchw = vec![0.0f32; 1 * 3 * 112 * 112];

    let mut frame_idx = 0usize;
    let mut total_det_times: Vec<f64> = Vec::new();
    let mut total_emb_times: Vec<f64> = Vec::new();
    let mut all_faces: Vec<DetectedFace> = Vec::new();
    let mut faces_per_frame_hist: Vec<usize> = vec![0; 10]; // 0..9 faces

    let wall_start = Instant::now();

    println!("Starting frame loop...");

    loop {
        match reader.read_exact(&mut frame_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break; // Stream finished
            }
            Err(e) => {
                eprintln!("Error reading from ffmpeg: {e}");
                break;
            }
        }

        let t_s = args.from.unwrap_or(0.0) + (frame_idx as f64) / args.fps;

        // Populate BGR NCHW tensor for YuNet
        // In frame_buf: pixels are [B, G, R] interleaved
        for y in 0..FRAME_H {
            for x in 0..FRAME_W {
                let pixel_idx = (y * FRAME_W + x) * 3;
                let b = frame_buf[pixel_idx + 0] as f32;
                let g = frame_buf[pixel_idx + 1] as f32;
                let r = frame_buf[pixel_idx + 2] as f32;
                bgr_nchw[0 * FRAME_W * FRAME_H + y * FRAME_W + x] = b;
                bgr_nchw[1 * FRAME_W * FRAME_H + y * FRAME_W + x] = g;
                bgr_nchw[2 * FRAME_W * FRAME_H + y * FRAME_W + x] = r;
            }
        }

        // Run detection
        let t0_det = Instant::now();
        let boxes = detector.detect(&bgr_nchw, args.score_threshold, args.nms_threshold)?;
        let det_ms = t0_det.elapsed().as_secs_f64() * 1000.0;
        total_det_times.push(det_ms);

        // Filter boxes strictly inside active image area (top 360 lines)
        let active_boxes: Vec<_> = boxes
            .into_iter()
            .filter(|b| {
                let cy = b.y + b.h / 2.0;
                cy < ACTIVE_H as f32 && b.y < ACTIVE_H as f32 && b.w >= 8.0 && b.h >= 8.0
            })
            .collect();

        let num_faces = active_boxes.len();
        if num_faces < faces_per_frame_hist.len() {
            faces_per_frame_hist[num_faces] += 1;
        } else {
            *faces_per_frame_hist.last_mut().unwrap() += 1;
        }

        for (face_sub_idx, b) in active_boxes.into_iter().enumerate() {
            let x0 = b.x.max(0.0).min(FRAME_W as f32 - 1.0) as u32;
            let y0 = b.y.max(0.0).min(ACTIVE_H as f32 - 1.0) as u32;
            let x1 = (b.x + b.w).max(0.0).min(FRAME_W as f32) as u32;
            let y1 = (b.y + b.h).max(0.0).min(ACTIVE_H as f32) as u32;

            let crop_w = x1.saturating_sub(x0);
            let crop_h = y1.saturating_sub(y0);
            if crop_w < 8 || crop_h < 8 {
                continue;
            }

            // Extract crop in RGB
            let mut crop_img = RgbImage::new(crop_w, crop_h);
            for cy in 0..crop_h {
                for cx in 0..crop_w {
                    let src_x = (x0 + cx) as usize;
                    let src_y = (y0 + cy) as usize;
                    let p_idx = (src_y * FRAME_W + src_x) * 3;
                    let b_val = frame_buf[p_idx + 0];
                    let g_val = frame_buf[p_idx + 1];
                    let r_val = frame_buf[p_idx + 2];
                    crop_img.put_pixel(cx, cy, image::Rgb([r_val, g_val, b_val]));
                }
            }

            // Resize to 112x112 for SFace
            let resized = image::imageops::resize(&crop_img, 112, 112, FilterType::Triangle);
            for y in 0..112 {
                for x in 0..112 {
                    let p = resized.get_pixel(x, y);
                    rgb_sface_nchw[0 * 112 * 112 + y as usize * 112 + x as usize] = p[0] as f32;
                    rgb_sface_nchw[1 * 112 * 112 + y as usize * 112 + x as usize] = p[1] as f32;
                    rgb_sface_nchw[2 * 112 * 112 + y as usize * 112 + x as usize] = p[2] as f32;
                }
            }

            // Run SFace embedding
            let t0_emb = Instant::now();
            let embedding = embedder.embed(&rgb_sface_nchw)?;
            let emb_ms = t0_emb.elapsed().as_secs_f64() * 1000.0;
            total_emb_times.push(emb_ms);

            all_faces.push(DetectedFace {
                frame_idx,
                t_s,
                face_idx: face_sub_idx,
                x: b.x,
                y: b.y,
                w: b.w,
                h: b.h,
                score: b.score,
                embedding,
                crop: resized,
                cluster_id: 0,
            });
        }

        frame_idx += 1;
        if frame_idx % 250 == 0 {
            let elapsed = wall_start.elapsed().as_secs_f64();
            let fps_curr = frame_idx as f64 / elapsed;
            println!(
                "  [frame {:5}] t={:7.2}s | faces so far: {:5} | speed: {:5.1} fps",
                frame_idx, t_s, all_faces.len(), fps_curr
            );
        }
    }

    let _ = child.wait();
    let total_wall_time = wall_start.elapsed().as_secs_f64();

    println!("\n=================== Inference Completed ===================");
    println!("Total frames:        {}", frame_idx);
    println!("Total faces:         {}", all_faces.len());
    println!("Total wall time:     {:.2} s", total_wall_time);
    println!("Overall throughput:  {:.2} fps", frame_idx as f64 / total_wall_time);

    // Compute stats
    let (det_mean, det_med, det_min, det_max) = stats(&mut total_det_times);
    let (emb_mean, emb_med, emb_min, emb_max) = stats(&mut total_emb_times);

    println!("\n--- Detection Timing (YuNet) ---");
    println!("  Mean:   {:6.2} ms/frame", det_mean);
    println!("  Median: {:6.2} ms/frame", det_med);
    println!("  Min:    {:6.2} ms/frame", det_min);
    println!("  Max:    {:6.2} ms/frame", det_max);

    println!("\n--- Embedding Timing (SFace) ---");
    println!("  Mean:   {:6.2} ms/face", emb_mean);
    println!("  Median: {:6.2} ms/face", emb_med);
    println!("  Min:    {:6.2} ms/face", emb_min);
    println!("  Max:    {:6.2} ms/face", emb_max);

    println!("\n--- Histogram: Faces per Frame ---");
    for (count, &frames) in faces_per_frame_hist.iter().enumerate() {
        if frames > 0 || count <= 3 {
            let pct = (frames as f64 / frame_idx as f64) * 100.0;
            if count == faces_per_frame_hist.len() - 1 {
                println!("  {:2}+ faces: {:5} frames ({:5.1}%)", count, frames, pct);
            } else {
                println!("  {:2}  faces: {:5} frames ({:5.1}%)", count, frames, pct);
            }
        }
    }

    // Run clustering
    println!("\nRunning greedy clustering (similarity threshold = {:.2})...", args.sim_threshold);
    let clusters = cluster_faces(&mut all_faces, args.sim_threshold);
    println!("Found {} unique face clusters.", clusters.len());

    println!("\n--- Top Clusters ---");
    for (i, c) in clusters.iter().take(10).enumerate() {
        let count = c.face_indices.len();
        let pct = (count as f64 / all_faces.len() as f64) * 100.0;
        let first_face = &all_faces[c.face_indices[0]];
        let last_face = &all_faces[*c.face_indices.last().unwrap()];
        println!(
            "  Cluster {:2}: {:5} faces ({:5.1}%) | first seen: {:7.2}s, last seen: {:7.2}s",
            i, count, pct, first_face.t_s, last_face.t_s
        );
    }

    // Write CSV
    println!("\nWriting CSV to {:?}...", csv_path);
    write_faces_csv(&all_faces, &csv_path)?;

    // Save sample crops
    println!("Saving sample PNG crops to {:?}...", args.out_dir);
    save_cluster_samples(&clusters, &all_faces, &args.out_dir)?;

    println!("Done!");
    Ok(())
}
