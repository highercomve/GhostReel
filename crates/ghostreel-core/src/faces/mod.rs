pub mod detector;
pub mod embedder;
pub mod track;

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub use detector::{FaceBox, OrtDetector, box_iou, decode_yunet};
pub use embedder::{
    ARCFACE_TEMPLATE_112, OrtEmbedder, SimilarityTransform, align_face_112, cosine_similarity, l2_normalize,
    laplacian_variance,
};
pub use track::{DetectedFace, FaceBoxRecord, FaceTrack, FaceTracker, cluster_tracks_average_linkage};

use crate::Error;

/// Serializes a 128-d f32 embedding into a 512-byte little-endian BLOB.
pub fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        blob.extend_from_slice(&val.to_le_bytes());
    }
    blob
}

/// Deserializes a little-endian BLOB into a 128-d f32 embedding.
pub fn blob_to_embedding(blob: &[u8]) -> Result<Vec<f32>, Error> {
    if blob.len() % 4 != 0 {
        return Err(Error::Invalid(format!("invalid embedding BLOB length: {}", blob.len())));
    }
    let mut embedding = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().map_err(|_| Error::Invalid("chunk slice error".into()))?;
        embedding.push(f32::from_le_bytes(arr));
    }
    Ok(embedding)
}

/// Detect and track faces through a video file at 5 fps using ffmpeg and YuNet + SFace.
///
/// `on_progress` gets the seconds of video done. Stops within a frame when `cancel` is set
/// ([`Error::Cancelled`], ffmpeg killed). An ffmpeg that fails is an error with its message, not
/// an empty result: "no faces" must only ever mean the video was looked at.
#[allow(clippy::too_many_arguments)]
pub fn process_video_faces(
    ffmpeg_path: &Path,
    video_path: &Path,
    video_id: i64,
    orig_w: u32,
    orig_h: u32,
    detector: &mut OrtDetector,
    embedder: &mut OrtEmbedder,
    cancel: Option<&Arc<AtomicBool>>,
    mut on_progress: impl FnMut(f64),
) -> Result<Vec<FaceTrack>, Error> {
    const FRAME_W: usize = 640;
    const FRAME_H: usize = 640;
    const FRAME_BYTES: usize = FRAME_W * FRAME_H * 3;

    let (active_w, active_h) = if orig_w == 0 || orig_h == 0 {
        (640.0f32, 360.0f32)
    } else if orig_w >= orig_h {
        let h = (640.0 * (orig_h as f32) / (orig_w as f32)).round();
        (640.0f32, h)
    } else {
        let w = (640.0 * (orig_w as f32) / (orig_h as f32)).round();
        (w, 640.0f32)
    };

    let mut cmd = crate::proc::std_command(ffmpeg_path);
    cmd.arg("-nostdin").arg("-hide_banner").arg("-loglevel").arg("error");
    cmd.arg("-i").arg(video_path);
    cmd.arg("-vf").arg("fps=5,scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:0:0:black");
    cmd.arg("-f").arg("rawvideo").arg("-pix_fmt").arg("bgr24").arg("pipe:1");
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| Error::Io(video_path.to_path_buf(), e))?;
    let stdout = child.stdout.take().ok_or_else(|| Error::Invalid("failed to capture ffmpeg stdout".into()))?;
    // Drained on its own thread so a chatty ffmpeg cannot stall on a full pipe; kept for the error.
    let stderr = child.stderr.take().map(|mut e| {
        std::thread::spawn(move || {
            let mut s = String::new();
            let _ = e.read_to_string(&mut s);
            s
        })
    });
    let mut reader = std::io::BufReader::with_capacity(4 * 1024 * 1024, stdout);

    let mut frame_buf = vec![0u8; FRAME_BYTES];
    let mut bgr_nchw = vec![0.0f32; 1 * 3 * FRAME_W * FRAME_H];
    let mut rgb_sface_nchw = [0.0f32; 1 * 3 * 112 * 112];
    let mut gray_112 = [0.0f32; 112 * 112];

    let mut tracker = FaceTracker::new(video_id);
    let mut frame_idx = 0usize;

    loop {
        if let Some(c) = cancel
            && c.load(Ordering::SeqCst)
        {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Error::Cancelled);
        }

        match reader.read_exact(&mut frame_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::Io(video_path.to_path_buf(), e));
            }
        }

        let t_s = (frame_idx as f64) / 5.0;

        // Populate BGR NCHW tensor for YuNet
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

        // YuNet detection with score >= 0.8 and NMS IoU 0.3
        let boxes = match detector.detect(&bgr_nchw, 0.8, 0.3) {
            Ok(b) => b,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::Invalid(format!("YuNet inference error: {e}")));
            }
        };

        // Filter boxes inside the active image area
        let active_boxes: Vec<_> = boxes
            .into_iter()
            .filter(|b| {
                let cy = b.y + b.h / 2.0;
                let cx = b.x + b.w / 2.0;
                cy < active_h && b.y < active_h && cx < active_w && b.x < active_w && b.w >= 8.0 && b.h >= 8.0
            })
            .collect();

        let mut frame_detections = Vec::with_capacity(active_boxes.len());
        for b in active_boxes {
            // Align and warp face to 112x112 ArcFace template
            align_face_112(&frame_buf, FRAME_W, FRAME_H, &b.landmarks, &mut rgb_sface_nchw, &mut gray_112);

            let sharpness = laplacian_variance(&gray_112);

            let touches_left = b.x <= 1.0;
            let touches_top = b.y <= 1.0;
            let touches_right = (b.x + b.w) >= (active_w - 1.0);
            let touches_bottom = (b.y + b.h) >= (active_h - 1.0);
            let cut_off = touches_left || touches_top || touches_right || touches_bottom;

            let embedding = match embedder.embed(&rgb_sface_nchw) {
                Ok(emb) => emb,
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Error::Invalid(format!("SFace embedding error: {e}")));
                }
            };

            let norm_x = (b.x / active_w).clamp(0.0, 1.0);
            let norm_y = (b.y / active_h).clamp(0.0, 1.0);
            let norm_w = (b.w / active_w).clamp(0.0, 1.0);
            let norm_h = (b.h / active_h).clamp(0.0, 1.0);

            let norm_landmarks = b.landmarks.map(|(lx, ly)| (lx / active_w, ly / active_h));
            let landmarks_json = serde_json::to_string(&norm_landmarks).unwrap_or_default();
            let size_px = b.h.max(b.w);

            frame_detections.push(DetectedFace {
                t_s,
                box_raw: (b.x, b.y, b.w, b.h),
                norm_x,
                norm_y,
                norm_w,
                norm_h,
                score: b.score,
                landmarks_json,
                sharpness,
                cut_off,
                embedding,
                size_px,
            });
        }

        tracker.process_frame(t_s, frame_detections);
        frame_idx += 1;
        if frame_idx.is_multiple_of(5) {
            on_progress(frame_idx as f64 / 5.0);
        }
    }

    let status = child.wait().map_err(|e| Error::Io(video_path.to_path_buf(), e))?;
    let errors = stderr.and_then(|h| h.join().ok()).unwrap_or_default();
    if !status.success() || frame_idx == 0 {
        let why = errors.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("no frames decoded").trim();
        return Err(Error::Invalid(format!("ffmpeg could not decode the video for faces ({status}): {why}")));
    }
    Ok(tracker.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;

    #[test]
    fn test_embedding_blob_roundtrip() {
        let mut original = vec![0.0f32; 128];
        for i in 0..128 {
            original[i] = ((i as f32) * 1.37 - 64.0).sin();
        }
        let blob = embedding_to_blob(&original);
        assert_eq!(blob.len(), 512, "128 f32s must serialize to 512 bytes");

        let recovered = blob_to_embedding(&blob).expect("must deserialize cleanly");
        assert_eq!(recovered.len(), 128);
        assert_eq!(recovered, original, "recovered embedding must match bit-identically");
    }

    #[test]
    fn test_face_db_roundtrip() {
        let db = Db::open_in_memory().expect("open in-memory db");

        // Insert dummy video
        db.conn
            .execute(
                "INSERT INTO videos (id, content_hash, size, duration_s)
                 VALUES (1, 'hash_test_1', 1024, 60.0)",
                [],
            )
            .expect("insert video");

        let mut emb = vec![0.0f32; 128];
        emb[0] = 0.5;
        emb[1] = 0.866;

        let b1 = FaceBoxRecord {
            track_id: 1,
            t_s: 1.0,
            x: 0.1,
            y: 0.2,
            w: 0.15,
            h: 0.25,
            score: 0.95,
            landmarks_json: "[[0.12,0.22]]".into(),
            sharpness: 125.5,
            cut_off: false,
        };
        let b2 = FaceBoxRecord {
            track_id: 1,
            t_s: 1.2,
            x: 0.11,
            y: 0.21,
            w: 0.15,
            h: 0.25,
            score: 0.96,
            landmarks_json: "[[0.13,0.23]]".into(),
            sharpness: 130.0,
            cut_off: true,
        };

        let track1 = FaceTrack {
            id: 1,
            video_id: 1,
            start_s: 1.0,
            end_s: 1.2,
            n_faces: 2,
            mean_embedding: emb.clone(),
            avg_box_h: 0.25,
            boxes: vec![b1, b2],
        };

        db.set_face_tracks(1, &[track1]).expect("set face tracks");

        let tracks = db.face_tracks(1).expect("get face tracks");
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].video_id, 1);
        assert_eq!(tracks[0].start_s, 1.0);
        assert_eq!(tracks[0].end_s, 1.2);
        assert_eq!(tracks[0].n_faces, 2);
        assert!((tracks[0].avg_box_h - 0.25).abs() < 1e-4);
        assert_eq!(tracks[0].mean_embedding, emb);

        let boxes = db.face_boxes(1, 0.5, 1.5).expect("get face boxes");
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].t_s, 1.0);
        assert!((boxes[0].sharpness - 125.5).abs() < 1e-3);
        assert!(!boxes[0].cut_off);
        assert_eq!(boxes[1].t_s, 1.2);
        assert!(boxes[1].cut_off);

        let track_boxes = db.face_boxes_for_track(tracks[0].id).expect("get track boxes");
        assert_eq!(track_boxes.len(), 2);

        // Replace tracks test
        let track2 = FaceTrack {
            id: 2,
            video_id: 1,
            start_s: 5.0,
            end_s: 5.5,
            n_faces: 1,
            mean_embedding: emb.clone(),
            avg_box_h: 0.3,
            boxes: vec![FaceBoxRecord {
                track_id: 2,
                t_s: 5.0,
                x: 0.4,
                y: 0.4,
                w: 0.2,
                h: 0.3,
                score: 0.99,
                landmarks_json: "[]".into(),
                sharpness: 88.0,
                cut_off: false,
            }],
        };

        db.set_face_tracks(1, &[track2]).expect("replace face tracks");
        let tracks_after = db.face_tracks(1).expect("get face tracks after replace");
        assert_eq!(tracks_after.len(), 1);
        assert_eq!(tracks_after[0].start_s, 5.0);
        assert_eq!(tracks_after[0].n_faces, 1);

        let boxes_after = db.face_boxes(1, 0.0, 10.0).expect("get face boxes after replace");
        assert_eq!(boxes_after.len(), 1);
        assert_eq!(boxes_after[0].t_s, 5.0);
    }
}
