use crate::embedder::cosine_similarity;
use image::RgbImage;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Clone)]
pub struct DetectedFace {
    pub frame_idx: usize,
    pub t_s: f64,
    pub face_idx: usize,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
    pub embedding: Vec<f32>,
    pub crop: RgbImage,
    pub cluster_id: usize,
}

pub struct Cluster {
    pub id: usize,
    pub centroid: Vec<f32>,
    pub face_indices: Vec<usize>,
}

pub fn cluster_faces(faces: &mut [DetectedFace], sim_threshold: f32) -> Vec<Cluster> {
    let mut clusters: Vec<Cluster> = Vec::new();

    for (face_idx, face) in faces.iter_mut().enumerate() {
        let mut best_sim = -1.0f32;
        let mut best_cluster_idx = None;

        for (c_idx, cluster) in clusters.iter().enumerate() {
            let sim = cosine_similarity(&face.embedding, &cluster.centroid);
            if sim > best_sim {
                best_sim = sim;
                best_cluster_idx = Some(c_idx);
            }
        }

        if let Some(c_idx) = best_cluster_idx {
            if best_sim >= sim_threshold {
                clusters[c_idx].face_indices.push(face_idx);
                let n = clusters[c_idx].face_indices.len() as f32;
                let c_len = clusters[c_idx].centroid.len();
                for i in 0..c_len {
                    clusters[c_idx].centroid[i] =
                        (clusters[c_idx].centroid[i] * (n - 1.0) + face.embedding[i]) / n;
                }
                clusters[c_idx].centroid = crate::embedder::l2_normalize(&clusters[c_idx].centroid);
                face.cluster_id = clusters[c_idx].id;
                continue;
            }
        }

        // New cluster
        let new_id = clusters.len();
        face.cluster_id = new_id;
        clusters.push(Cluster {
            id: new_id,
            centroid: face.embedding.clone(),
            face_indices: vec![face_idx],
        });
    }

    // Sort clusters by size descending and renumber them 0, 1, ...
    clusters.sort_by(|a, b| b.face_indices.len().cmp(&a.face_indices.len()));
    let mut old_to_new = vec![0; clusters.len()];
    for (new_rank, c) in clusters.iter_mut().enumerate() {
        old_to_new[c.id] = new_rank;
        c.id = new_rank;
    }

    for face in faces.iter_mut() {
        face.cluster_id = old_to_new[face.cluster_id];
    }

    clusters
}

pub fn write_faces_csv(faces: &[DetectedFace], path: impl AsRef<Path>) -> anyhow::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "t_s,face_idx,x,y,w,h,score,cluster")?;
    for f in faces {
        writeln!(
            file,
            "{:.3},{},{:.1},{:.1},{:.1},{:.1},{:.4},{}",
            f.t_s, f.face_idx, f.x, f.y, f.w, f.h, f.score, f.cluster_id
        )?;
    }
    Ok(())
}

pub fn save_cluster_samples(
    clusters: &[Cluster],
    faces: &[DetectedFace],
    out_dir: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    for cluster in clusters {
        let count = cluster.face_indices.len();
        if count == 0 {
            continue;
        }

        let sample_indices = if count == 1 {
            vec![cluster.face_indices[0]]
        } else if count == 2 {
            vec![cluster.face_indices[0], cluster.face_indices[1]]
        } else {
            vec![
                cluster.face_indices[0],
                cluster.face_indices[count / 2],
                cluster.face_indices[count - 1],
            ]
        };

        for (s_idx, &face_idx) in sample_indices.iter().enumerate() {
            let filename = format!("cluster_{}_sample_{}.png", cluster.id, s_idx + 1);
            let path = out_dir.join(filename);
            faces[face_idx].crop.save(&path)?;
        }
    }

    Ok(())
}
