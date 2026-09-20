//! Model files: find an existing copy (GhostReel's models dir, configured search paths, sibling
//! apps like GhostPen/LM Studio) before downloading, and download with resume + progress.

use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use crate::Error;
use crate::config::Config;
use crate::paths::Paths;

/// Effective directory for downloaded models: `config.models.dir` if set, else `paths.models_dir()`.
pub fn effective_models_dir(paths: &Paths, config: &Config) -> PathBuf {
    config.models.dir.clone().unwrap_or_else(|| paths.models_dir())
}

/// Category of model in GhostReel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Whisper,
    Vision,
    VisionProjector,
    Embedding,
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelKind::Whisper => write!(f, "whisper"),
            ModelKind::Vision => write!(f, "vision"),
            ModelKind::VisionProjector => write!(f, "vision_projector"),
            ModelKind::Embedding => write!(f, "embedding"),
        }
    }
}

/// A downloadable model file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub file_name: String,
    pub url: String,
}

/// An entry in GhostReel's curated model catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: String,
    pub kind: ModelKind,
    pub file_name: String,
    pub url: String,
    pub size_bytes: u64,
    pub speed: u8,
    pub accuracy: u8,
    pub note: String,
    pub languages: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_file_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_mb: Option<u64>,
}

impl CatalogEntry {
    pub fn spec(&self) -> ModelSpec {
        ModelSpec { file_name: self.file_name.clone(), url: self.url.clone() }
    }

    pub fn mmproj_spec(&self) -> Option<ModelSpec> {
        match (&self.mmproj_file_name, &self.mmproj_url) {
            (Some(file_name), Some(url)) => Some(ModelSpec { file_name: file_name.clone(), url: url.clone() }),
            _ => None,
        }
    }

    pub fn total_size_bytes(&self) -> u64 {
        self.size_bytes + self.mmproj_size_bytes.unwrap_or(0)
    }
}

/// Installation status of a catalog model on this computer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStatus {
    pub entry: CatalogEntry,
    pub installed_path: Option<PathBuf>,
    pub in_own_dir: bool,
    pub partial_bytes: Option<u64>,
}

/// Built-in catalog of recommended models with verified byte sizes.
pub fn catalog() -> Vec<CatalogEntry> {
    let (vision_spec, proj_spec) = (
        hf("prism-ml/Bonsai-27B-gguf", "Bonsai-27B-Q1_0.gguf"),
        hf("prism-ml/Bonsai-27B-gguf", "Bonsai-27B-mmproj-Q8_0.gguf"),
    );
    let embed_spec = embeddinggemma();
    vec![
        // Whisper models (HF ggerganov/whisper.cpp)
        CatalogEntry {
            id: "tiny".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-tiny.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin".into(),
            size_bytes: 77_691_713,
            speed: 5,
            accuracy: 1,
            note: "fastest, lowest accuracy".into(),
            languages: "multilingual".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: None,
        },
        CatalogEntry {
            id: "tiny.en".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-tiny.en.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin".into(),
            size_bytes: 77_704_715,
            speed: 5,
            accuracy: 2,
            note: "fastest, English-only".into(),
            languages: "english".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: None,
        },
        CatalogEntry {
            id: "base".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-base.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin".into(),
            size_bytes: 147_951_465,
            speed: 4,
            accuracy: 2,
            note: "fast, basic accuracy".into(),
            languages: "multilingual".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: None,
        },
        CatalogEntry {
            id: "base.en".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-base.en.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin".into(),
            size_bytes: 147_964_211,
            speed: 4,
            accuracy: 3,
            note: "fast, English-only".into(),
            languages: "english".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: None,
        },
        CatalogEntry {
            id: "small".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-small.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin".into(),
            size_bytes: 487_601_967,
            speed: 3,
            accuracy: 4,
            note: "balanced — sweet spot on a GPU".into(),
            languages: "multilingual".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: None,
        },
        CatalogEntry {
            id: "small.en".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-small.en.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin".into(),
            size_bytes: 487_614_201,
            speed: 3,
            accuracy: 4,
            note: "balanced, English-only".into(),
            languages: "english".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: None,
        },
        CatalogEntry {
            id: "medium".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-medium.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin".into(),
            size_bytes: 1_533_763_059,
            speed: 2,
            accuracy: 5,
            note: "most accurate classic whisper, heaviest".into(),
            languages: "multilingual".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: None,
        },
        CatalogEntry {
            id: "medium.en".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-medium.en.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin".into(),
            size_bytes: 1_533_774_781,
            speed: 2,
            accuracy: 5,
            note: "most accurate classic whisper, English-only".into(),
            languages: "english".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: None,
        },
        CatalogEntry {
            id: "large-v3-turbo".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-large-v3-turbo.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin".into(),
            size_bytes: 1_624_555_275,
            speed: 4,
            accuracy: 5,
            note: "best accuracy, needs ~2 GB VRAM".into(),
            languages: "multilingual".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: Some(2000),
        },
        CatalogEntry {
            id: "large-v3-turbo-q8_0".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-large-v3-turbo-q8_0.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin".into(),
            size_bytes: 874_188_075,
            speed: 4,
            accuracy: 5,
            note: "high accuracy, 8-bit quantized (~874 MB)".into(),
            languages: "multilingual".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: Some(1500),
        },
        CatalogEntry {
            id: "large-v3-turbo-q5_0".into(),
            kind: ModelKind::Whisper,
            file_name: "ggml-large-v3-turbo-q5_0.bin".into(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin".into(),
            size_bytes: 574_041_195,
            speed: 5,
            accuracy: 4,
            note: "fast, 5-bit quantized (~574 MB)".into(),
            languages: "multilingual".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: Some(1000),
        },
        // Vision models: pair of (model, projector) under one id
        CatalogEntry {
            id: "bonsai-27b".into(),
            kind: ModelKind::Vision,
            file_name: vision_spec.file_name,
            url: vision_spec.url,
            size_bytes: 3_803_452_480,
            speed: 3,
            accuracy: 4,
            note: "recommended — 1-bit 27B vision model (~4.4 GB total)".into(),
            languages: "multilingual".into(),
            mmproj_file_name: Some(proj_spec.file_name),
            mmproj_url: Some(proj_spec.url),
            mmproj_size_bytes: Some(629_246_880),
            vram_mb: Some(6000),
        },
        CatalogEntry {
            id: "gemma-3-4b-it".into(),
            kind: ModelKind::Vision,
            file_name: "gemma-3-4b-it-Q4_K_M.gguf".into(),
            url: "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf".into(),
            size_bytes: 2_489_757_856,
            speed: 4,
            accuracy: 3,
            note: "less accurate, fits 6 GB (~3.3 GB total)".into(),
            languages: "multilingual".into(),
            mmproj_file_name: Some("gemma-3-4b-it-mmproj-f16.gguf".into()),
            mmproj_url: Some("https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf".into()),
            mmproj_size_bytes: Some(851_251_104),
            vram_mb: Some(4500),
        },
        CatalogEntry {
            id: "qwen3.5-9b".into(),
            kind: ModelKind::Vision,
            file_name: "Qwen3.5-9B-UD-Q4_K_XL.gguf".into(),
            url: "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf".into(),
            size_bytes: 5_966_095_584,
            speed: 3,
            accuracy: 5,
            // The only local model here that can actually write a script. Bonsai-27B describes a
            // frame well but its Q1_0 quantisation cannot hold a 15k-token speech digest: given a
            // prompt naming "Northwest Hills" nineteen times it replied that no such footage
            // existed. Qwen2.5-VL is a different failure — `vision: Unknown Token Type`, a chat
            // template this build cannot apply — so neither 3B nor 7B of that family works at all.
            note: "writes scripts as well as describes; needs ~7 GB".into(),
            languages: "multilingual".into(),
            // Upstream's own name, unprefixed, because a paired entry counts as installed only
            // when *both* files resolve — and discovery matches on the file name, so renaming it
            // here would hide a copy already on disk (it did: this read "not installed" beside a
            // 5.7 GB file it was looking straight at).
            mmproj_file_name: Some("mmproj-F16.gguf".into()),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/mmproj-F16.gguf".into()),
            mmproj_size_bytes: Some(918_166_080),
            vram_mb: Some(7500),
        },
        CatalogEntry {
            id: "qwen2.5-vl-7b".into(),
            kind: ModelKind::Vision,
            file_name: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf".into(),
            url: "https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf".into(),
            size_bytes: 4_683_072_032,
            speed: 3,
            accuracy: 4,
            note: "strong visual details, fits 8 GB (~5.5 GB total)".into(),
            languages: "multilingual".into(),
            mmproj_file_name: Some("mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf".into()),
            mmproj_url: Some("https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf".into()),
            mmproj_size_bytes: Some(853_119_712),
            vram_mb: Some(6500),
        },
        CatalogEntry {
            id: "qwen2.5-vl-3b".into(),
            kind: ModelKind::Vision,
            file_name: "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf".into(),
            url: "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf".into(),
            size_bytes: 1_929_901_056,
            speed: 5,
            accuracy: 3,
            note: "fastest vision, fits 4 GB (~2.8 GB total)".into(),
            languages: "multilingual".into(),
            mmproj_file_name: Some("mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf".into()),
            mmproj_url: Some("https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf".into()),
            mmproj_size_bytes: Some(844_757_728),
            vram_mb: Some(3500),
        },
        // Search embeddings: embeddinggemma 300M
        CatalogEntry {
            id: "embeddinggemma-300M-Q8_0".into(),
            kind: ModelKind::Embedding,
            file_name: embed_spec.file_name,
            url: embed_spec.url,
            size_bytes: 333_590_944,
            speed: 5,
            accuracy: 5,
            note: "768-dim text embeddings, runs on CPU (~334 MB)".into(),
            languages: "multilingual".into(),
            mmproj_file_name: None,
            mmproj_url: None,
            mmproj_size_bytes: None,
            vram_mb: None,
        },
    ]
}

/// Find a catalog entry by ID, alias, or exact file name.
pub fn find_entry(id: &str) -> Option<CatalogEntry> {
    let trimmed = id.trim();
    let cat = catalog();
    if let Some(e) = cat.iter().find(|e| e.id.eq_ignore_ascii_case(trimmed)) {
        return Some(e.clone());
    }
    // Aliases
    if trimmed.eq_ignore_ascii_case("embeddinggemma") {
        return cat.iter().find(|e| e.id == "embeddinggemma-300M-Q8_0").cloned();
    }
    if trimmed.eq_ignore_ascii_case("bonsai-27b-projector") || trimmed.eq_ignore_ascii_case("bonsai-27b-mmproj") {
        return cat.iter().find(|e| e.id == "bonsai-27b").cloned();
    }
    cat.into_iter().find(|e| {
        e.file_name.eq_ignore_ascii_case(trimmed)
            || e.mmproj_file_name.as_deref().is_some_and(|f| f.eq_ignore_ascii_case(trimmed))
    })
}

/// Query installation status for every catalog entry.
pub fn status(models_dir: &Path, search_paths: &[PathBuf]) -> Vec<ModelStatus> {
    let roots = search_roots(models_dir, search_paths);
    catalog()
        .into_iter()
        .map(|entry| {
            if let Some(proj_name) = &entry.mmproj_file_name {
                let model_own = models_dir.join(&entry.file_name);
                let proj_own = models_dir.join(proj_name);
                let model_in_own = model_own.is_file() && model_own.metadata().map(|m| m.len() > 0).unwrap_or(false);
                let proj_in_own = proj_own.is_file() && proj_own.metadata().map(|m| m.len() > 0).unwrap_or(false);

                let model_path = if model_in_own { Some(model_own) } else { find(&roots, &entry.file_name) };
                let proj_path = if proj_in_own { Some(proj_own) } else { find(&roots, proj_name) };

                if let (Some(m_p), Some(_p_p)) = (model_path, proj_path) {
                    let in_own = m_p.starts_with(models_dir);
                    return ModelStatus { entry, installed_path: Some(m_p), in_own_dir: in_own, partial_bytes: None };
                }

                // If not fully installed, check for partial files or already downloaded individual files
                let m_part = models_dir.join(format!("{}.part", entry.file_name));
                let p_part = models_dir.join(format!("{proj_name}.part"));
                let mut partial = 0u64;
                if let Ok(m) = m_part.metadata() {
                    partial += m.len();
                }
                if let Ok(m) = p_part.metadata() {
                    partial += m.len();
                }
                if model_in_own && let Ok(m) = models_dir.join(&entry.file_name).metadata() {
                    partial += m.len();
                }
                if proj_in_own && let Ok(m) = models_dir.join(proj_name).metadata() {
                    partial += m.len();
                }
                let partial_bytes = if partial > 0 { Some(partial) } else { None };
                return ModelStatus { entry, installed_path: None, in_own_dir: false, partial_bytes };
            }

            let own = models_dir.join(&entry.file_name);
            if own.is_file() && own.metadata().map(|m| m.len() > 0).unwrap_or(false) {
                return ModelStatus { entry, installed_path: Some(own), in_own_dir: true, partial_bytes: None };
            }
            if let Some(found) = find(&roots, &entry.file_name) {
                let in_own = found.starts_with(models_dir);
                return ModelStatus { entry, installed_path: Some(found), in_own_dir: in_own, partial_bytes: None };
            }
            let part = models_dir.join(format!("{}.part", entry.file_name));
            let partial_bytes = part.metadata().ok().map(|m| m.len()).filter(|&l| l > 0);
            ModelStatus { entry, installed_path: None, in_own_dir: false, partial_bytes }
        })
        .collect()
}

/// Remove a model file (and any `.part` file) from GhostReel's own `models_dir`.
/// Never touches copies found in search_paths or external apps like GhostPen or LM Studio.
pub fn remove(models_dir: &Path, id: &str) -> Result<PathBuf, Error> {
    let entry = find_entry(id);
    let (file_name, proj_file_name) = if let Some(ref e) = entry {
        (e.file_name.clone(), e.mmproj_file_name.clone())
    } else if let Ok(spec) = whisper(id) {
        (spec.file_name, None)
    } else {
        return Err(Error::NotFound(format!("unknown model '{id}'")));
    };

    let dest = models_dir.join(&file_name);
    let part = models_dir.join(format!("{file_name}.part"));
    let mut removed = false;

    if dest.is_file() {
        std::fs::remove_file(&dest).map_err(|e| Error::Io(dest.clone(), e))?;
        removed = true;
    }
    if part.is_file() {
        let _ = std::fs::remove_file(&part);
        removed = true;
    }

    if let Some(proj_name) = proj_file_name {
        let p_dest = models_dir.join(&proj_name);
        let p_part = models_dir.join(format!("{proj_name}.part"));
        if p_dest.is_file() {
            let _ = std::fs::remove_file(&p_dest);
            removed = true;
        }
        if p_part.is_file() {
            let _ = std::fs::remove_file(&p_part);
            removed = true;
        }
    }

    if removed {
        Ok(dest)
    } else {
        Err(Error::NotFound(format!("model '{id}' ({file_name}) is not installed in {}", models_dir.display())))
    }
}

/// whisper.cpp ggml model by name (`large-v3-turbo`, `small`, `base.en`, …).
pub fn whisper(model: &str) -> Result<ModelSpec, Error> {
    let name = model.trim();
    let valid = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_');
    if !valid || name.contains("..") {
        return Err(Error::Invalid(format!("invalid whisper model name '{model}'")));
    }
    let file_name = format!("ggml-{name}.bin");
    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file_name}");
    Ok(ModelSpec { file_name, url })
}

fn hf(repo: &str, file: &str) -> ModelSpec {
    ModelSpec { file_name: file.to_string(), url: format!("https://huggingface.co/{repo}/resolve/main/{file}") }
}

/// Resolve a vision model pair by catalog ID.
pub fn vision_pair(id: &str) -> Option<(ModelSpec, ModelSpec)> {
    let entry = find_entry(id)?;
    if entry.kind != ModelKind::Vision {
        return None;
    }
    let mmproj = entry.mmproj_spec()?;
    Some((entry.spec(), mmproj))
}

/// Default local vision model: Bonsai-27B at 1-bit (~3.6 GB) + its image projector (S0).
pub fn bonsai_vision() -> (ModelSpec, ModelSpec) {
    vision_pair("bonsai-27b").unwrap_or_else(|| {
        (
            hf("prism-ml/Bonsai-27B-gguf", "Bonsai-27B-Q1_0.gguf"),
            hf("prism-ml/Bonsai-27B-gguf", "Bonsai-27B-mmproj-Q8_0.gguf"),
        )
    })
}

/// The one embedding model GhostReel uses everywhere (plan D5).
pub fn embeddinggemma() -> ModelSpec {
    hf("ggml-org/embeddinggemma-300M-GGUF", "embeddinggemma-300M-Q8_0.gguf")
}

/// Directories where other local apps keep compatible model files.
pub fn well_known_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(d) = dirs::data_dir() {
        out.push(d.join("GhostPen").join("models")); // whisper ggml models
    }
    if let Some(h) = dirs::home_dir() {
        out.push(h.join(".lmstudio").join("models")); // GGUF models
    }
    out
}

/// Look for `file_name` in `roots` (a few levels deep).
pub fn find(roots: &[PathBuf], file_name: &str) -> Option<PathBuf> {
    fn walk(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
        let entries = std::fs::read_dir(dir).ok()?;
        let mut subdirs = Vec::new();
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                subdirs.push(p);
            } else if e.file_name().to_string_lossy().eq_ignore_ascii_case(name)
                && e.metadata().map(|m| m.len() > 0).unwrap_or(false)
            {
                return Some(p);
            }
        }
        if depth == 0 {
            return None;
        }
        subdirs.iter().find_map(|d| walk(d, name, depth - 1))
    }
    roots.iter().find_map(|r| walk(r, file_name, 4))
}

/// Search order: GhostReel's own models dir, configured paths, well-known app dirs.
pub fn search_roots(models_dir: &Path, configured: &[PathBuf]) -> Vec<PathBuf> {
    let mut roots = vec![models_dir.to_path_buf()];
    roots.extend(configured.iter().cloned());
    roots.extend(well_known_dirs());
    roots
}

/// Download `spec` into `dir` (resuming a previous `.part`), reporting `(bytes_done, bytes_total)`.
pub async fn download(
    spec: &ModelSpec,
    dir: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, Error> {
    tokio::fs::create_dir_all(dir).await.map_err(|e| Error::Io(dir.to_path_buf(), e))?;
    let dest = dir.join(&spec.file_name);
    let part = dir.join(format!("{}.part", spec.file_name));
    let already = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| Error::Invalid(e.to_string()))?;
    let mut req = client.get(&spec.url);
    if already > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={already}-"));
    }
    let resp = req.send().await.map_err(|e| Error::Download(format!("{}: {e}", spec.url)))?;
    let status = resp.status();
    let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
    if !status.is_success() {
        return Err(Error::Download(format!("{}: HTTP {status}", spec.url)));
    }
    let start = if resumed { already } else { 0 };
    let total = resp.content_length().map(|l| l + start);

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&part)
        .await
        .map_err(|e| Error::Io(part.clone(), e))?;
    let mut done = start;
    on_progress(done, total);
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| Error::Download(format!("{}: {e}", spec.url)))?;
        file.write_all(&chunk).await.map_err(|e| Error::Io(part.clone(), e))?;
        done += chunk.len() as u64;
        on_progress(done, total);
    }
    file.flush().await.map_err(|e| Error::Io(part.clone(), e))?;
    drop(file);
    if let Some(t) = total
        && done != t
    {
        return Err(Error::Download(format!("{}: incomplete ({done} of {t} bytes)", spec.url)));
    }
    tokio::fs::rename(&part, &dest).await.map_err(|e| Error::Io(dest.clone(), e))?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn whisper_names() {
        let s = whisper("large-v3-turbo").unwrap();
        assert_eq!(s.file_name, "ggml-large-v3-turbo.bin");
        assert!(s.url.ends_with("/ggml-large-v3-turbo.bin"));
        assert!(whisper("../etc/passwd").is_err());
        assert!(whisper("").is_err());
    }

    #[test]
    fn finds_existing_copies() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("lm/org/repo");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("model.gguf"), b"x").unwrap();
        std::fs::write(tmp.path().join("empty.bin"), b"").unwrap();
        let roots = vec![tmp.path().join("missing"), tmp.path().to_path_buf()];
        assert!(find(&roots, "MODEL.gguf").is_some());
        assert!(find(&roots, "empty.bin").is_none(), "zero-byte files are not models");
    }

    /// Serves `body`, honouring `Range: bytes=N-`.
    async fn serve_file(body: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                let body = body.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_lowercase();
                    let from = req.lines().find_map(|l| l.strip_prefix("range: bytes=")).and_then(|r| {
                        r.trim_end_matches('-')
                            .trim_end_matches("-\r")
                            .trim()
                            .trim_end_matches('-')
                            .parse::<usize>()
                            .ok()
                    });
                    let (status, slice) = match from {
                        Some(f) => ("206 Partial Content", &body[f..]),
                        None => ("200 OK", &body[..]),
                    };
                    let head =
                        format!("HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", slice.len());
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.write_all(slice).await;
                });
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn downloads_and_resumes() {
        let body: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let base = serve_file(body.clone()).await;
        let tmp = tempfile::tempdir().unwrap();
        let spec = ModelSpec { file_name: "m.bin".into(), url: format!("{base}/m.bin") };

        // A previous interrupted download left the first 30 000 bytes.
        std::fs::write(tmp.path().join("m.bin.part"), &body[..30_000]).unwrap();
        let mut last = (0, None);
        let path = download(&spec, tmp.path(), |d, t| last = (d, t)).await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert_eq!(last, (100_000, Some(100_000)));
        assert!(!tmp.path().join("m.bin.part").exists());
    }

    #[test]
    fn catalog_has_expected_entries() {
        let cat = catalog();
        assert!(cat.len() >= 14);
        assert!(cat.iter().any(|e| e.id == "tiny" && e.kind == ModelKind::Whisper));
        assert!(cat.iter().any(|e| e.id == "large-v3-turbo" && e.size_bytes == 1_624_555_275));
        assert!(
            cat.iter().any(|e| e.id == "bonsai-27b" && e.kind == ModelKind::Vision && e.mmproj_file_name.is_some())
        );
        assert!(
            cat.iter().any(|e| e.id == "gemma-3-4b-it" && e.kind == ModelKind::Vision && e.mmproj_file_name.is_some())
        );
        assert!(
            cat.iter().any(|e| e.id == "qwen2.5-vl-7b" && e.kind == ModelKind::Vision && e.mmproj_file_name.is_some())
        );
        // The standalone script writer. Without a vision pair it cannot be selected at all, and
        // without it standalone has no local model that can draft: Bonsai refuses on a long
        // digest and Qwen2.5-VL's template does not load.
        assert!(
            cat.iter().any(|e| e.id == "qwen3.5-9b" && e.kind == ModelKind::Vision && e.mmproj_file_name.is_some())
        );
        assert!(cat.iter().any(|e| e.id == "embeddinggemma-300M-Q8_0" && e.kind == ModelKind::Embedding));

        let (b_model, b_proj) = bonsai_vision();
        assert_eq!(b_model.file_name, "Bonsai-27B-Q1_0.gguf");
        assert_eq!(b_proj.file_name, "Bonsai-27B-mmproj-Q8_0.gguf");

        let (g_model, g_proj) = vision_pair("gemma-3-4b-it").unwrap();
        assert_eq!(g_model.file_name, "gemma-3-4b-it-Q4_K_M.gguf");
        assert_eq!(g_proj.file_name, "gemma-3-4b-it-mmproj-f16.gguf");
    }

    #[test]
    fn find_entry_aliases() {
        assert_eq!(find_entry("tiny").unwrap().file_name, "ggml-tiny.bin");
        assert_eq!(find_entry("embeddinggemma").unwrap().id, "embeddinggemma-300M-Q8_0");
        assert_eq!(find_entry("bonsai-27b-projector").unwrap().id, "bonsai-27b");
        assert_eq!(find_entry("bonsai-27b-mmproj").unwrap().id, "bonsai-27b");
        assert_eq!(find_entry("Bonsai-27B-Q1_0.gguf").unwrap().id, "bonsai-27b");
        assert_eq!(find_entry("Bonsai-27B-mmproj-Q8_0.gguf").unwrap().id, "bonsai-27b");
        assert_eq!(find_entry("gemma-3-4b-it").unwrap().file_name, "gemma-3-4b-it-Q4_K_M.gguf");
        assert!(find_entry("nonexistent-model-xyz").is_none());
    }

    #[test]
    fn status_and_remove() {
        let dir = tempfile::tempdir().unwrap();
        let models_dir = dir.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();

        // Model not installed anywhere (large-v3-turbo-q5_0)
        let st = status(&models_dir, &[]);
        let q5_st = st.iter().find(|s| s.entry.id == "large-v3-turbo-q5_0").unwrap();
        assert!(q5_st.installed_path.is_none());
        assert!(!q5_st.in_own_dir);
        assert!(q5_st.partial_bytes.is_none());

        // Create a fake partial file for q5_0
        std::fs::write(models_dir.join("ggml-large-v3-turbo-q5_0.bin.part"), b"partial data").unwrap();
        let st = status(&models_dir, &[]);
        let q5_st = st.iter().find(|s| s.entry.id == "large-v3-turbo-q5_0").unwrap();
        assert_eq!(q5_st.partial_bytes, Some(12));

        // Create full file
        std::fs::write(models_dir.join("ggml-large-v3-turbo-q5_0.bin"), b"full data").unwrap();
        let st = status(&models_dir, &[]);
        let q5_st = st.iter().find(|s| s.entry.id == "large-v3-turbo-q5_0").unwrap();
        assert_eq!(q5_st.installed_path, Some(models_dir.join("ggml-large-v3-turbo-q5_0.bin")));
        assert!(q5_st.in_own_dir);

        // Remove
        let removed = remove(&models_dir, "large-v3-turbo-q5_0").unwrap();
        assert_eq!(removed, models_dir.join("ggml-large-v3-turbo-q5_0.bin"));
        assert!(!models_dir.join("ggml-large-v3-turbo-q5_0.bin").exists());
        assert!(!models_dir.join("ggml-large-v3-turbo-q5_0.bin.part").exists());

        // Removing again gives error
        assert!(remove(&models_dir, "large-v3-turbo-q5_0").is_err());

        // External model: `remove` on GhostReel's own dir when file only exists externally does nothing and errors
        assert!(!models_dir.join("ggml-tiny.bin").exists());
        let tiny_st = status(&models_dir, &[]).into_iter().find(|s| s.entry.id == "tiny").unwrap();
        if !tiny_st.in_own_dir {
            // Never removes files outside models_dir
            assert!(remove(&models_dir, "tiny").is_err());
        }

        // Vision pair status: only installed when BOTH model and projector files exist
        let gemma_st = status(&models_dir, &[]).into_iter().find(|s| s.entry.id == "gemma-3-4b-it").unwrap();
        assert!(gemma_st.installed_path.is_none());

        // Model only: still not installed
        std::fs::write(models_dir.join("gemma-3-4b-it-Q4_K_M.gguf"), b"model data").unwrap();
        let gemma_st = status(&models_dir, &[]).into_iter().find(|s| s.entry.id == "gemma-3-4b-it").unwrap();
        assert!(gemma_st.installed_path.is_none());
        assert!(gemma_st.partial_bytes.is_some());

        // Both model and projector: installed!
        std::fs::write(models_dir.join("gemma-3-4b-it-mmproj-f16.gguf"), b"proj data").unwrap();
        let gemma_st = status(&models_dir, &[]).into_iter().find(|s| s.entry.id == "gemma-3-4b-it").unwrap();
        assert!(gemma_st.installed_path.is_some());
        assert!(gemma_st.in_own_dir);
        assert!(gemma_st.partial_bytes.is_none());

        // Removing gemma-3-4b-it removes both files!
        remove(&models_dir, "gemma-3-4b-it").unwrap();
        assert!(!models_dir.join("gemma-3-4b-it-Q4_K_M.gguf").exists());
        assert!(!models_dir.join("gemma-3-4b-it-mmproj-f16.gguf").exists());
    }

    #[test]
    fn effective_models_dir_respects_config() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths { config_file: dir.path().join("config.toml"), data_dir: dir.path().join("data") };
        let mut cfg = Config::default();
        assert_eq!(effective_models_dir(&paths, &cfg), paths.models_dir());

        let custom = dir.path().join("custom_models");
        cfg.models.dir = Some(custom.clone());
        assert_eq!(effective_models_dir(&paths, &cfg), custom);
    }
}
