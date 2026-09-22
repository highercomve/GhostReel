//! Everything an indexing run needs from the machine: helper binaries and the resolved AI
//! backends (plan §2a). Resolved once per run so starting/stopping GhostPen or highllama is
//! picked up by the next run without restarting GhostReel.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::config::{Backend, Config};
use crate::doctor::locate;
use crate::models::{self, ModelSpec};
use crate::paths::Paths;
use crate::probe::{self, Target};
use crate::stt::Engine;

/// How transcription will run this time.
#[derive(Debug, Clone)]
pub enum SttSetup {
    Ready(Engine),
    /// Local whisper is chosen but the model file must be downloaded first.
    NeedsModel {
        spec: ModelSpec,
        dir: PathBuf,
        asr_bin: PathBuf,
        language: String,
    },
    /// Can't transcribe now (jobs stay pending and are retried by a later run).
    Unavailable(String),
}

impl SttSetup {
    pub fn describe(&self) -> String {
        match self {
            SttSetup::Ready(e) => e.label(),
            SttSetup::NeedsModel { spec, .. } => format!("local whisper (downloading {})", spec.file_name),
            SttSetup::Unavailable(why) => format!("unavailable: {why}"),
        }
    }
}

/// How frame descriptions will run this time.
#[derive(Debug, Clone)]
pub enum VisionSetup {
    Server(crate::vision::ServerVision),
    /// Local `ghostreel-llm`; `missing` model files are downloaded before the helper starts.
    Local {
        helper: PathBuf,
        models_dir: PathBuf,
        model: crate::models::ModelSpec,
        mmproj: crate::models::ModelSpec,
        found: Vec<PathBuf>,
        /// Context window / KV cache / flash attention for this capability.
        runtime: crate::vision::HelperRuntime,
    },
    /// Coding-agent CLI (claude / agy / opencode).
    Cli(crate::config::CliAgentConfig),
    Unavailable(String),
}

impl VisionSetup {
    pub fn describe(&self) -> String {
        match self {
            VisionSetup::Server(s) => {
                format!("{} @ {}", if s.model.is_empty() { "vision server" } else { &s.model }, s.url)
            }
            VisionSetup::Local { model, runtime, .. } => {
                format!(
                    "local {} ({} ctx, kv {}, flash {})",
                    model.file_name, runtime.ctx_tokens, runtime.kv_cache, runtime.flash_attn
                )
            }
            VisionSetup::Cli(cfg) => {
                if cfg.model.is_empty() {
                    format!("CLI {}", cfg.tool)
                } else {
                    format!("CLI {} ({})", cfg.tool, cfg.model)
                }
            }
            VisionSetup::Unavailable(why) => format!("unavailable: {why}"),
        }
    }
}

/// How text embeddings will be computed this time.
#[derive(Debug, Clone)]
pub enum EmbedSetup {
    Server { url: String, model: String },
    Local { helper: PathBuf, models_dir: PathBuf, spec: ModelSpec, found: Option<PathBuf> },
    Unavailable(String),
}

impl EmbedSetup {
    pub fn describe(&self) -> String {
        match self {
            EmbedSetup::Server { url, model } => format!("{model} @ {url}"),
            EmbedSetup::Local { .. } => "local embeddinggemma (CPU)".into(),
            EmbedSetup::Unavailable(why) => format!("unavailable: {why}"),
        }
    }
}

/// Start an embedder for `setup`, downloading the local model if needed.
pub async fn start_embedder(
    setup: &EmbedSetup,
    on_download: impl FnMut(u64, Option<u64>),
) -> Result<crate::embed::Embedder, String> {
    use crate::embed::Embedder;
    match setup {
        EmbedSetup::Server { url, model } => Ok(Embedder::server(url, model)),
        EmbedSetup::Unavailable(why) => Err(why.clone()),
        EmbedSetup::Local { helper, models_dir, spec, found } => {
            let path = match found {
                Some(p) => p.clone(),
                None => models::download(spec, models_dir, on_download).await.map_err(|e| e.to_string())?,
            };
            // Embeddings keep an f16 cache: the model is tiny and the vectors must stay comparable.
            let models = crate::vision::LocalModels {
                helper: helper.clone(),
                vision: None,
                embed: Some(path),
                cpu: true,
                runtime: crate::vision::HelperRuntime { kv_cache: "f16".into(), ..Default::default() },
                concurrency: 1,
            };
            Embedder::local(&models).await.map_err(|e| e.to_string())
        }
    }
}

#[derive(Debug, Clone)]
pub struct Runtime {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub stt: SttSetup,
    /// Where frames and other derived files live.
    pub data_dir: PathBuf,
    /// Keyframe extraction settings; `None` disables the frames stage (tests).
    pub frames: Option<crate::frames::FrameOptions>,
    /// Frames described at once against a server or local helper (`vision.describe_concurrency`).
    /// Memory-bandwidth bound, so a batch amortises reading the weights over several answers.
    pub describe_concurrency: usize,
    pub vision: VisionSetup,
    pub embed: EmbedSetup,
    /// How steady the camera is, measured alongside keyframes. `None` skips it (tests).
    pub steadiness: Option<SteadinessOptions>,
    /// Measure which audio track carries the speech and who is off the microphone. Off in tests,
    /// where the fixtures are not real media and the measurement would run ffmpeg on them.
    pub measure_audio: bool,
}

/// Window and stride used when measuring how steady a video is, from `[script]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteadinessOptions {
    pub window_s: f64,
    pub stride_s: f64,
}

impl From<&crate::config::ScriptConfig> for Option<SteadinessOptions> {
    fn from(c: &crate::config::ScriptConfig) -> Self {
        (c.max_shake_jerk > 0.0).then_some(SteadinessOptions { window_s: c.shake_window_s, stride_s: c.shake_stride_s })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSummary {
    pub stt: String,
}

impl Runtime {
    pub fn summary(&self) -> RuntimeSummary {
        RuntimeSummary { stt: self.stt.describe() }
    }
}

/// Find the local transcription helper. Dev builds also look in the sibling `release/` dir,
/// because `scripts/build-helpers.sh` builds it in release mode only.
pub fn locate_asr() -> Option<PathBuf> {
    locate_helper("ghostreel-asr")
}

pub fn locate_helper(name: &str) -> Option<PathBuf> {
    locate(name).or_else(|| {
        if !cfg!(debug_assertions) {
            return None;
        }
        let exe = std::env::current_exe().ok()?;
        let target = exe.parent()?.parent()?;
        let file = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
        let p = target.join("release").join(file);
        p.is_file().then_some(p)
    })
}

pub async fn resolve(paths: &Paths, config: &Config) -> Result<Runtime, crate::Error> {
    let ffmpeg =
        locate("ffmpeg").ok_or_else(|| crate::Error::Invalid("ffmpeg not found (see `ghostreel doctor`)".into()))?;
    let ffprobe =
        locate("ffprobe").ok_or_else(|| crate::Error::Invalid("ffprobe not found (see `ghostreel doctor`)".into()))?;
    let (stt, vision, embed, gpus) = tokio::join!(
        resolve_stt(paths, config),
        resolve_vision(paths, config),
        resolve_embed(paths, config),
        crate::doctor::nvidia_gpus()
    );
    let total_vram = gpus.first().map(|g| g.vram_total_mib);
    let model_mb = crate::models::find_entry(&config.vision.local_model).and_then(|e| e.vram_mb);
    let describe_concurrency =
        crate::doctor::calculate_describe_concurrency(config.vision.describe_concurrency, total_vram, model_mb);
    Ok(Runtime {
        ffmpeg,
        ffprobe,
        stt,
        data_dir: paths.data_dir.clone(),
        frames: Some(crate::frames::FrameOptions::from_config(&config.frames)),
        describe_concurrency,
        vision,
        embed,
        steadiness: (&config.script).into(),
        measure_audio: true,
    })
}

pub async fn resolve_embed(paths: &Paths, config: &Config) -> EmbedSetup {
    let cfg = &config.embed;
    let probe = match cfg.backend {
        Backend::Local => None,
        _ => Some(probe::embeddings(&probe::probe_client(), &cfg.url, &cfg.model).await),
    };
    let resolution = probe::resolve(cfg.backend, probe);
    match resolution.target {
        Target::Server => {
            return EmbedSetup::Server { url: cfg.url.trim_end_matches('/').to_string(), model: cfg.model.clone() };
        }
        Target::Unavailable => return EmbedSetup::Unavailable(resolution.reason),
        Target::Local => {}
    }
    let Some(helper) = locate_helper("ghostreel-llm") else {
        return EmbedSetup::Unavailable("local model helper ghostreel-llm not found".into());
    };
    let models_dir = models::effective_models_dir(paths, config);
    let spec = models::embeddinggemma();
    let found = models::find(&models::search_roots(&models_dir, &config.models.search_paths), &spec.file_name);
    EmbedSetup::Local { helper, models_dir, spec, found }
}

/// Frame descriptions (indexing): `[vision]`.
pub async fn resolve_vision(paths: &Paths, config: &Config) -> VisionSetup {
    resolve_llm(paths, config, &config.vision).await
}

/// Script chat: `[chat_model]`, or `[vision]` with a bigger window on older configs.
pub async fn resolve_chat(paths: &Paths, config: &Config) -> VisionSetup {
    resolve_llm(paths, config, &config.chat_model()).await
}

async fn resolve_llm(paths: &Paths, config: &Config, cfg: &crate::config::VisionConfig) -> VisionSetup {
    // CLI backend: resolve binary; `auto` never picks this.
    if cfg.backend == Backend::Cli {
        let agent_cfg = cfg.cli.clone();
        let tool = &agent_cfg.tool;
        if tool.is_empty() {
            return VisionSetup::Unavailable("vision.cli.tool is not set (set to claude, agy, or opencode)".into());
        }
        return match crate::cliagent::find_binary(&agent_cfg) {
            Some(_) => VisionSetup::Cli(agent_cfg),
            None => VisionSetup::Unavailable(format!("CLI tool '{tool}' not found on PATH")),
        };
    }

    let probe = match cfg.backend {
        Backend::Local | Backend::Cli => None,
        _ => Some(probe::vision(&probe::probe_client(), &cfg.url, &cfg.model).await),
    };
    let model_id = probe.as_ref().and_then(|p| p.model.clone()).unwrap_or_default();
    let resolution = probe::resolve(cfg.backend, probe);
    match resolution.target {
        Target::Server => {
            let model = if cfg.model.is_empty() { model_id } else { cfg.model.clone() };
            return VisionSetup::Server(crate::vision::ServerVision::new(&cfg.url, &model, &cfg.api_key));
        }
        Target::Unavailable => return VisionSetup::Unavailable(resolution.reason),
        Target::Local => {}
    }
    let (model, mmproj) = match models::vision_pair(&cfg.local_model) {
        Some(pair) => pair,
        None => {
            return VisionSetup::Unavailable(format!("unknown vision model '{}'", cfg.local_model));
        }
    };
    let Some(helper) = locate_helper("ghostreel-llm") else {
        return VisionSetup::Unavailable("local model helper ghostreel-llm not found".into());
    };
    let models_dir = models::effective_models_dir(paths, config);
    let roots = models::search_roots(&models_dir, &config.models.search_paths);
    let found = [&model, &mmproj].iter().filter_map(|s| models::find(&roots, &s.file_name)).collect();
    VisionSetup::Local { helper, models_dir, model, mmproj, found, runtime: cfg.into() }
}

pub async fn resolve_stt(paths: &Paths, config: &Config) -> SttSetup {
    let cfg = &config.stt;
    let probe = match cfg.backend {
        Backend::Local => None,
        _ => Some(probe::stt(&probe::probe_client(), &cfg.url).await),
    };
    let resolution = probe::resolve(cfg.backend, probe);
    match resolution.target {
        Target::Server => return SttSetup::Ready(Engine::server(&cfg.url)),
        Target::Unavailable => return SttSetup::Unavailable(resolution.reason),
        Target::Local => {}
    }
    let model = if cfg.model.trim().eq_ignore_ascii_case("auto") {
        auto_whisper_model(&crate::doctor::nvidia_gpus().await).to_string()
    } else {
        cfg.model.clone()
    };
    let models_dir = models::effective_models_dir(paths, config);
    local_stt(&models_dir, &config.models.search_paths, &model)
}

/// large-v3-turbo (1.6 GB, ~2 GB VRAM) is far more accurate — names, accents, Spanish — but on CPU
/// or small GPUs `small` (466 MB) keeps transcription fast.
pub fn auto_whisper_model(gpus: &[crate::doctor::Gpu]) -> &'static str {
    if gpus.iter().any(|g| g.vram_total_mib >= 6000) { "large-v3-turbo" } else { "small" }
}

fn local_stt(models_dir: &Path, search_paths: &[PathBuf], model: &str) -> SttSetup {
    let Some(asr_bin) = locate_asr() else {
        return SttSetup::Unavailable("local transcription helper ghostreel-asr not found".into());
    };
    let spec = match models::whisper(model) {
        Ok(s) => s,
        Err(e) => return SttSetup::Unavailable(e.to_string()),
    };
    let language = "auto".to_string();
    match models::find(&models::search_roots(models_dir, search_paths), &spec.file_name) {
        Some(model) => SttSetup::Ready(Engine::Local { asr_bin, model, language }),
        None => SttSetup::NeedsModel { spec, dir: models_dir.to_path_buf(), asr_bin, language },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::Gpu;

    #[test]
    fn whisper_model_follows_vram() {
        let gpu = |mib| Gpu { name: "g".into(), vram_total_mib: mib, vram_used_mib: 0, driver: "d".into() };
        assert_eq!(auto_whisper_model(&[gpu(8192)]), "large-v3-turbo");
        assert_eq!(auto_whisper_model(&[gpu(4096)]), "small");
        assert_eq!(auto_whisper_model(&[]), "small");
    }

    #[tokio::test]
    async fn resolve_vision_unknown_model_is_unavailable() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths { config_file: dir.path().join("config.toml"), data_dir: dir.path().join("data") };
        let mut cfg = Config::default();
        cfg.vision.backend = Backend::Local;
        cfg.vision.local_model = "nonexistent-model-xyz".into();

        let setup = resolve_vision(&paths, &cfg).await;
        match setup {
            VisionSetup::Unavailable(why) => {
                assert!(why.contains("unknown vision model 'nonexistent-model-xyz'"), "{why}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn resolve_vision_known_models_pick_correct_specs() {
        let dir = tempfile::tempdir().unwrap();
        let fake_helper = dir.path().join("fake_llm");
        std::fs::write(&fake_helper, b"fake helper").unwrap();
        // Set env var so locate_helper finds it
        unsafe {
            std::env::set_var("GHOSTREEL_GHOSTREEL-LLM", &fake_helper);
        }

        let paths = Paths { config_file: dir.path().join("config.toml"), data_dir: dir.path().join("data") };

        // Test default bonsai-27b
        let mut cfg = Config::default();
        cfg.vision.backend = Backend::Local;
        cfg.vision.local_model = "bonsai-27b".into();
        let setup = resolve_vision(&paths, &cfg).await;
        match setup {
            VisionSetup::Local { model, mmproj, .. } => {
                assert_eq!(model.file_name, "Bonsai-27B-Q1_0.gguf");
                assert_eq!(mmproj.file_name, "Bonsai-27B-mmproj-Q8_0.gguf");
            }
            other => panic!("expected Local, got {other:?}"),
        }

        // Test gemma-3-4b-it
        cfg.vision.local_model = "gemma-3-4b-it".into();
        let setup = resolve_vision(&paths, &cfg).await;
        match setup {
            VisionSetup::Local { model, mmproj, .. } => {
                assert_eq!(model.file_name, "gemma-3-4b-it-Q4_K_M.gguf");
                assert_eq!(mmproj.file_name, "gemma-3-4b-it-mmproj-f16.gguf");
            }
            other => panic!("expected Local, got {other:?}"),
        }

        // Clean up env var
        unsafe {
            std::env::remove_var("GHOSTREEL_GHOSTREEL-LLM");
        }
    }
}
