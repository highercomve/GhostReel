//! `ghostreel doctor`: everything needed to index, checked in one report (app + CLI share it).

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;

use crate::config::{Backend, Config};
use crate::db::Db;
use crate::paths::Paths;
use crate::probe::{self, Resolution, Target};

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    pub name: String,
    pub path: Option<PathBuf>,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Gpu {
    pub name: String,
    pub vram_total_mib: u64,
    pub vram_used_mib: u64,
    pub driver: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DbStatus {
    pub path: PathBuf,
    pub ok: bool,
    pub schema_version: Option<u32>,
    pub sqlite_vec: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelFile {
    pub role: String,
    /// File name pattern searched for.
    pub pattern: String,
    pub found: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub version: &'static str,
    pub config_file: PathBuf,
    pub config_error: Option<String>,
    pub data_dir: PathBuf,
    pub db: DbStatus,
    pub ffmpeg: Tool,
    pub ffprobe: Tool,
    pub gpu: Vec<Gpu>,
    /// Frame descriptions.
    pub vision: Resolution,
    /// Script chat (its own backend and model settings).
    pub chat: Resolution,
    /// The local context window / KV cache of both, for the report.
    pub local_runtime: LocalRuntimeInfo,
    pub embeddings: Resolution,
    pub stt: Resolution,
    pub models: Vec<ModelFile>,
    /// Installed coding-agent CLI tools (name → path) and which capability uses one.
    pub cli_tools: CliToolsInfo,
}

/// CLI tool availability and configuration.
#[derive(Debug, Clone, Serialize, Default)]
pub struct CliToolsInfo {
    /// Detected installations: (tool_name, path).
    pub installed: Vec<(String, PathBuf)>,
    /// Human-readable note about which capability is configured to use CLI (if any).
    pub active_for: Vec<String>,
}

impl Report {
    /// Problems that block indexing (empty = ready).\
    pub fn blockers(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(e) = &self.config_error {
            out.push(format!("config: {e}"));
        }
        if !self.db.ok {
            out.push(format!("database: {}", self.db.error.as_deref().unwrap_or("unavailable")));
        }
        for t in [&self.ffmpeg, &self.ffprobe] {
            if t.path.is_none() {
                out.push(format!("{} not found (bundled with installers; on dev boxes install it)", t.name));
            }
        }
        for (name, r) in [("vision", &self.vision), ("embeddings", &self.embeddings), ("stt", &self.stt)] {
            if r.target == Target::Unavailable {
                out.push(format!("{name}: {}", r.reason));
            }
        }
        out
    }
}

pub async fn run(paths: &Paths) -> Report {
    let (config, config_error) = match Config::load(&paths.config_file) {
        Ok(c) => (c, None),
        Err(e) => (Config::default(), Some(e.to_string())),
    };

    let db = db_status(&paths.db_file());
    let client = probe::probe_client();

    let vision_probe = async {
        match config.vision.backend {
            Backend::Local | Backend::Cli => None,
            _ => Some(probe::vision(&client, &config.vision.url, &config.vision.model).await),
        }
    };
    let embed_probe = async {
        match config.embed.backend {
            Backend::Local | Backend::Cli => None,
            _ => Some(probe::embeddings(&client, &config.embed.url, &config.embed.model).await),
        }
    };
    let stt_probe = async {
        match config.stt.backend {
            Backend::Local | Backend::Cli => None,
            _ => Some(probe::stt(&client, &config.stt.url).await),
        }
    };
    let (ffmpeg, ffprobe, gpu, vp, ep, sp) =
        tokio::join!(tool("ffmpeg"), tool("ffprobe"), nvidia_gpus(), vision_probe, embed_probe, stt_probe);

    let models_dir = crate::models::effective_models_dir(paths, &config);
    let mut search: Vec<PathBuf> = vec![models_dir];
    search.extend(config.models.search_paths.iter().cloned());

    let (vision_file, proj_file) = match crate::models::vision_pair(&config.vision.local_model) {
        Some((m, p)) => (m.file_name, p.file_name),
        None => ("Bonsai-27B-Q1_0.gguf".to_string(), "Bonsai-27B-mmproj-Q8_0.gguf".to_string()),
    };

    let model_files = vec![
        ("vision (local)".to_string(), vision_file),
        ("vision projector (local)".to_string(), proj_file),
        ("embeddings (local)".to_string(), "embeddinggemma-300M-Q8_0.gguf".to_string()),
        ("whisper (local, GPU)".to_string(), "ggml-large-v3-turbo.bin".to_string()),
        ("whisper (local, CPU)".to_string(), "ggml-small.bin".to_string()),
    ];

    let models = model_files
        .into_iter()
        .map(|(role, pattern)| {
            let found = find_file(&search, &pattern, 5);
            ModelFile { role, pattern, found }
        })
        .collect();

    // Detect installed CLI tools and which capability uses one.
    let cli_tools = {
        use crate::cliagent::find_binary;
        use crate::config::CLI_TOOLS;
        let mut installed = Vec::new();
        for &tool_name in CLI_TOOLS {
            let probe_cfg = crate::config::CliAgentConfig { tool: tool_name.to_string(), ..Default::default() };
            if let Some(path) = find_binary(&probe_cfg) {
                installed.push((tool_name.to_string(), path));
            }
        }
        let mut active_for = Vec::new();
        if config.vision.backend == Backend::Cli {
            active_for.push(format!("frame descriptions ({})", config.vision.cli.tool));
        }
        if config.chat_model().backend == Backend::Cli {
            active_for.push(format!("script chat ({})", config.chat_model().cli.tool));
        }
        CliToolsInfo { installed, active_for }
    };

    Report {
        version: env!("CARGO_PKG_VERSION"),
        config_file: paths.config_file.clone(),
        config_error,
        data_dir: paths.data_dir.clone(),
        db,
        ffmpeg,
        ffprobe,
        gpu,
        vision: probe::resolve(config.vision.backend, vp.clone()),
        chat: probe::resolve(config.chat_model().backend, vp),
        local_runtime: LocalRuntimeInfo {
            describe_ctx: config.vision.ctx_tokens,
            describe_kv: config.vision.kv_cache.clone(),
            chat_ctx: config.chat_model().ctx_tokens,
            chat_kv: config.chat_model().kv_cache.clone(),
        },
        embeddings: probe::resolve(config.embed.backend, ep),
        stt: probe::resolve(config.stt.backend, sp),
        models,
        cli_tools,
    }
}

/// Local helper settings shown by `doctor`.
#[derive(Debug, Clone, Serialize)]
pub struct LocalRuntimeInfo {
    pub describe_ctx: u32,
    pub describe_kv: String,
    pub chat_ctx: u32,
    pub chat_kv: String,
}

fn db_status(path: &Path) -> DbStatus {
    match Db::open(path) {
        Ok(db) => DbStatus {
            path: path.to_path_buf(),
            ok: true,
            schema_version: db.schema_version().ok(),
            sqlite_vec: db.vec_version().ok(),
            error: None,
        },
        Err(e) => DbStatus {
            path: path.to_path_buf(),
            ok: false,
            schema_version: None,
            sqlite_vec: None,
            error: Some(e.to_string()),
        },
    }
}

fn exe_name(name: &str) -> String {
    if cfg!(windows) { format!("{name}.exe") } else { name.to_string() }
}

/// Locate a helper binary: `GHOSTREEL_<NAME>` env, next to our executable (bundled sidecar), `PATH`.
pub fn locate(name: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(format!("GHOSTREEL_{}", name.to_uppercase())) {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let file = exe_name(name);
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        let p = dir.join(&file);
        if p.is_file() {
            return Some(p);
        }
    }
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(&file)).find(|p| p.is_file())
}

async fn output(program: &Path, args: &[&str]) -> Option<String> {
    let fut = crate::proc::command(program).args(args).kill_on_drop(true).output();
    let out = tokio::time::timeout(Duration::from_secs(5), fut).await.ok()?.ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

async fn tool(name: &str) -> Tool {
    let path = locate(name);
    let version = match &path {
        Some(p) => output(p, &["-version"]).await.and_then(|s| {
            // "ffmpeg version n8.0 Copyright ..." → "n8.0"
            s.lines().next().and_then(|l| l.split_whitespace().nth(2)).map(str::to_string)
        }),
        None => None,
    };
    Tool { name: name.into(), path, version }
}

pub async fn nvidia_gpus() -> Vec<Gpu> {
    let Some(smi) = locate("nvidia-smi") else { return Vec::new() };
    let Some(csv) =
        output(&smi, &["--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"])
            .await
    else {
        return Vec::new();
    };
    parse_nvidia_smi(&csv)
}

fn parse_nvidia_smi(csv: &str) -> Vec<Gpu> {
    csv.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            (f.len() == 4).then(|| Gpu {
                name: f[0].to_string(),
                vram_total_mib: f[1].parse().unwrap_or(0),
                vram_used_mib: f[2].parse().unwrap_or(0),
                driver: f[3].to_string(),
            })
        })
        .collect()
}

/// Calculate optimal describe concurrency based on GPU VRAM.
/// Returns 1 on CPU/no GPU or when VRAM is tight, up to 8 on high-VRAM cards.
pub fn optimal_describe_concurrency(total_vram_mib: Option<u64>, model_vram_mib: Option<u64>) -> usize {
    let Some(total) = total_vram_mib else {
        return 1;
    };
    if total < 4000 {
        return 1;
    }
    // Estimated model weight footprint (default ~4.5 GB for Bonsai-27B/Qwen2.5-VL-7B)
    // plus 1000 MiB base system/driver reserve.
    let reserve = model_vram_mib.unwrap_or(4500).saturating_add(1000);
    if total <= reserve {
        return 1;
    }
    let headroom = total - reserve;
    // With 768px images, each concurrent sequence requires ~600-800 MiB KV + activation headroom.
    let slots = (headroom / 800).clamp(1, 8);
    slots as usize
}

/// Calculate effective describe concurrency taking configuration and GPU VRAM into account.
pub fn calculate_describe_concurrency(
    configured: u32,
    total_vram_mib: Option<u64>,
    model_vram_mib: Option<u64>,
) -> usize {
    let optimal = optimal_describe_concurrency(total_vram_mib, model_vram_mib);
    if configured == 0 { optimal } else { (configured as usize).min(optimal).max(1) }
}

/// Breadth-limited search for `file_name` under `roots` (model dirs are shallow trees).
fn find_file(roots: &[PathBuf], file_name: &str, max_depth: usize) -> Option<PathBuf> {
    fn walk(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
        let entries = std::fs::read_dir(dir).ok()?;
        let mut subdirs = Vec::new();
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                subdirs.push(p);
            } else if e.file_name().to_string_lossy().eq_ignore_ascii_case(name) {
                return Some(p);
            }
        }
        if depth == 0 {
            return None;
        }
        subdirs.iter().find_map(|d| walk(d, name, depth - 1))
    }
    roots.iter().find_map(|r| walk(r, file_name, max_depth))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_csv() {
        let gpus = parse_nvidia_smi("NVIDIA GeForce RTX 4070, 12282, 9504, 580.82\n");
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].vram_total_mib, 12282);
        assert_eq!(gpus[0].driver, "580.82");
        assert!(parse_nvidia_smi("garbage").is_empty());
    }

    #[test]
    fn finds_nested_model_files() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("ggml-org/embeddinggemma-300M-GGUF");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("embeddinggemma-300M-Q8_0.gguf"), b"x").unwrap();
        let roots = vec![dir.path().join("missing"), dir.path().to_path_buf()];
        assert!(find_file(&roots, "embeddinggemma-300M-Q8_0.gguf", 3).is_some());
        assert!(find_file(&roots, "embeddinggemma-300M-Q8_0.gguf", 1).is_none());
    }

    #[tokio::test]
    async fn report_with_everything_local_has_no_server_blockers() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths { config_file: dir.path().join("config.toml"), data_dir: dir.path().join("data") };
        let mut cfg = Config::default();
        cfg.vision.backend = Backend::Local;
        cfg.embed.backend = Backend::Local;
        cfg.stt.backend = Backend::Local;
        cfg.save(&paths.config_file).unwrap();

        let r = run(&paths).await;
        assert!(r.db.ok, "{:?}", r.db.error);
        assert_eq!(r.vision.target, Target::Local);
        assert!(r.blockers().iter().all(|b| !b.starts_with("vision") && !b.starts_with("stt")));
    }

    #[test]
    fn test_optimal_describe_concurrency() {
        assert_eq!(optimal_describe_concurrency(None, None), 1);
        assert_eq!(optimal_describe_concurrency(Some(2048), None), 1);
        assert_eq!(optimal_describe_concurrency(Some(6000), Some(4500)), 1);
        assert_eq!(optimal_describe_concurrency(Some(8192), Some(4500)), 3);
        assert_eq!(optimal_describe_concurrency(Some(12282), Some(4500)), 8);
        assert_eq!(optimal_describe_concurrency(Some(24576), Some(4500)), 8);

        // calculate_describe_concurrency caps or defaults:
        assert_eq!(calculate_describe_concurrency(0, Some(12282), Some(4500)), 8);
        assert_eq!(calculate_describe_concurrency(4, Some(12282), Some(4500)), 4);
        assert_eq!(calculate_describe_concurrency(8, Some(6000), Some(4500)), 1);
        assert_eq!(calculate_describe_concurrency(4, None, None), 1);
    }
}
