//! Where GhostReel keeps its files: one folder per user, `~/.ghostreel/`
//! (`%USERPROFILE%\.ghostreel\` on Windows), holding
//! `config.toml`, `ghostreel.db`, `frames/`, `proxies/`, `previews/` and `models/`.
//!
//! `GHOSTREEL_HOME` moves the whole folder; `GHOSTREEL_CONFIG` / `GHOSTREEL_DATA` override the config
//! file / data folder individually (tests, portable installs). Data from older builds
//! (`~/.local/share/ghostreel`, `~/.config/ghostreel/config.toml`) is moved in on first start.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::Error;

#[derive(Debug, Clone, Serialize)]
pub struct Paths {
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
}

/// `~/.ghostreel` (or `GHOSTREEL_HOME`).
pub fn home_dir() -> Result<PathBuf, Error> {
    if let Some(p) = std::env::var_os("GHOSTREEL_HOME") {
        return Ok(PathBuf::from(p));
    }
    Ok(dirs::home_dir().ok_or(Error::NoHomeDir("home"))?.join(".ghostreel"))
}

impl Paths {
    pub fn resolve() -> Result<Self, Error> {
        let env_config = std::env::var_os("GHOSTREEL_CONFIG").map(PathBuf::from);
        let env_data = std::env::var_os("GHOSTREEL_DATA").map(PathBuf::from);
        let home = match (&env_config, &env_data) {
            (Some(_), Some(_)) => None,
            _ => Some(home_dir()?),
        };
        if env_config.is_none()
            && env_data.is_none()
            && std::env::var_os("GHOSTREEL_HOME").is_none()
            && let Some(h) = &home
        {
            migrate_legacy(h);
        }
        let config_file = env_config.unwrap_or_else(|| home.clone().expect("home resolved").join("config.toml"));
        let data_dir = env_data.unwrap_or_else(|| home.expect("home resolved"));
        Ok(Self { config_file, data_dir })
    }

    /// A self-contained layout under `dir` (the config file beside the data): for tests and
    /// portable installs.
    pub fn in_dir(dir: &Path) -> Self {
        Self { config_file: dir.join("config.toml"), data_dir: dir.to_path_buf() }
    }

    pub fn db_file(&self) -> PathBuf {
        self.data_dir.join("ghostreel.db")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    pub fn thumbs_dir(&self) -> PathBuf {
        self.data_dir.join("thumbs")
    }
}

/// Move data and config from the pre-`~/.ghostreel` locations, once. Best effort: a failed move
/// leaves the old files in place (nothing is deleted).
fn migrate_legacy(home: &Path) {
    let old_data = dirs::data_local_dir().map(|d| d.join("ghostreel"));
    let old_config = dirs::config_dir().map(|d| d.join("ghostreel").join("config.toml"));
    migrate_from(home, old_data.as_deref(), old_config.as_deref());
}

fn migrate_from(home: &Path, old_data: Option<&Path>, old_config: Option<&Path>) {
    if let Some(old) = old_data.filter(|p| p.is_dir() && *p != home) {
        if !home.exists() {
            if let Some(parent) = home.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            // Same filesystem: instant rename, even for GBs of models.
            if std::fs::rename(old, home).is_err() {
                let _ = copy_dir(old, home);
            }
        } else {
            // Home already exists: bring over entries it doesn't have yet.
            if let Ok(entries) = std::fs::read_dir(old) {
                for e in entries.flatten() {
                    let target = home.join(e.file_name());
                    if !target.exists() && std::fs::rename(e.path(), &target).is_err() {
                        let _ = if e.path().is_dir() {
                            copy_dir(&e.path(), &target)
                        } else {
                            std::fs::copy(e.path(), &target).map(|_| ())
                        };
                    }
                }
            }
        }
    }
    if let Some(old) = old_config.filter(|p| p.is_file()) {
        let target = home.join("config.toml");
        if !target.exists() {
            let _ = std::fs::create_dir_all(home);
            if std::fs::rename(old, &target).is_err() {
                let _ = std::fs::copy(old, &target);
            }
        }
    }
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let target = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_dir(&e.path(), &target)?;
        } else {
            std::fs::copy(e.path(), &target)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_old_data_and_config_into_home() {
        let tmp = tempfile::tempdir().unwrap();
        let old_data = tmp.path().join("share/ghostreel");
        std::fs::create_dir_all(old_data.join("models")).unwrap();
        std::fs::write(old_data.join("ghostreel.db"), b"db").unwrap();
        std::fs::write(old_data.join("models/ggml-tiny.bin"), b"m").unwrap();
        let old_config = tmp.path().join("config/ghostreel/config.toml");
        std::fs::create_dir_all(old_config.parent().unwrap()).unwrap();
        std::fs::write(&old_config, b"[stt]\n").unwrap();

        let home = tmp.path().join("home/.ghostreel");
        migrate_from(&home, Some(&old_data), Some(&old_config));
        assert_eq!(std::fs::read(home.join("ghostreel.db")).unwrap(), b"db");
        assert!(home.join("models/ggml-tiny.bin").is_file());
        assert_eq!(std::fs::read(home.join("config.toml")).unwrap(), b"[stt]\n");
        assert!(!old_data.exists(), "moved, not copied");

        // Running again is a no-op.
        migrate_from(&home, Some(&old_data), Some(&old_config));
        assert!(home.join("ghostreel.db").is_file());
    }

    #[test]
    fn merges_into_existing_home_without_overwriting() {
        let tmp = tempfile::tempdir().unwrap();
        let old_data = tmp.path().join("old");
        std::fs::create_dir_all(old_data.join("models")).unwrap();
        std::fs::write(old_data.join("ghostreel.db"), b"old").unwrap();
        std::fs::write(old_data.join("models/a.bin"), b"a").unwrap();
        let home = tmp.path().join(".ghostreel");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(home.join("ghostreel.db"), b"new").unwrap();

        migrate_from(&home, Some(&old_data), None);
        assert_eq!(std::fs::read(home.join("ghostreel.db")).unwrap(), b"new", "existing file kept");
        assert!(home.join("models/a.bin").is_file(), "missing folder brought over");
    }
}
