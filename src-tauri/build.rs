fn main() {
    fail_if_the_checkout_moved();
    tauri_build::build()
}

/// Stop with an explanation when `target/` was filled from a different location.
///
/// Cargo records absolute paths in build-script output and never notices that the checkout has
/// been renamed. tauri-build then reads its plugin permissions out of a directory that no longer
/// exists and fails with a path nobody recognises — the repository's old name — rather than with
/// the reason. This repo hit it twice after moving from HighVid/ to ghostreel/, and llama-cpp-sys
/// and whisper-rs-sys fail the same way.
///
/// The stamp is only a tripwire: a different path that still exists means a shared
/// `CARGO_TARGET_DIR`, which is fine, so only a path that has stopped existing is an error.
fn fail_if_the_checkout_moved() {
    use std::path::{Path, PathBuf};

    let Ok(out_dir) = std::env::var("OUT_DIR") else { return };
    let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") else { return };
    // OUT_DIR is <target>/<profile>/build/<crate>-<hash>/out.
    let Some(target_root) = Path::new(&out_dir).ancestors().nth(4).map(PathBuf::from) else { return };
    // src-tauri/ -> the repository.
    let Some(repo) = Path::new(&manifest).parent().map(PathBuf::from) else { return };

    let stamp = target_root.join(".built-from");
    println!("cargo:rerun-if-changed={}", stamp.display());

    if let Ok(previous) = std::fs::read_to_string(&stamp) {
        let previous = previous.trim();
        if !previous.is_empty() && Path::new(previous) != repo && !Path::new(previous).exists() {
            panic!(
                "\n\n  target/ was built from {previous}, which no longer exists.\n  \
                 Cargo caches absolute paths in build scripts, so this build would read files \
                 from there and fail with a confusing error.\n\n  \
                 Run:  scripts/clean-stale-target.sh\n\n"
            );
        }
    }
    let _ = std::fs::write(&stamp, repo.to_string_lossy().as_bytes());
}
