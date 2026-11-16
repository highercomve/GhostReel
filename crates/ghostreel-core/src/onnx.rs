//! The one ONNX Runtime environment of the process. ONNX Runtime allows a single environment per
//! process, so every model runner (faces, speakers, and whatever comes next) takes this shared one
//! instead of building its own: a run that does faces and then speakers in one process works.

use std::sync::{Arc, Mutex};

use ort::environment::Environment;

use crate::Error;

static ENV: Mutex<Option<Arc<Environment>>> = Mutex::new(None);

/// The shared environment, created on first use.
pub fn environment() -> Result<Arc<Environment>, Error> {
    let mut slot = ENV.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(env) = slot.as_ref() {
        return Ok(env.clone());
    }
    let env = Arc::new(builder()?.build().map_err(|e| Error::Invalid(format!("Failed to init ort environment: {e}")))?);
    *slot = Some(env.clone());
    Ok(env)
}

/// Every target but Intel macOS links ONNX Runtime into the binary.
#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
fn builder() -> Result<ort::environment::EnvironmentBuilder, Error> {
    Ok(ort::init())
}

/// Intel macOS loads the onnxruntime dylib shipped next to the program (see Cargo.toml). Without
/// it the stages that need ONNX Runtime report that they are unavailable instead of panicking
/// inside ort, which would otherwise look for the dylib on the system search path.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn builder() -> Result<ort::environment::EnvironmentBuilder, Error> {
    let Some(path) = dylib::find() else {
        return Err(Error::Invalid(format!(
            "face and speaker matching unavailable: {} not found next to the program (looked in {})",
            dylib::NAME,
            dylib::candidates().iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        )));
    };
    ort::init_from(&path).map_err(|e| {
        Error::Invalid(format!("face and speaker matching unavailable: cannot load {}: {e}", path.display()))
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
mod dylib {
    use std::path::PathBuf;

    pub const NAME: &str = "libonnxruntime.dylib";

    /// Where the dylib may be, in order: `ORT_DYLIB_PATH` (development and tests), the app bundle's
    /// Frameworks folder (Contents/MacOS/<exe> -> Contents/Frameworks), next to the executable (the
    /// CLI in target/release or an unpacked CLI folder), and its lib/ folder.
    pub fn candidates() -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Some(p) = std::env::var_os("ORT_DYLIB_PATH").filter(|p| !p.is_empty()) {
            out.push(PathBuf::from(p));
        }
        if let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(PathBuf::from)) {
            out.push(dir.join("../Frameworks").join(NAME));
            out.push(dir.join(NAME));
            out.push(dir.join("lib").join(NAME));
        }
        out
    }

    pub fn find() -> Option<PathBuf> {
        candidates().into_iter().find(|p| p.is_file())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_caller_gets_the_same_environment() {
        let a = super::environment().unwrap();
        let b = super::environment().unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b));
    }
}
