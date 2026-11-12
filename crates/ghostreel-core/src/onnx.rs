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
    let env = Arc::new(ort::init().build().map_err(|e| Error::Invalid(format!("Failed to init ort environment: {e}")))?);
    *slot = Some(env.clone());
    Ok(env)
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
