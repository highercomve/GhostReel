//! Stopping long work: one flag, handed from whoever owns the work (the app's queue, a CLI
//! Ctrl+C) down into every loop that can take a while.
//!
//! The flag is the `Arc<AtomicBool>` every producer already takes (`index::Options::cancel`,
//! `PreviewOptions::cancel`, `clips::run`…). This module is what they do with it:
//!
//! - a loop over videos, frames or windows checks [`is_set`] once per item;
//! - an `await` that can last (a model call, an ffmpeg decode, a slot on a busy endpoint) goes
//!   through [`race`], which drops the future as soon as the flag is set. Child processes are
//!   spawned with `kill_on_drop(true)`, so dropping the future is what kills them;
//! - blocking code (an ONNX loop on a thread) polls [`is_set`] per frame and kills its own child.
//!
//! Stopped work ends with [`crate::Error::Cancelled`].

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// The shared stop flag.
pub type CancelFlag = Arc<AtomicBool>;

/// How often [`wait`] looks at the flag: well inside the 1–2 s a Stop should take to show.
pub const POLL: Duration = Duration::from_millis(100);

/// Whether stop was asked for (`None`: the work cannot be stopped).
pub fn is_set(flag: Option<&CancelFlag>) -> bool {
    flag.is_some_and(|f| f.load(Ordering::SeqCst))
}

/// Resolves once the flag is set (never, for `None`).
pub async fn wait(flag: Option<&CancelFlag>) {
    let Some(flag) = flag else { return std::future::pending().await };
    while !flag.load(Ordering::SeqCst) {
        tokio::time::sleep(POLL).await;
    }
}

/// Run `fut` unless stop is asked for first: `Err(Cancelled)` then, and `fut` is dropped — with
/// any `kill_on_drop` child process it was waiting on.
pub async fn race<F: Future>(flag: Option<&CancelFlag>, fut: F) -> Result<F::Output, crate::Error> {
    if is_set(flag) {
        return Err(crate::Error::Cancelled);
    }
    tokio::select! {
        biased;
        _ = wait(flag) => Err(crate::Error::Cancelled),
        out = fut => Ok(out),
    }
}

/// Blocking `Command::output` that kills the child as soon as stop is asked for. A stop is an
/// `Interrupted` error ([`is_cancel_io`] tells it apart). For blocking producers (a render on a
/// thread) whose ffmpeg can run for minutes per call.
pub fn output(cmd: &mut std::process::Command, flag: Option<&CancelFlag>) -> std::io::Result<std::process::Output> {
    use std::io::Read;
    use std::process::Stdio;
    let Some(flag) = flag else { return cmd.output() };
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut p) = pipe {
                let _ = p.read_to_end(&mut buf);
            }
            buf
        })
    };
    let out = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let err = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if flag.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(std::io::ErrorKind::Interrupted, CANCELLED_IO));
        }
        std::thread::sleep(POLL);
    };
    Ok(std::process::Output { status, stdout: out.join().unwrap_or_default(), stderr: err.join().unwrap_or_default() })
}

const CANCELLED_IO: &str = "cancelled";

/// Whether an I/O error is [`output`]'s stop.
pub fn is_cancel_io(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::Interrupted && e.to_string() == CANCELLED_IO
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn race_returns_the_result_when_not_stopped() {
        let flag = Arc::new(AtomicBool::new(false));
        assert_eq!(race(Some(&flag), async { 7 }).await.unwrap(), 7);
        assert_eq!(race(None, async { 8 }).await.unwrap(), 8);
    }

    #[tokio::test]
    async fn race_stops_a_long_future_within_a_poll() {
        let flag = Arc::new(AtomicBool::new(false));
        let f2 = flag.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            f2.store(true, Ordering::SeqCst);
        });
        let t = Instant::now();
        let r = race(Some(&flag), tokio::time::sleep(Duration::from_secs(30))).await;
        assert!(matches!(r, Err(crate::Error::Cancelled)));
        assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
    }

    /// Dropping the raced future kills the child process it was waiting on.
    #[cfg(unix)]
    #[tokio::test]
    async fn race_kills_the_child_process() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut child = crate::proc::command("sleep").arg("30").kill_on_drop(true).spawn().unwrap();
        let pid = child.id().unwrap();
        let f2 = flag.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            f2.store(true, Ordering::SeqCst);
        });
        let r = race(Some(&flag), async move { child.wait().await }).await;
        assert!(r.is_err());
        // The child is reaped asynchronously after the kill; give it a moment.
        let gone = (0..50).any(|_| {
            std::thread::sleep(Duration::from_millis(20));
            !std::path::Path::new(&format!("/proc/{pid}")).exists()
                || std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|s| s.contains(") Z"))
        });
        assert!(gone, "sleep {pid} still running after cancel");
    }

    #[cfg(unix)]
    #[test]
    fn blocking_output_kills_the_child_on_stop() {
        let flag = Arc::new(AtomicBool::new(false));
        let f2 = flag.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            f2.store(true, Ordering::SeqCst);
        });
        let t = Instant::now();
        let r = output(std::process::Command::new("sleep").arg("30"), Some(&flag));
        assert!(r.as_ref().is_err_and(is_cancel_io), "{r:?}");
        assert!(t.elapsed() < Duration::from_secs(2));
        let ok = output(std::process::Command::new("echo").arg("hi"), Some(&Arc::new(AtomicBool::new(false)))).unwrap();
        assert_eq!(String::from_utf8_lossy(&ok.stdout).trim(), "hi");
    }

    #[test]
    fn cancelled_is_recognised_whatever_layer_raised_it() {
        assert!(crate::Error::Cancelled.is_cancelled());
        assert!(crate::Error::Preview("cancelled".into()).is_cancelled());
        assert!(!crate::Error::Invalid("boom".into()).is_cancelled());
    }
}
