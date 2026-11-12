//! Spawning helpers (ffmpeg, whisper, the model helpers, the OTIO sidecar) without flashing a
//! console window on Windows: a console subsystem program started from a GUI app gets its own
//! `conhost` window unless `CREATE_NO_WINDOW` is set. No effect on Linux or macOS.

use std::ffi::OsStr;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A [`std::process::Command`] that shows no console window.
pub fn std_command(program: impl AsRef<OsStr>) -> std::process::Command {
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// A [`tokio::process::Command`] that shows no console window.
pub fn command(program: impl AsRef<OsStr>) -> tokio::process::Command {
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut cmd = tokio::process::Command::new(program);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    // A child lives no longer than the task that owns it: when Stop drops the work (see
    // `cancel::race`), ffmpeg, helpers and CLI agents are killed with it instead of running on.
    cmd.kill_on_drop(true);
    cmd
}
