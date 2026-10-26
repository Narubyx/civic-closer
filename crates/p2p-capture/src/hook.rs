use std::backtrace::Backtrace;
use std::collections::BTreeMap;
use std::panic::{self, PanicHookInfo};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::incident::{Incident, Location};

/// Context attached to every incident (app version, build, scenario...).
static CONTEXT: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

/// Attach a key/value pair to all incidents captured from now on.
/// Values are redacted when the incident is built, so it is fine to pass raw data.
pub fn set_context(key: &str, value: &str) {
    // A poisoned lock only means another thread panicked while holding it; the data is still usable.
    let mut ctx = CONTEXT.lock().unwrap_or_else(|e| e.into_inner());
    ctx.insert(key.into(), value.into());
}

/// Install a panic hook that saves each panic as `<dir>/<timestamp>-<fingerprint>.json`.
///
/// The previous hook still runs afterwards, so the normal panic message is still printed.
pub fn install(dir: impl Into<PathBuf>) {
    let dir = dir.into();
    let previous = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        // Never panic inside a panic hook (that aborts the process): just report failures.
        match capture(info).save(&dir) {
            Ok(path) => eprintln!("[p2p] incident saved: {}", path.display()),
            Err(err) => eprintln!("[p2p] could not save incident: {err}"),
        }
        previous(info);
    }));
}

fn capture(info: &PanicHookInfo<'_>) -> Incident {
    let message = info
        .payload_as_str()
        .unwrap_or("<non-string panic payload>");
    let location = info.location().map(|l| Location {
        file: l.file().to_string(),
        line: l.line(),
        column: l.column(),
    });
    let thread = std::thread::current();
    // `force_capture` ignores RUST_BACKTRACE, so we always get a backtrace.
    let backtrace = Backtrace::force_capture().to_string();
    // `try_lock` so a panic raised while the lock is held can never deadlock the hook.
    let context = CONTEXT.try_lock().map(|c| c.clone()).unwrap_or_default();
    Incident::new(
        message,
        location,
        thread.name().unwrap_or("<unnamed>"),
        &backtrace,
        &context,
    )
}
