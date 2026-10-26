//! Run a child process with a hard timeout, killing the whole process tree on expiry.

use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::task::JoinHandle;

/// Per stream. Compile errors and test summaries sit at the end, so we keep the tail.
const MAX_STREAM_BYTES: usize = 32 * 1024;
/// How long to wait for the pipes to close after the process is gone.
const DRAIN_GRACE: Duration = Duration::from_secs(3);

pub struct Captured {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub stdout: String,
    pub stderr: String,
}

pub async fn run(mut cmd: Command, timeout: Duration) -> Result<Captured> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Own process group on Unix, so one signal can reach cargo *and* the test binary.
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);

    let mut child = tokio::process::Command::from(cmd)
        .kill_on_drop(true)
        .spawn()
        .context("could not start the process")?;
    let pid = child.id();
    let stdout = tokio::spawn(read_tail(child.stdout.take().context("no stdout pipe")?));
    let stderr = tokio::spawn(read_tail(child.stderr.take().context("no stderr pipe")?));

    let (status, timed_out) = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(status) => (Some(status?), false),
        Err(_) => {
            // `child.kill()` alone would only stop cargo and leave the test binary running.
            // On Windows cargo already ties its children to a job object, so this is belt and
            // braces there; on Unix the process-group kill is what stops the test binary.
            kill_tree(pid).await;
            let _ = child.start_kill();
            let _ = child.wait().await;
            (None, true)
        }
    };

    Ok(Captured {
        success: status.is_some_and(|s| s.success()),
        exit_code: status.and_then(|s| s.code()),
        timed_out,
        stdout: join_tail(stdout).await,
        stderr: join_tail(stderr).await,
    })
}

/// Read a stream to the end but only keep the last `MAX_STREAM_BYTES`.
async fn read_tail(mut stream: impl AsyncRead + Unpin) -> (Vec<u8>, bool) {
    let mut kept = Vec::new();
    let mut truncated = false;
    let mut chunk = [0u8; 8192];
    // A read error just ends the capture; whatever we have so far is still useful.
    while let Ok(n) = stream.read(&mut chunk).await {
        if n == 0 {
            break;
        }
        kept.extend_from_slice(&chunk[..n]);
        if kept.len() > MAX_STREAM_BYTES {
            kept.drain(..kept.len() - MAX_STREAM_BYTES);
            truncated = true;
        }
    }
    (kept, truncated)
}

async fn join_tail(handle: JoinHandle<(Vec<u8>, bool)>) -> String {
    let (bytes, truncated) = match tokio::time::timeout(DRAIN_GRACE, handle).await {
        Ok(Ok(result)) => result,
        // Pipe still held open by a stray process, or the task failed: give up on the rest.
        _ => (Vec::new(), false),
    };
    let text = String::from_utf8_lossy(&bytes);
    if truncated {
        format!("[... earlier output truncated ...]\n{text}")
    } else {
        text.into_owned()
    }
}

async fn kill_tree(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    #[cfg(windows)]
    let killer = tokio::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .output()
        .await;
    #[cfg(unix)]
    let killer = tokio::process::Command::new("kill")
        .args(["-KILL", "--", &format!("-{pid}")])
        .output()
        .await;
    // Best effort: the process may already be gone.
    let _ = killer;
}
