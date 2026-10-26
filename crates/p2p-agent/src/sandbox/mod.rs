//! A throwaway copy of the user's project where the agent reads, edits and tests.
//!
//! The user's working tree is never written to. The only thing that leaves the sandbox
//! is a unified diff ([`Sandbox::diff`]) that a human can review and apply.
//!
//! [`LocalSandbox`] (a temp-directory copy) is the only backend today; the trait leaves
//! room for a stronger one (container, Modal) without touching the agent loop.

mod exec;
mod local;

use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::{Result, bail};

pub use local::{LocalSandbox, check_patch_applies};

/// How a `cargo test` run ended. The distinction matters for the red/green check:
/// a regression test must fail with `TestsFailed`, not because the code doesn't compile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestOutcome {
    Passed,
    TestsFailed,
    BuildFailed,
    TimedOut,
}

#[derive(Debug, Clone)]
pub struct TestRun {
    pub outcome: TestOutcome,
    pub exit_code: Option<i32>,
    /// stdout and stderr (tail-capped, paths and secrets scrubbed). Safe to show a model.
    pub output: String,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct TestOptions {
    /// Only run tests whose name contains this text (passed to `cargo test`).
    pub filter: Option<String>,
    pub timeout: Duration,
}

impl Default for TestOptions {
    fn default() -> Self {
        Self {
            filter: None,
            timeout: Duration::from_secs(120),
        }
    }
}

/// Paths are always relative to the project root and use `/` (also on Windows).
pub trait Sandbox {
    fn read_file(&self, path: &str) -> Result<String>;
    /// Create or overwrite a file (parent folders are created).
    fn write_file(&self, path: &str, content: &str) -> Result<()>;
    /// Every file under `dir` (`""` means the project root), sorted.
    fn list_files(&self, dir: &str) -> Result<Vec<String>>;
    /// Unified diff of everything changed in the sandbox, ready for `git apply`.
    fn diff(&self) -> Result<String>;
    fn run_tests(&self, opts: &TestOptions) -> impl Future<Output = Result<TestRun>> + Send;

    /// Replace `old` with `new`; `old` must match exactly once (see [`apply_edit`]).
    fn edit_file(&self, path: &str, old: &str, new: &str) -> Result<()> {
        let content = self.read_file(path)?;
        self.write_file(path, &apply_edit(&content, old, new)?)
    }
}

/// Exact search/replace. Errors say how to fix the call, because the model reads them.
pub fn apply_edit(content: &str, old: &str, new: &str) -> Result<String> {
    if old.is_empty() {
        bail!("`old` must not be empty");
    }
    // Files checked out on Windows may use CRLF while the model writes plain `\n`.
    let crlf = content.contains("\r\n") && !old.contains('\r');
    let (old, new) = if crlf {
        (old.replace('\n', "\r\n"), new.replace('\n', "\r\n"))
    } else {
        (old.to_string(), new.to_string())
    };
    match content.matches(old.as_str()).count() {
        0 => bail!("`old` text not found; it must match the file exactly, including whitespace"),
        1 => Ok(content.replacen(&old, &new, 1)),
        n => bail!(
            "`old` text matches {n} places; include more surrounding lines so it matches once"
        ),
    }
}

/// Turn a model-supplied relative path into a real path inside `root`.
///
/// `root` must already be canonical. Rejects absolute paths, drive prefixes and `..`,
/// and also catches symlinks that would lead outside the root.
pub(crate) fn resolve(root: &Path, rel: &str) -> Result<PathBuf> {
    let rel_path = Path::new(rel);
    for component in rel_path.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            _ => bail!("path `{rel}` must be relative and stay inside the project"),
        }
    }
    let full = root.join(rel_path);
    // The file may not exist yet, so check the deepest ancestor that does.
    let mut existing = full.as_path();
    while !existing.exists() {
        match existing.parent() {
            Some(parent) => existing = parent,
            None => bail!("path `{rel}` is not inside the project"),
        }
    }
    if !dunce::canonicalize(existing)?.starts_with(root) {
        bail!("path `{rel}` leads outside the project");
    }
    Ok(full)
}

/// `cargo test` exits with 101 both for failing tests and for compile errors,
/// so the harness summary line is what tells them apart.
pub(crate) fn classify(success: bool, timed_out: bool, stdout: &str) -> TestOutcome {
    if timed_out {
        TestOutcome::TimedOut
    } else if success {
        TestOutcome::Passed
    } else if stdout.contains("test result: FAILED") {
        TestOutcome::TestsFailed
    } else {
        TestOutcome::BuildFailed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_replaces_a_unique_match() {
        let out = apply_edit("let x = a - b;\nlet y = 1;\n", "a - b", "a + b").unwrap();
        assert_eq!(out, "let x = a + b;\nlet y = 1;\n");
    }

    #[test]
    fn edit_rejects_missing_ambiguous_and_empty() {
        assert!(
            apply_edit("abc", "zzz", "x")
                .unwrap_err()
                .to_string()
                .contains("not found")
        );
        assert!(
            apply_edit("aa", "a", "b")
                .unwrap_err()
                .to_string()
                .contains("2 places")
        );
        assert!(apply_edit("abc", "", "b").is_err());
    }

    #[test]
    fn edit_handles_crlf_files() {
        let out = apply_edit(
            "fn f() {\r\n    a - b\r\n}\r\n",
            "{\n    a - b\n}",
            "{\n    a + b\n}",
        )
        .unwrap();
        assert_eq!(out, "fn f() {\r\n    a + b\r\n}\r\n");
    }

    #[test]
    fn resolve_accepts_nested_relative_paths() {
        let dir = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(dir.path()).unwrap();
        assert_eq!(
            resolve(&root, "src/new/mod.rs").unwrap(),
            root.join("src/new/mod.rs")
        );
        assert!(resolve(&root, "").is_ok());
        assert!(resolve(&root, "./src").is_ok());
    }

    #[test]
    fn resolve_rejects_escapes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(dir.path()).unwrap();
        let mut bad = vec!["../x", "src/../../x", "/etc/passwd"];
        // On Unix a backslash is an ordinary filename character, so these only escape on Windows.
        if cfg!(windows) {
            bad.extend([r"C:\Windows\win.ini", "C:evil", r"\\server\share\x"]);
        }
        for path in bad {
            assert!(resolve(&root, path).is_err(), "`{path}` should be rejected");
        }
    }

    #[test]
    fn classify_separates_test_failures_from_build_failures() {
        assert_eq!(classify(true, false, ""), TestOutcome::Passed);
        assert_eq!(classify(false, true, ""), TestOutcome::TimedOut);
        assert_eq!(
            classify(false, false, "test result: FAILED. 0 passed; 1 failed"),
            TestOutcome::TestsFailed
        );
        assert_eq!(classify(false, false, ""), TestOutcome::BuildFailed);
    }
}
