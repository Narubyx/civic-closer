use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::ffi::OsStr;
use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ignore::WalkBuilder;
use p2p_capture::{redact, short_hash};
use similar::TextDiff;
use tempfile::TempDir;

use super::{Sandbox, TestOptions, TestRun, classify, exec, resolve};

/// Environment variables the child is allowed to inherit: enough for cargo, rustup and the
/// MSVC linker to work, and nothing else. In particular `MODEL_API_KEY` is NOT here, so
/// AI-written test code cannot read it.
const ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "USERNAME",
    "LANG",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN",
];

/// A temp-directory copy of the project. The copy is deleted when this value is dropped.
pub struct LocalSandbox {
    /// Held only for its `Drop`, which deletes the directory.
    _tmp: TempDir,
    root: PathBuf,
    /// Shared across sandboxes of the same project so dependencies are not rebuilt every run.
    target_dir: PathBuf,
    /// Content of each touched file *before* its first write (`None` = file did not exist).
    originals: Mutex<BTreeMap<String, Option<String>>>,
}

impl LocalSandbox {
    /// Copy `project` (a folder with a `Cargo.toml`) into a fresh temp directory.
    pub fn create(project: &Path) -> Result<Self> {
        let project = dunce::canonicalize(project)
            .with_context(|| format!("project folder not found: {}", project.display()))?;
        if !project.join("Cargo.toml").is_file() {
            bail!("no Cargo.toml in {}", project.display());
        }
        let tmp = tempfile::Builder::new().prefix("p2p-sandbox-").tempdir()?;
        // Canonical (and not `\\?\`-prefixed) so path-prefix checks and cargo both behave.
        let root = dunce::canonicalize(tmp.path())?;
        copy_project(&project, &root)?;
        let target_dir = std::env::temp_dir()
            .join("p2p-target-cache")
            .join(short_hash(&project.to_string_lossy()));
        Ok(Self {
            _tmp: tmp,
            root,
            target_dir,
            originals: Mutex::default(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The `cargo test` command, with a scrubbed environment.
    fn cargo_test_command(&self, filter: Option<&str>) -> Result<Command> {
        let mut cmd = Command::new("cargo");
        cmd.args(["test", "--color", "never"]);
        // Libraries often don't commit Cargo.lock; `--locked` needs one to exist.
        if self.root.join("Cargo.lock").is_file() {
            cmd.arg("--locked");
        }
        if let Some(filter) = filter.filter(|f| !f.is_empty()) {
            // The filter comes from a model: never let it become a cargo flag.
            if filter.starts_with('-') {
                bail!("test filter must not start with `-`");
            }
            cmd.arg(filter);
        }
        cmd.current_dir(&self.root).env_clear();
        for key in ENV_ALLOWLIST {
            if let Some(value) = std::env::var_os(key) {
                cmd.env(key, value);
            }
        }
        cmd.env("CARGO_TARGET_DIR", &self.target_dir);
        Ok(cmd)
    }
}

impl Sandbox for LocalSandbox {
    fn read_file(&self, path: &str) -> Result<String> {
        let full = resolve(&self.root, path)?;
        fs::read_to_string(&full).with_context(|| format!("cannot read `{path}` as text"))
    }

    fn write_file(&self, path: &str, content: &str) -> Result<()> {
        let full = resolve(&self.root, path)?;
        if rel_key(&self.root, &full).is_empty() {
            bail!("`{path}` is the project folder, not a file");
        }
        self.remember_original(&full)?;
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&full, content).with_context(|| format!("cannot write `{path}`"))
    }

    fn list_files(&self, dir: &str) -> Result<Vec<String>> {
        let start = resolve(&self.root, dir)?;
        let mut files = Vec::new();
        collect_files(&self.root, &start, &mut files)?;
        files.sort();
        Ok(files)
    }

    fn diff(&self) -> Result<String> {
        let originals = self.originals.lock().unwrap_or_else(|e| e.into_inner());
        let mut patch = String::new();
        for (rel, original) in originals.iter() {
            let current = fs::read_to_string(self.root.join(rel))?;
            if original.as_deref() != Some(current.as_str()) {
                patch.push_str(&file_diff(rel, original.as_deref(), &current));
            }
        }
        Ok(patch)
    }

    async fn run_tests(&self, opts: &TestOptions) -> Result<TestRun> {
        let cmd = self.cargo_test_command(opts.filter.as_deref())?;
        let started = Instant::now();
        let run = exec::run(cmd, opts.timeout).await?;

        let outcome = classify(run.success, run.timed_out, &run.stdout);
        let text = format!(
            "--- stdout ---\n{}\n--- stderr ---\n{}",
            run.stdout, run.stderr
        );
        // Show the model project-relative paths, then scrub anything secret-looking.
        let text = text.replace(self.root.to_string_lossy().as_ref(), "<project>");
        Ok(TestRun {
            outcome,
            exit_code: run.exit_code,
            output: redact(&text),
            duration: started.elapsed(),
        })
    }
}

impl LocalSandbox {
    /// Snapshot a file the first time it is about to change, so `diff` has a baseline.
    fn remember_original(&self, full: &Path) -> Result<()> {
        let rel = rel_key(&self.root, full);
        let mut originals = self.originals.lock().unwrap_or_else(|e| e.into_inner());
        // Only the first write to a file records a baseline; later writes keep it.
        if let Entry::Vacant(slot) = originals.entry(rel) {
            let original = match fs::read_to_string(full) {
                Ok(text) => Some(text),
                Err(e) if e.kind() == ErrorKind::NotFound => None,
                Err(e) => {
                    let rel = slot.key();
                    return Err(e)
                        .with_context(|| format!("cannot edit `{rel}` (not UTF-8 text?)"));
                }
            };
            slot.insert(original);
        }
        Ok(())
    }
}

/// Check that `patch` applies cleanly to the real project, without changing anything
/// (`git apply --check`). Needs `git` on the PATH.
pub fn check_patch_applies(project: &Path, patch: &str) -> Result<()> {
    let mut child = Command::new("git")
        .args(["apply", "--check", "-"])
        .current_dir(project)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("could not run `git`")?;
    // Dropping the handle after the write closes stdin, which tells git the patch is complete.
    child
        .stdin
        .take()
        .context("no stdin pipe")?
        .write_all(patch.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!(
            "patch does not apply cleanly: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

/// What never gets copied into the sandbox: secrets, build output, VCS data, our own output.
fn is_excluded(name: &OsStr, depth: usize) -> bool {
    let name = name.to_string_lossy();
    match name.as_ref() {
        ".git" => true,
        // Only at the project root, so a module folder called `target/` is not lost.
        "target" | "incidents" | ".p2p" => depth == 1,
        n => n.starts_with(".env") || n.ends_with(".key") || n.ends_with(".pem"),
    }
}

fn copy_project(src: &Path, dst: &Path) -> Result<()> {
    let walker = WalkBuilder::new(src)
        .hidden(false) // keep dotfiles such as .cargo/config.toml
        .require_git(false) // honour .gitignore even when the folder is not a git repo
        .git_global(false)
        .filter_entry(|e| e.depth() == 0 || !is_excluded(e.file_name(), e.depth()))
        .build();
    for entry in walker {
        let entry = entry?;
        let rel = entry.path().strip_prefix(src)?;
        if rel.as_os_str().is_empty() {
            continue;
        }
        let target = dst.join(rel);
        match entry.file_type() {
            Some(t) if t.is_dir() => fs::create_dir_all(&target)?,
            Some(t) if t.is_file() => {
                fs::copy(entry.path(), &target)?;
            }
            // Symlinks (and anything else) are skipped on purpose: they could point outside.
            _ => {}
        }
    }
    Ok(())
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    const MAX_FILES: usize = 1000;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files(root, &entry.path(), out)?;
        } else if file_type.is_file() && out.len() < MAX_FILES {
            out.push(rel_key(root, &entry.path()));
        }
    }
    Ok(())
}

/// `src\lib.rs` -> `src/lib.rs`, relative to the sandbox root.
fn rel_key(root: &Path, full: &Path) -> String {
    let rel = full.strip_prefix(root).unwrap_or(full);
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// One file's section of a git-style unified diff.
fn file_diff(rel: &str, old: Option<&str>, new: &str) -> String {
    let mut out = format!("diff --git a/{rel} b/{rel}\n");
    let old_name = match old {
        Some(_) => format!("a/{rel}"),
        None => {
            out.push_str("new file mode 100644\n");
            "/dev/null".to_string()
        }
    };
    let diff = TextDiff::from_lines(old.unwrap_or(""), new);
    out.push_str(
        &diff
            .unified_diff()
            .context_radius(3)
            .header(&old_name, &format!("b/{rel}"))
            .to_string(),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A project containing both things that must be copied and things that must not.
    fn project() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let files = [
            (
                "Cargo.toml",
                "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            ("src/lib.rs", "pub fn f() -> i32 {\n    1\n}\n"),
            (
                "src/target/mod.rs",
                "// a module that happens to be called target\n",
            ),
            (".cargo/config.toml", "# kept\n"),
            (".gitignore", "ignored.txt\n"),
            ("ignored.txt", "skip me\n"),
            (".env", "MODEL_API_KEY=should-not-be-copied\n"),
            (".env.local", "X=1\n"),
            ("deploy.pem", "private\n"),
            ("target/debug/big.bin", "build output\n"),
            ("incidents/1.json", "{}\n"),
            (".git/config", "[core]\n"),
        ];
        for (path, content) in files {
            let full = dir.path().join(path);
            fs::create_dir_all(full.parent().unwrap()).unwrap();
            fs::write(full, content).unwrap();
        }
        dir
    }

    #[test]
    fn copy_keeps_sources_and_drops_secrets_and_build_output() {
        let project = project();
        let sandbox = LocalSandbox::create(project.path()).unwrap();
        let files = sandbox.list_files("").unwrap();
        assert_eq!(
            files,
            [
                ".cargo/config.toml",
                ".gitignore",
                "Cargo.toml",
                "src/lib.rs",
                "src/target/mod.rs"
            ]
        );
    }

    #[test]
    fn dropping_the_sandbox_deletes_the_copy() {
        let project = project();
        let sandbox = LocalSandbox::create(project.path()).unwrap();
        let root = sandbox.root().to_path_buf();
        assert!(root.exists());
        drop(sandbox);
        assert!(!root.exists());
    }

    #[test]
    fn edits_stay_in_the_copy_and_show_up_in_the_diff() {
        let project = project();
        let sandbox = LocalSandbox::create(project.path()).unwrap();
        sandbox
            .edit_file("src/lib.rs", "    1\n", "    2\n")
            .unwrap();
        sandbox
            .write_file("tests/new.rs", "#[test]\nfn t() {}\n")
            .unwrap();

        // The user's real file is untouched.
        assert_eq!(
            fs::read_to_string(project.path().join("src/lib.rs")).unwrap(),
            "pub fn f() -> i32 {\n    1\n}\n"
        );
        assert!(!project.path().join("tests").exists());

        let diff = sandbox.diff().unwrap();
        assert!(
            diff.contains(
                "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n"
            ),
            "{diff}"
        );
        assert!(diff.contains("-    1\n+    2\n"), "{diff}");
        assert!(
            diff.contains("new file mode 100644\n--- /dev/null\n+++ b/tests/new.rs\n"),
            "{diff}"
        );
        assert!(diff.contains("+#[test]\n"), "{diff}");
    }

    #[test]
    fn rewriting_a_file_back_to_the_original_leaves_no_diff() {
        let project = project();
        let sandbox = LocalSandbox::create(project.path()).unwrap();
        sandbox.write_file("src/lib.rs", "changed\n").unwrap();
        sandbox
            .write_file("src/lib.rs", "pub fn f() -> i32 {\n    1\n}\n")
            .unwrap();
        assert_eq!(sandbox.diff().unwrap(), "");
    }

    #[test]
    fn paths_cannot_escape_the_sandbox() {
        let project = project();
        let sandbox = LocalSandbox::create(project.path()).unwrap();
        assert!(sandbox.read_file("../Cargo.toml").is_err());
        assert!(sandbox.write_file("../evil.txt", "x").is_err());
        assert!(sandbox.list_files("..").is_err());
    }

    #[test]
    fn test_command_has_a_scrubbed_environment_and_safe_args() {
        let project = project();
        let sandbox = LocalSandbox::create(project.path()).unwrap();
        let cmd = sandbox.cargo_test_command(Some("my_test")).unwrap();

        let set: Vec<_> = cmd
            .get_envs()
            .map(|(k, _)| k.to_string_lossy().to_uppercase())
            .collect();
        assert!(!set.iter().any(|k| k.contains("MODEL_API_KEY")));
        assert!(set.contains(&"CARGO_TARGET_DIR".to_string()));
        let args: Vec<_> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args, ["test", "--color", "never", "my_test"]); // no Cargo.lock => no --locked

        assert!(
            sandbox
                .cargo_test_command(Some("--manifest-path=/etc/x"))
                .is_err()
        );
        // Nothing in the allowlist should look like a secret.
        for key in ENV_ALLOWLIST {
            assert!(
                !["KEY", "TOKEN", "SECRET", "PASSWORD"]
                    .iter()
                    .any(|w| key.contains(w)),
                "{key}"
            );
        }
    }
}
