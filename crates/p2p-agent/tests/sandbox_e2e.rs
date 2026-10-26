//! End-to-end sandbox tests: they run the real `cargo test` on tiny fixture projects.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use p2p_agent::sandbox::check_patch_applies;
use p2p_agent::{LocalSandbox, Sandbox, TestOptions, TestOutcome};

fn write_project(dir: &Path, lib_rs: &str) {
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(dir.join("src/lib.rs"), lib_rs).unwrap();
}

const BUGGY: &str = "pub fn add(a: i32, b: i32) -> i32 {
    a - b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_two_numbers() {
        assert_eq!(add(2, 3), 5);
    }
}
";

#[tokio::test]
async fn red_fix_green_and_patch_applies_to_the_original() {
    let project = tempfile::tempdir().unwrap();
    write_project(project.path(), BUGGY);
    let sandbox = LocalSandbox::create(project.path()).unwrap();
    let opts = TestOptions {
        timeout: Duration::from_secs(300),
        ..Default::default()
    };

    // Red: the test fails (not the build).
    let red = sandbox.run_tests(&opts).await.unwrap();
    assert_eq!(red.outcome, TestOutcome::TestsFailed, "{}", red.output);
    assert!(red.output.contains("adds_two_numbers"), "{}", red.output);
    assert!(
        !red.output.contains(&*sandbox.root().to_string_lossy()),
        "sandbox path leaked into output"
    );

    // Fix, then green.
    sandbox.edit_file("src/lib.rs", "a - b", "a + b").unwrap();
    let green = sandbox.run_tests(&opts).await.unwrap();
    assert_eq!(green.outcome, TestOutcome::Passed, "{}", green.output);
    assert_eq!(green.exit_code, Some(0));

    // The patch applies to the untouched original (checked, never applied).
    let patch = sandbox.diff().unwrap();
    assert!(patch.contains("-    a - b\n+    a + b\n"), "{patch}");
    check_patch_applies(project.path(), &patch).unwrap();
    assert!(
        fs::read_to_string(project.path().join("src/lib.rs"))
            .unwrap()
            .contains("a - b")
    );

    // A patch made for different content must be rejected.
    fs::write(project.path().join("src/lib.rs"), "pub fn add() {}\n").unwrap();
    assert!(check_patch_applies(project.path(), &patch).is_err());

    // A compile error is reported as BuildFailed, distinct from a failing test.
    sandbox.edit_file("src/lib.rs", "a + b", "a +").unwrap();
    let broken = sandbox.run_tests(&opts).await.unwrap();
    assert_eq!(
        broken.outcome,
        TestOutcome::BuildFailed,
        "{}",
        broken.output
    );
}

#[tokio::test]
async fn filter_selects_tests() {
    let project = tempfile::tempdir().unwrap();
    write_project(project.path(), BUGGY);
    let sandbox = LocalSandbox::create(project.path()).unwrap();
    let opts = TestOptions {
        filter: Some("no_such_test".into()),
        timeout: Duration::from_secs(300),
    };
    // Nothing matches the filter, so nothing fails.
    assert_eq!(
        sandbox.run_tests(&opts).await.unwrap().outcome,
        TestOutcome::Passed
    );
}

#[tokio::test]
async fn timeout_kills_the_whole_process_tree() {
    let project = tempfile::tempdir().unwrap();
    // Sleeps 5s, then leaves a marker file. If the test binary survives the kill, the marker appears.
    write_project(
        project.path(),
        "#[cfg(test)]
mod tests {
    #[test]
    fn sleepy() {
        std::thread::sleep(std::time::Duration::from_secs(5));
        std::fs::write(\"survived.txt\", \"yes\").unwrap();
    }
}
",
    );
    let sandbox = LocalSandbox::create(project.path()).unwrap();

    // Warm-up compiles the test binary, so the timed run below only measures execution.
    let warm = TestOptions {
        filter: Some("no_such_test".into()),
        timeout: Duration::from_secs(300),
    };
    assert_eq!(
        sandbox.run_tests(&warm).await.unwrap().outcome,
        TestOutcome::Passed
    );

    let started = Instant::now();
    let run = sandbox
        .run_tests(&TestOptions {
            filter: None,
            timeout: Duration::from_secs(2),
        })
        .await
        .unwrap();
    assert_eq!(run.outcome, TestOutcome::TimedOut);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "took {:?}",
        started.elapsed()
    );

    // Wait past the point where a surviving test binary would have written its marker.
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert!(
        !sandbox.root().join("survived.txt").exists(),
        "test binary outlived the timeout"
    );
}
