//! End-to-end: install the hook, panic for real, and read the saved JSON back.
//! Kept as the only test in this file because the panic hook is process-global.

use std::fs;
use std::panic;

use p2p_capture::{Incident, install, set_context};

#[test]
fn panic_is_saved_as_redacted_incident_json() {
    let dir = tempfile::tempdir().unwrap();
    set_context("app_version", "0.0.1-demo");
    set_context("owner", "grace@example.com");
    install(dir.path());

    // `catch_unwind` lets the test survive the panic; the hook has already run by then.
    let result = panic::catch_unwind(|| {
        let secret = "password=hunter2";
        panic!("config load failed: {secret}");
    });
    assert!(result.is_err());

    let files: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(
        files.len(),
        1,
        "expected exactly one incident file, got {files:?}"
    );

    let raw = fs::read_to_string(&files[0]).unwrap();
    assert!(!raw.contains("hunter2"), "secret leaked: {raw}");
    assert!(!raw.contains("grace@"), "email leaked: {raw}");

    let incident: Incident = serde_json::from_str(&raw).unwrap();
    assert_eq!(incident.message, "config load failed: password=[REDACTED]");
    assert_eq!(incident.context["app_version"], "0.0.1-demo");
    assert!(incident.location.unwrap().file.ends_with("panic_hook.rs"));
    assert!(!incident.backtrace.is_empty());
    assert_eq!(incident.fingerprint.len(), 16);
}
