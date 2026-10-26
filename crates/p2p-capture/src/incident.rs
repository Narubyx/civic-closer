use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::redact::{redact, redact_value};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Location {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

/// A captured panic. Every text field is redacted by [`Incident::new`], so an
/// `Incident` is always safe to save, log or send to a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Incident {
    /// Groups repeats of the same bug (see [`fingerprint`]).
    pub fingerprint: String,
    /// Milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    pub message: String,
    pub location: Option<Location>,
    pub thread: String,
    pub backtrace: String,
    /// Free-form key/value context (app version, level, ...). `BTreeMap` keeps the JSON key order stable.
    pub context: BTreeMap<String, String>,
}

impl Incident {
    /// Build an incident from raw, possibly sensitive data. Redaction happens here and only here.
    pub fn new(
        message: &str,
        location: Option<Location>,
        thread: &str,
        backtrace: &str,
        context: &BTreeMap<String, String>,
    ) -> Self {
        let message = redact(message);
        Self {
            fingerprint: fingerprint(&message, location.as_ref()),
            timestamp_ms: now_ms(),
            message,
            location: location.map(|l| Location {
                file: redact(&l.file),
                ..l
            }),
            thread: redact(thread),
            backtrace: redact(backtrace),
            context: context
                .iter()
                .map(|(k, v)| (redact(k), redact_value(k, v)))
                .collect(),
        }
    }

    /// Write `<dir>/<timestamp>-<fingerprint>.json` and return its path.
    pub fn save(&self, dir: &Path) -> io::Result<PathBuf> {
        fs::create_dir_all(dir)?;
        let path = dir.join(format!("{}-{}.json", self.timestamp_ms, self.fingerprint));
        let json = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        fs::write(&path, json)?;
        Ok(path)
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

static NUMBERS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"0x[0-9a-fA-F]+|\d+").expect("valid pattern"));

/// Stable id for "the same bug". It hashes the panic location plus the message with
/// numbers blanked out, so `index is 7` and `index is 9` from one line dedupe together.
/// (We use SHA-256 rather than `DefaultHasher`, whose output may change between Rust versions.)
pub fn fingerprint(message: &str, location: Option<&Location>) -> String {
    let normalized = NUMBERS.replace_all(message, "N");
    let place = location
        .map(|l| format!("{}:{}", l.file, l.line))
        .unwrap_or_default();
    let digest = Sha256::digest(format!("{place}|{normalized}").as_bytes());
    // First 8 bytes (16 hex chars) is plenty for grouping incidents.
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(file: &str, line: u32) -> Option<Location> {
        Some(Location {
            file: file.into(),
            line,
            column: 5,
        })
    }

    fn make(message: &str, location: Option<Location>) -> Incident {
        Incident::new(message, location, "main", "", &BTreeMap::new())
    }

    #[test]
    fn fingerprint_ignores_numbers_but_not_location_or_text() {
        let a = fingerprint(
            "index out of bounds: the len is 3 but the index is 7",
            loc("src/a.rs", 10).as_ref(),
        );
        let b = fingerprint(
            "index out of bounds: the len is 5 but the index is 9",
            loc("src/a.rs", 10).as_ref(),
        );
        let other_line = fingerprint(
            "index out of bounds: the len is 3 but the index is 7",
            loc("src/a.rs", 11).as_ref(),
        );
        let other_text = fingerprint("division by zero", loc("src/a.rs", 10).as_ref());
        assert_eq!(a, b);
        assert_ne!(a, other_line);
        assert_ne!(a, other_text);
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn new_redacts_every_text_field() {
        let mut ctx = BTreeMap::new();
        ctx.insert("user".to_string(), "carol@example.com".to_string());
        ctx.insert("db_password".to_string(), "hunter2".to_string());
        let incident = Incident::new(
            "login failed for dave@example.com token=abc123",
            loc(r"C:\Users\Eve\proj\src\auth.rs", 3),
            "worker",
            "frame at /home/frank/proj/src/auth.rs:3\nBearer abcdefgh12345678",
            &ctx,
        );
        let json = serde_json::to_string(&incident).unwrap();
        for secret in [
            "dave@",
            "carol@",
            "abc123",
            "hunter2",
            "Eve",
            "frank",
            "abcdefgh12345678",
        ] {
            assert!(!json.contains(secret), "leaked `{secret}` in {json}");
        }
        assert!(incident.message.contains("[REDACTED_EMAIL]"));
    }

    #[test]
    fn fingerprint_is_computed_from_redacted_message() {
        // Two different users hitting the same bug must dedupe together.
        let a = make(
            "cannot load profile for alice@example.com",
            loc("src/p.rs", 1),
        );
        let b = make(
            "cannot load profile for bob@example.com",
            loc("src/p.rs", 1),
        );
        assert_eq!(a.fingerprint, b.fingerprint);
    }

    #[test]
    fn json_round_trips() {
        let incident = make("boom", loc("src/lib.rs", 8));
        let json = serde_json::to_string(&incident).unwrap();
        assert_eq!(serde_json::from_str::<Incident>(&json).unwrap(), incident);
    }

    #[test]
    fn save_writes_a_named_json_file() {
        let dir = tempfile::tempdir().unwrap();
        let incident = make("boom", loc("src/lib.rs", 8));
        let path = incident.save(&dir.path().join("nested")).unwrap();
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(&format!("-{}.json", incident.fingerprint))
        );
        let loaded: Incident = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(loaded, incident);
    }
}
