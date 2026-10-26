//! Secret redaction. Everything that can reach a model or a report goes through [`redact`].
//!
//! This is pattern based, so it is a safety net, not a guarantee: it catches common
//! secret shapes (keys, tokens, emails, passwords in `key=value` pairs, home directories).

use std::sync::LazyLock;

use regex::Regex;

struct Rule {
    re: Regex,
    replacement: &'static str,
}

// Order matters: specific shapes first, the generic `key=value` rule after them.
// `LazyLock` compiles each regex once, the first time it is needed.
static RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    let rule = |pattern: &str, replacement| Rule {
        re: Regex::new(pattern).expect("redaction pattern is valid"),
        replacement,
    };
    vec![
        rule(
            r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            "[REDACTED_PRIVATE_KEY]",
        ),
        rule(
            r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}",
            "[REDACTED_JWT]",
        ),
        // AWS access keys, GitHub tokens, `sk-` style API keys, Slack tokens.
        rule(
            r"\b(?:(?:AKIA|ASIA)[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{16,}|xox[baprs]-[A-Za-z0-9-]{10,})",
            "[REDACTED_TOKEN]",
        ),
        rule(
            r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}",
            "${1} [REDACTED]",
        ),
        // `password=hunter2`, `"api_key": "abc"`, `MODEL_API_KEY=abc`. The name stays, the value goes.
        // No suffix after the keyword, so `max_tokens: 100` is left alone.
        rule(
            r#"(?i)\b([a-z0-9_-]*(?:api[_-]?key|secret|token|password|passwd|pwd))(["']?\s*[:=]\s*)(["']?)[^\s"',;&]+"#,
            "${1}${2}${3}[REDACTED]",
        ),
        rule(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
            "[REDACTED_EMAIL]",
        ),
        // Home directories leak the OS user name. Keep the rest of the path so files stay findable.
        rule(r"(?i)([A-Z]:\\Users\\)[^\\/\s]+", "${1}<user>"),
        rule(r"(/(?:home|Users)/)[^/\s]+", "${1}<user>"),
    ]
});

/// Replace anything that looks like a secret with a `[REDACTED_...]` marker.
pub fn redact(input: &str) -> String {
    let mut text = input.to_string();
    for rule in RULES.iter() {
        // `replace_all` returns a `Cow`: borrowed if nothing matched, owned if it changed.
        text = rule.re.replace_all(&text, rule.replacement).into_owned();
    }
    text
}

static SENSITIVE_KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)api[_-]?key|secret|token|password|passwd|pwd").expect("valid pattern")
});

/// Redact a value stored under a known key (e.g. a context map entry).
///
/// A bare value like `hunter2` has no shape to match, so a sensitive-looking key
/// (`db_password`, `auth_token`, ...) redacts the whole value. Otherwise we fall back to [`redact`].
pub fn redact_value(key: &str, value: &str) -> String {
    if SENSITIVE_KEY.is_match(key) {
        "[REDACTED]".to_string()
    } else {
        redact(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitive_keys_redact_the_whole_value() {
        assert_eq!(redact_value("db_password", "hunter2"), "[REDACTED]");
        assert_eq!(redact_value("API-KEY", "abc"), "[REDACTED]");
        assert_eq!(redact_value("owner", "bob@example.com"), "[REDACTED_EMAIL]");
        assert_eq!(redact_value("app_version", "0.1.0"), "0.1.0");
    }

    // Fake secrets are assembled at runtime so secret scanners don't flag this file.
    fn fake_github_token() -> String {
        format!("ghp_{}", "a1B2".repeat(9))
    }

    #[test]
    fn redacts_emails() {
        let out = redact("failed for jane.doe+test@example.com on retry");
        assert_eq!(out, "failed for [REDACTED_EMAIL] on retry");
    }

    #[test]
    fn redacts_known_token_shapes() {
        let aws = format!("AKIA{}", "ABCD".repeat(4));
        let sk = format!("sk-{}", "x9".repeat(12));
        let gh = fake_github_token();
        let out = redact(&format!("keys: {aws} {sk} {gh}"));
        assert_eq!(
            out,
            "keys: [REDACTED_TOKEN] [REDACTED_TOKEN] [REDACTED_TOKEN]"
        );
    }

    #[test]
    fn redacts_bearer_tokens() {
        let out = redact("Authorization: Bearer abc123.def456-ghi789");
        assert_eq!(out, "Authorization: Bearer [REDACTED]");
    }

    #[test]
    fn redacts_jwt() {
        let jwt = format!(
            "eyJ{}.eyJ{}.{}",
            "a".repeat(10),
            "b".repeat(10),
            "c".repeat(10)
        );
        assert_eq!(
            redact(&format!("token was {jwt}")),
            "token was [REDACTED_JWT]"
        );
    }

    #[test]
    fn redacts_private_key_blocks() {
        let pem =
            "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK\nline2\n-----END RSA PRIVATE KEY-----";
        assert_eq!(
            redact(&format!("before\n{pem}\nafter")),
            "before\n[REDACTED_PRIVATE_KEY]\nafter"
        );
    }

    #[test]
    fn redacts_key_value_pairs_but_keeps_the_name() {
        assert_eq!(
            redact("password=hunter2 user=bob"),
            "password=[REDACTED] user=bob"
        );
        assert_eq!(
            redact(r#"{"api_key": "abc123"}"#),
            r#"{"api_key": "[REDACTED]"}"#
        );
        assert_eq!(redact("MODEL_API_KEY=abc123"), "MODEL_API_KEY=[REDACTED]");
        assert_eq!(
            redact("client_secret: s3cr3t;"),
            "client_secret: [REDACTED];"
        );
    }

    #[test]
    fn redacts_home_directory_user_names() {
        assert_eq!(
            redact(r"C:\Users\Alice\.cargo\registry\src\lib.rs"),
            r"C:\Users\<user>\.cargo\registry\src\lib.rs"
        );
        assert_eq!(
            redact("/home/alice/project/src/main.rs"),
            "/home/<user>/project/src/main.rs"
        );
        assert_eq!(redact("/Users/alice/project"), "/Users/<user>/project");
    }

    #[test]
    fn leaves_ordinary_text_alone() {
        for text in [
            "index out of bounds: the len is 3 but the index is 7",
            "max_tokens: 100",
            "called `Option::unwrap()` on a `None` value",
            "src/inventory.rs:42:9",
        ] {
            assert_eq!(redact(text), text);
        }
    }

    #[test]
    fn is_idempotent() {
        let once = redact("mail bob@example.com password=hunter2");
        assert_eq!(redact(&once), once);
    }
}
