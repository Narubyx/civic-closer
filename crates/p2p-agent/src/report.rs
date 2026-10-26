//! The Markdown incident report and the files written next to it.
//!
//! Verification facts (test runs, patch) come from the harness's own records. Only the
//! root cause, explanation and confidence are the model's words, and they are labelled so.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use p2p_capture::Incident;

use crate::agent::AgentOutcome;
use crate::sandbox::TestOutcome;
use crate::tools::{Confidence, RunRecord, Status};

/// Lines of test output quoted per run in the report.
const EXCERPT_LINES: usize = 25;

pub fn render_markdown(incident: &Incident, outcome: &AgentOutcome) -> String {
    let mut md = String::new();
    let status = match (&outcome.report, outcome.verified) {
        (Some(r), true) if r.status == Status::Fixed => "Fixed and verified",
        (Some(r), false) if r.status == Status::Fixed => "Fix proposed, NOT verified",
        (Some(_), _) => "Unable to fix",
        (None, _) => "No report (the agent stopped before finishing)",
    };
    let _ = writeln!(md, "# Incident `{}`\n", incident.fingerprint);
    let _ = writeln!(md, "- **Status:** {status}");
    if let Some(report) = &outcome.report {
        let _ = writeln!(
            md,
            "- **Confidence:** {} (reported by the model)",
            confidence(report.confidence)
        );
    }
    let _ = writeln!(md, "- **Agent turns:** {}\n", outcome.turns);

    md.push_str("## Error\n\n");
    let _ = writeln!(
        md,
        "- **Message:** `{}`",
        incident.message.replace('`', "'")
    );
    if let Some(l) = &incident.location {
        let _ = writeln!(md, "- **Location:** `{}:{}:{}`", l.file, l.line, l.column);
    }
    let _ = writeln!(md, "- **Thread:** `{}`", incident.thread);
    for (key, value) in &incident.context {
        let _ = writeln!(md, "- **{key}:** {value}");
    }

    md.push_str("\n## Log (backtrace)\n\n");
    md.push_str(&fence("text", &incident.backtrace));

    if let Some(report) = &outcome.report {
        md.push_str("\n## Root cause\n\n");
        md.push_str(report.root_cause.trim());
        md.push_str("\n\n## Explanation of the fix\n\n");
        md.push_str(report.explanation.trim());
        md.push('\n');

        md.push_str("\n## Regression test\n\n");
        match (&report.regression_test_name, &report.regression_test_path) {
            (Some(name), Some(path)) => {
                let _ = writeln!(md, "`{name}` in `{path}`");
            }
            (Some(name), None) => {
                let _ = writeln!(md, "`{name}`");
            }
            _ => md.push_str("None was reported.\n"),
        }
    }

    md.push_str("\n## Patch\n\n");
    if outcome.patch.is_empty() {
        md.push_str("No changes were made.\n");
    } else {
        md.push_str(&fence("diff", &outcome.patch));
    }

    md.push_str("\n## Test results (recorded by the harness)\n\n");
    md.push_str("| # | Filter | Result | Time |\n|---|--------|--------|------|\n");
    for (i, run) in outcome.runs.iter().enumerate() {
        let filter = run.filter.as_deref().unwrap_or("(all tests)");
        let _ = writeln!(
            md,
            "| {} | `{filter}` | {:?} | {:.1}s |",
            i + 1,
            run.outcome,
            run.duration.as_secs_f32()
        );
    }
    if outcome.runs.is_empty() {
        md.push_str("| - | - | no test was run | - |\n");
    }
    if let Some(red) = outcome
        .runs
        .iter()
        .find(|r| r.outcome == TestOutcome::TestsFailed)
    {
        md.push_str("\n**Before the fix** (first failing run):\n\n");
        md.push_str(&fence("text", &excerpt(red)));
    }
    if let Some(green) = outcome
        .runs
        .last()
        .filter(|r| r.outcome == TestOutcome::Passed)
    {
        md.push_str("\n**Final run:**\n\n");
        md.push_str(&fence("text", &excerpt(green)));
    }
    md
}

/// Write `report.md` (and `fix.patch` if there is one) into `<out_dir>/<fingerprint>/`.
pub fn save(out_dir: &Path, incident: &Incident, outcome: &AgentOutcome) -> io::Result<PathBuf> {
    let dir = out_dir.join(&incident.fingerprint);
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("report.md"), render_markdown(incident, outcome))?;
    if !outcome.patch.is_empty() {
        fs::write(dir.join("fix.patch"), &outcome.patch)?;
    }
    Ok(dir)
}

fn confidence(c: Confidence) -> &'static str {
    match c {
        Confidence::Low => "low",
        Confidence::Medium => "medium",
        Confidence::High => "high",
    }
}

/// The last few lines of a run's output, which hold the failure or the summary.
fn excerpt(run: &RunRecord) -> String {
    let lines: Vec<&str> = run
        .output
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    lines[lines.len().saturating_sub(EXCERPT_LINES)..].join("\n")
}

/// A code fence that is always longer than any backtick run inside `text`.
fn fence(lang: &str, text: &str) -> String {
    let mut longest = 0;
    let mut run = 0;
    for c in text.chars() {
        run = if c == '`' { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    let ticks = "`".repeat((longest + 1).max(3));
    format!("{ticks}{lang}\n{}\n{ticks}\n", text.trim_end())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::tools::Report;

    fn incident() -> Incident {
        Incident::new(
            "attempt to divide by zero",
            Some(p2p_capture::Location {
                file: "src/lib.rs".into(),
                line: 6,
                column: 5,
            }),
            "main",
            "0: inventory::average_price_cents",
            &Default::default(),
        )
    }

    fn run(filter: Option<&str>, outcome: TestOutcome, output: &str) -> RunRecord {
        RunRecord {
            filter: filter.map(Into::into),
            outcome,
            exit_code: Some(0),
            duration: Duration::from_secs(2),
            output: output.into(),
            edit_count: 0,
        }
    }

    fn outcome(verified: bool, status: Status) -> AgentOutcome {
        AgentOutcome {
            report: Some(Report {
                status,
                root_cause: "Division by an empty slice length.".into(),
                explanation: "Return 0 for empty input.".into(),
                confidence: Confidence::High,
                regression_test_name: Some("empty_shelf".into()),
                regression_test_path: Some("src/lib.rs".into()),
            }),
            verified,
            patch: "diff --git a/src/lib.rs b/src/lib.rs\n-old\n+new\n".into(),
            runs: vec![
                run(
                    Some("empty_shelf"),
                    TestOutcome::TestsFailed,
                    "test empty_shelf ... FAILED\n\nattempt to divide by zero",
                ),
                run(None, TestOutcome::Passed, "test result: ok. 3 passed"),
            ],
            turns: 9,
        }
    }

    #[test]
    fn verified_report_has_all_required_sections() {
        let md = render_markdown(&incident(), &outcome(true, Status::Fixed));
        for needle in [
            "# Incident `",
            "**Status:** Fixed and verified",
            "## Error",
            "## Log (backtrace)",
            "## Root cause",
            "Division by an empty slice length.",
            "## Regression test",
            "`empty_shelf` in `src/lib.rs`",
            "## Patch",
            "```diff",
            "## Test results (recorded by the harness)",
            "| 1 | `empty_shelf` | TestsFailed |",
            "| 2 | `(all tests)` | Passed |",
            "attempt to divide by zero",
            "test result: ok. 3 passed",
            "**Confidence:** high",
        ] {
            assert!(md.contains(needle), "missing `{needle}` in:\n{md}");
        }
    }

    #[test]
    fn unverified_and_missing_reports_are_labelled_honestly() {
        assert!(
            render_markdown(&incident(), &outcome(false, Status::Fixed)).contains("NOT verified")
        );
        assert!(
            render_markdown(&incident(), &outcome(false, Status::UnableToFix))
                .contains("Unable to fix")
        );
        let mut none = outcome(false, Status::Fixed);
        none.report = None;
        none.patch.clear();
        none.runs.clear();
        let md = render_markdown(&incident(), &none);
        assert!(
            md.contains("No report")
                && md.contains("No changes were made")
                && md.contains("no test was run")
        );
    }

    #[test]
    fn fence_outgrows_backticks_in_the_text() {
        assert!(fence("diff", "a ```` b").starts_with("`````diff"));
        assert!(fence("text", "plain").starts_with("```text"));
    }

    #[test]
    fn save_writes_report_and_patch_under_the_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let inc = incident();
        let out = save(dir.path(), &inc, &outcome(true, Status::Fixed)).unwrap();
        assert!(out.ends_with(&inc.fingerprint));
        assert!(
            fs::read_to_string(out.join("report.md"))
                .unwrap()
                .contains("# Incident")
        );
        assert!(
            fs::read_to_string(out.join("fix.patch"))
                .unwrap()
                .starts_with("diff --git")
        );
    }
}
