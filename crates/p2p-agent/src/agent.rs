//! The fix loop: the model reads code, reproduces the bug with a failing test, patches it,
//! and the harness verifies the result by actually running `cargo test`.

use anyhow::Result;
use p2p_capture::{Incident, redact};
use serde_json::{Value, json};

use crate::client::{Model, function_output};
use crate::sandbox::Sandbox;
use crate::tools::{self, Report, RunRecord, Session};

/// Hard stop so a confused model cannot loop (and bill) forever.
pub const MAX_TURNS: usize = 25;
/// How many times we remind a model that answered without calling a tool.
const MAX_NUDGES: usize = 2;
/// Backtraces can be long; the top frames are what matter.
const MAX_BACKTRACE_CHARS: usize = 6_000;

const INSTRUCTIONS: &str = "\
You are Panic-to-Patch, an autonomous debugging agent for Rust projects. You are given a \
redacted panic incident from a project. Work in a sandbox copy of the project using the tools. \
Follow this workflow:
1. Explore: use list_files and read_file, starting at the panic location and backtrace.
2. State a root-cause hypothesis (briefly, in plain text).
3. Reproduce: write a regression test that fails because of THIS bug. Put it in the file's \
`#[cfg(test)]` module or in `tests/`. Run it with run_tests (filter = its name). It must end \
as TestsFailed; BuildFailed means your test does not compile, so fix that first.
4. Fix: make the smallest correct change with edit_file. Do not weaken or delete tests to make \
them pass, and do not touch unrelated code.
5. Verify: run_tests with no filter. The whole suite must pass.
6. Finish by calling write_report. Use status `fixed` only if step 5 passed on your final code; \
otherwise use `unable_to_fix` and say honestly what you found. Set confidence to what the \
evidence supports.
Notes: paths are relative to the project root. Secrets in the source appear as [REDACTED...] \
markers; never edit those lines. Tool errors are explained in their text: read them and adapt.";

/// Progress updates, for the CLI now and the terminal UI later.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    Turn(usize),
    /// Free text from the model (its reasoning between tool calls).
    Text(String),
    ToolCall {
        name: String,
        summary: String,
    },
    ToolResult {
        name: String,
        ok: bool,
        summary: String,
    },
}

pub struct AgentOutcome {
    /// `None` if the model never produced an accepted report (turn limit, repeated non-answers).
    pub report: Option<Report>,
    /// A `fixed` report that the harness confirmed with recorded red/green runs.
    pub verified: bool,
    /// Unified diff of everything changed in the sandbox.
    pub patch: String,
    pub runs: Vec<RunRecord>,
    pub turns: usize,
}

/// Build the first user message from the incident. Everything is redacted again as a safety net.
fn incident_prompt(incident: &Incident) -> String {
    let location = incident
        .location
        .as_ref()
        .map(|l| format!("{}:{}:{}", l.file, l.line, l.column))
        .unwrap_or_else(|| "unknown".into());
    let context: String = incident
        .context
        .iter()
        .map(|(k, v)| format!("- {k}: {v}\n"))
        .collect();
    let backtrace: String = incident
        .backtrace
        .chars()
        .take(MAX_BACKTRACE_CHARS)
        .collect();
    redact(&format!(
        "A panic was captured.\n\nmessage: {}\nlocation: {location}\nthread: {}\n\
         context:\n{context}\nbacktrace:\n{backtrace}\n\nInvestigate and fix it.",
        incident.message, incident.thread
    ))
}

pub async fn run_agent<M: Model, S: Sandbox>(
    model: &M,
    sandbox: &S,
    incident: &Incident,
    on_event: &mut dyn FnMut(AgentEvent),
) -> Result<AgentOutcome> {
    let tool_defs = tools::definitions();
    let mut session = Session::new(sandbox);
    let mut input: Vec<Value> = vec![
        json!({ "role": "system", "content": INSTRUCTIONS }),
        json!({ "role": "user", "content": incident_prompt(incident) }),
    ];
    let (mut turns, mut nudges) = (0, 0);

    while turns < MAX_TURNS && session.report.is_none() {
        turns += 1;
        on_event(AgentEvent::Turn(turns));
        let response = model.create(&input, &tool_defs).await?;

        let text = response.text();
        if !text.trim().is_empty() {
            on_event(AgentEvent::Text(text));
        }
        let calls = response.function_calls()?;
        // Echo the model's own output back, then answer each call with the same `call_id`.
        input.extend(response.output.iter().cloned());

        if calls.is_empty() {
            nudges += 1;
            if nudges > MAX_NUDGES {
                break;
            }
            input.push(json!({
                "role": "user",
                "content": "Keep going with the tools. When you are done, call write_report."
            }));
            continue;
        }

        for call in &calls {
            on_event(AgentEvent::ToolCall {
                name: call.name.clone(),
                summary: summarize(&call.arguments),
            });
            let result = session.execute(call).await;
            let (ok, text) = match result {
                Ok(text) => (true, text),
                Err(text) => (false, text),
            };
            on_event(AgentEvent::ToolResult {
                name: call.name.clone(),
                ok,
                summary: text
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(100)
                    .collect(),
            });
            input.push(function_output(&call.call_id, &text));
        }
    }

    Ok(AgentOutcome {
        verified: session.verified,
        report: session.report,
        patch: sandbox.diff()?,
        runs: session.runs,
        turns,
    })
}

/// One-line view of a tool call's arguments; long text fields (file bodies) are left out.
fn summarize(args: &Value) -> String {
    let Some(map) = args.as_object() else {
        return String::new();
    };
    map.iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "content" | "old" | "new" | "explanation" | "root_cause"
            )
        })
        .map(|(key, value)| {
            let text = value
                .as_str()
                .map_or_else(|| value.to_string(), str::to_string);
            let short: String = text.chars().take(60).collect();
            format!("{key}={short}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::sandbox::TestOutcome;
    use crate::testing::{FakeSandbox, ScriptedModel, calls, text};
    use crate::tools::Status;

    fn incident() -> Incident {
        let mut ctx = BTreeMap::new();
        ctx.insert(
            "scenario".to_string(),
            "empty shelf, owner ops@example.com".to_string(),
        );
        Incident::new(
            "attempt to divide by zero",
            Some(p2p_capture::Location {
                file: "src/lib.rs".into(),
                line: 6,
                column: 5,
            }),
            "main",
            "0: inventory::average_price_cents",
            &ctx,
        )
    }

    fn report_args(status: &str, test: &str) -> Value {
        json!({ "status": status, "root_cause": "empty slice", "explanation": "guard it",
                "confidence": "high", "regression_test_name": test })
    }

    #[tokio::test]
    async fn happy_path_reads_reproduces_fixes_verifies_and_reports() {
        let sb = FakeSandbox::with_files(&[("src/lib.rs", "fn avg() { total / len }\n")]);
        sb.script_runs(&[TestOutcome::TestsFailed, TestOutcome::Passed]);
        let model = ScriptedModel::new(vec![
            // Turn 1: two parallel calls.
            calls(
                "r1",
                &[
                    ("list_files", json!({})),
                    ("read_file", json!({ "path": "src/lib.rs" })),
                ],
            ),
            text("r2_text", "Hypothesis: no guard for empty input."),
            calls(
                "r3",
                &[(
                    "write_file",
                    json!({ "path": "tests/empty.rs", "content": "fn empty_shelf() {}\n" }),
                )],
            ),
            calls("r4", &[("run_tests", json!({ "filter": "empty_shelf" }))]),
            calls(
                "r5",
                &[(
                    "edit_file",
                    json!({ "path": "src/lib.rs", "old": "total / len", "new": "0" }),
                )],
            ),
            calls("r6", &[("run_tests", json!({}))]),
            calls(
                "r7",
                &[("write_report", report_args("fixed", "empty_shelf"))],
            ),
        ]);

        let mut events = Vec::new();
        let outcome = run_agent(&model, &sb, &incident(), &mut |e| events.push(e))
            .await
            .unwrap();

        assert!(outcome.verified);
        assert_eq!(outcome.report.unwrap().status, Status::Fixed);
        assert_eq!(outcome.runs.len(), 2);
        assert!(outcome.patch.contains("empty_shelf") && outcome.patch.contains("{ 0 }"));
        // r2 was text-only: the loop nudged instead of stopping, and the conversation kept growing.
        assert_eq!(outcome.turns, 7);
        let sizes = model.input_sizes.lock().unwrap().clone();
        assert!(sizes.windows(2).all(|w| w[0] < w[1]), "{sizes:?}");
        // Progress events were emitted.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::Text(t) if t.contains("Hypothesis")))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolCall { name, .. } if name == "run_tests"))
        );
    }

    #[tokio::test]
    async fn rejected_fixed_report_lets_the_model_recover() {
        let sb = FakeSandbox::with_files(&[("src/lib.rs", "x\n")]);
        sb.script_runs(&[TestOutcome::TestsFailed, TestOutcome::Passed]);
        let model = ScriptedModel::new(vec![
            calls(
                "a",
                &[(
                    "write_file",
                    json!({ "path": "t.rs", "content": "fn t_empty() {}\n" }),
                )],
            ),
            // Claims success without any test run: must be rejected.
            calls("b", &[("write_report", report_args("fixed", "t_empty"))]),
            calls("c", &[("run_tests", json!({}))]),
            calls(
                "d",
                &[(
                    "edit_file",
                    json!({ "path": "src/lib.rs", "old": "x", "new": "y" }),
                )],
            ),
            calls("e", &[("run_tests", json!({}))]),
            calls("f", &[("write_report", report_args("fixed", "t_empty"))]),
        ]);
        let outcome = run_agent(&model, &sb, &incident(), &mut |_| {})
            .await
            .unwrap();
        assert!(outcome.verified);
        assert_eq!(outcome.turns, 6);
    }

    #[tokio::test]
    async fn a_model_that_never_uses_tools_is_stopped() {
        let sb = FakeSandbox::with_files(&[]);
        let model = ScriptedModel::new(
            (0..5)
                .map(|i| text(&format!("t{i}"), "thinking..."))
                .collect(),
        );
        let outcome = run_agent(&model, &sb, &incident(), &mut |_| {})
            .await
            .unwrap();
        assert!(outcome.report.is_none() && !outcome.verified);
        assert_eq!(outcome.turns, MAX_NUDGES + 1);
    }

    #[tokio::test]
    async fn turn_limit_stops_a_looping_model() {
        let sb = FakeSandbox::with_files(&[("a.rs", "x\n")]);
        let script = (0..MAX_TURNS + 5)
            .map(|i| {
                calls(
                    &format!("l{i}"),
                    &[("read_file", json!({ "path": "a.rs" }))],
                )
            })
            .collect();
        let outcome = run_agent(&ScriptedModel::new(script), &sb, &incident(), &mut |_| {})
            .await
            .unwrap();
        assert!(outcome.report.is_none());
        assert_eq!(outcome.turns, MAX_TURNS);
    }

    #[test]
    fn prompt_is_redacted_and_has_the_essentials() {
        let prompt = incident_prompt(&incident());
        assert!(prompt.contains("attempt to divide by zero") && prompt.contains("src/lib.rs:6:5"));
        assert!(!prompt.contains("ops@example.com"));
    }

    #[test]
    fn summaries_skip_file_bodies() {
        let s = summarize(&json!({ "path": "src/a.rs", "content": "SECRET BODY", "filter": "t" }));
        assert!(
            s.contains("path=src/a.rs") && s.contains("filter=t") && !s.contains("BODY"),
            "{s}"
        );
    }
}
