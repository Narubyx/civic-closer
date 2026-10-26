//! The tools the model can call, and the bookkeeping that keeps its final report honest.
//!
//! The model never gets to *claim* a fix: [`Session`] records every test run itself and
//! refuses a `fixed` report unless the recorded runs prove it.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use p2p_capture::redact;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::client::FunctionCall;
use crate::sandbox::{Sandbox, TestOptions, TestOutcome};

/// Largest file we hand to the model, in characters.
const MAX_READ_CHARS: usize = 100_000;
/// Test output returned to the model (the full output is kept for the report).
const MAX_TEST_CHARS_FOR_MODEL: usize = 12_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Fixed,
    UnableToFix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    Low,
    Medium,
    High,
}

/// The model's final answer (arguments of `write_report`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub status: Status,
    pub root_cause: String,
    pub explanation: String,
    pub confidence: Confidence,
    pub regression_test_name: Option<String>,
    pub regression_test_path: Option<String>,
}

/// One `cargo test` run, as seen by the harness (not as described by the model).
#[derive(Debug, Clone)]
pub struct RunRecord {
    pub filter: Option<String>,
    pub outcome: TestOutcome,
    pub exit_code: Option<i32>,
    pub duration: Duration,
    pub output: String,
    /// How many edits had been made when this ran; tells us if anything changed since.
    pub edit_count: usize,
}

fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    // Flat Responses-API shape: `name`/`parameters` sit next to `type`.
    json!({
        "type": "function",
        "name": name,
        "description": description,
        "parameters": { "type": "object", "properties": properties, "required": required }
    })
}

pub fn definitions() -> Vec<Value> {
    vec![
        tool(
            "list_files",
            "List files in the project (paths relative to the project root, `/` separators).",
            json!({ "dir": { "type": "string", "description": "Folder to list; omit for the whole project." } }),
            &[],
        ),
        tool(
            "read_file",
            "Read a text file. Secrets in the source are replaced by [REDACTED...] markers; \
             never edit those lines.",
            json!({ "path": { "type": "string" } }),
            &["path"],
        ),
        tool(
            "write_file",
            "Create a new file (e.g. a test file) or fully replace one. For small changes to an \
             existing file use edit_file instead.",
            json!({ "path": { "type": "string" }, "content": { "type": "string" } }),
            &["path", "content"],
        ),
        tool(
            "edit_file",
            "Replace `old` with `new` in a file. `old` must match the file exactly (including \
             whitespace) and occur exactly once, so include enough surrounding lines.",
            json!({
                "path": { "type": "string" },
                "old": { "type": "string" },
                "new": { "type": "string" }
            }),
            &["path", "old", "new"],
        ),
        tool(
            "run_tests",
            "Run `cargo test` in the sandbox copy. Optional `filter` runs only tests whose name \
             contains it. Result is one of Passed, TestsFailed, BuildFailed (does not compile) or TimedOut.",
            json!({ "filter": { "type": "string" } }),
            &[],
        ),
        tool(
            "write_report",
            "Finish the task. Use status `fixed` only after run_tests (no filter) passed on your \
             final code and a regression test failed before the fix; the harness checks this. \
             If you could not fix it, use `unable_to_fix` and explain honestly.",
            json!({
                "status": { "type": "string", "enum": ["fixed", "unable_to_fix"] },
                "root_cause": { "type": "string", "description": "What is wrong and why it panics." },
                "explanation": { "type": "string", "description": "What the patch changes and why it is right." },
                "confidence": { "type": "string", "enum": ["low", "medium", "high"] },
                "regression_test_name": { "type": "string", "description": "Name of the new test function." },
                "regression_test_path": { "type": "string", "description": "File containing the new test." }
            }),
            &["status", "root_cause", "explanation", "confidence"],
        ),
    ]
}

#[derive(Deserialize)]
struct ListArgs {
    dir: Option<String>,
}
#[derive(Deserialize)]
struct PathArgs {
    path: String,
}
#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}
#[derive(Deserialize)]
struct EditArgs {
    path: String,
    old: String,
    new: String,
}
#[derive(Deserialize)]
struct TestArgs {
    filter: Option<String>,
}

/// Executes tool calls against a sandbox and remembers what happened.
pub struct Session<'a, S> {
    sandbox: &'a S,
    edit_count: usize,
    pub runs: Vec<RunRecord>,
    /// Set once a `write_report` call was accepted.
    pub report: Option<Report>,
    /// True if that report said `fixed` and the recorded runs back it up.
    pub verified: bool,
}

impl<'a, S: Sandbox> Session<'a, S> {
    pub fn new(sandbox: &'a S) -> Self {
        Self {
            sandbox,
            edit_count: 0,
            runs: Vec::new(),
            report: None,
            verified: false,
        }
    }

    /// Run one tool call. The `Err` text is sent to the model too: it explains how to recover.
    pub async fn execute(&mut self, call: &FunctionCall) -> Result<String, String> {
        self.dispatch(call)
            .await
            .map_err(|e| format!("error: {e:#}"))
    }

    async fn dispatch(&mut self, call: &FunctionCall) -> Result<String> {
        fn args<T: serde::de::DeserializeOwned>(call: &FunctionCall) -> Result<T> {
            serde_json::from_value(call.arguments.clone())
                .with_context(|| format!("invalid arguments for `{}`", call.name))
        }
        match call.name.as_str() {
            "list_files" => {
                let a: ListArgs = args(call)?;
                let files = self.sandbox.list_files(a.dir.as_deref().unwrap_or(""))?;
                Ok(if files.is_empty() {
                    "(no files)".into()
                } else {
                    files.join("\n")
                })
            }
            "read_file" => {
                let a: PathArgs = args(call)?;
                // Redact before the text can reach the model.
                let text = redact(&self.sandbox.read_file(&a.path)?);
                Ok(truncate_head(&text, MAX_READ_CHARS))
            }
            "write_file" => {
                let a: WriteArgs = args(call)?;
                self.sandbox.write_file(&a.path, &a.content)?;
                self.edit_count += 1;
                Ok(format!("wrote {} bytes to {}", a.content.len(), a.path))
            }
            "edit_file" => {
                let a: EditArgs = args(call)?;
                self.sandbox.edit_file(&a.path, &a.old, &a.new)?;
                self.edit_count += 1;
                Ok(format!("edited {}", a.path))
            }
            "run_tests" => self.run_tests(args(call)?).await,
            "write_report" => self.write_report(args(call)?),
            other => bail!("unknown tool `{other}`"),
        }
    }

    async fn run_tests(&mut self, a: TestArgs) -> Result<String> {
        let filter = a.filter.filter(|f| !f.trim().is_empty());
        let opts = TestOptions {
            filter: filter.clone(),
            ..Default::default()
        };
        let run = self.sandbox.run_tests(&opts).await?;
        let hint = match run.outcome {
            TestOutcome::BuildFailed => {
                "\nnote: the code does not compile, so this is NOT a failing regression test."
            }
            TestOutcome::TimedOut => "\nnote: the run hit the time limit.",
            _ => "",
        };
        let reply = format!(
            "outcome: {:?} (exit code {:?}, {:.1}s){hint}\n{}",
            run.outcome,
            run.exit_code,
            run.duration.as_secs_f32(),
            truncate_tail(&run.output, MAX_TEST_CHARS_FOR_MODEL),
        );
        self.runs.push(RunRecord {
            filter,
            outcome: run.outcome,
            exit_code: run.exit_code,
            duration: run.duration,
            output: run.output,
            edit_count: self.edit_count,
        });
        Ok(reply)
    }

    fn write_report(&mut self, report: Report) -> Result<String> {
        if report.status == Status::Fixed {
            let patch = self.sandbox.diff()?;
            if let Some(problem) = self.verification_problem(&report, &patch) {
                bail!("report rejected: {problem}");
            }
            self.verified = true;
        }
        self.report = Some(report);
        Ok("report accepted; you are done".into())
    }

    /// Why a `fixed` report is not backed by the recorded runs (`None` = it is).
    fn verification_problem(&self, report: &Report, patch: &str) -> Option<String> {
        if patch.is_empty() {
            return Some("the sandbox has no changes, so there is nothing to patch".into());
        }
        match report.regression_test_name.as_deref() {
            Some(name) if patch.contains(name) => {}
            Some(name) => {
                return Some(format!(
                    "regression test `{name}` is not part of your changes"
                ));
            }
            None => return Some("regression_test_name is required for status `fixed`".into()),
        }
        let Some(red) = self
            .runs
            .iter()
            .find(|r| r.outcome == TestOutcome::TestsFailed)
        else {
            return Some(
                "no run with a failing test was recorded; write a regression test that fails \
                 before the fix (BuildFailed does not count)"
                    .into(),
            );
        };
        match self.runs.last() {
            Some(last)
                if last.outcome == TestOutcome::Passed
                    && last.filter.is_none()
                    && last.edit_count == self.edit_count =>
            {
                if last.edit_count == red.edit_count {
                    return Some("tests went from failing to passing without any edit".into());
                }
                None
            }
            _ => Some(
                "run_tests without a filter must pass on your final code (no edits after that run)"
                    .into(),
            ),
        }
    }
}

/// Keep the first `max` characters (cutting on a char boundary).
fn truncate_head(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((cut, _)) => format!(
            "{}\n[... file truncated after {max} characters ...]",
            &text[..cut]
        ),
        None => text.to_string(),
    }
}

/// Keep the last `max` characters: compile errors and test summaries come last.
fn truncate_tail(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max {
        return text.to_string();
    }
    let skip = count - max;
    let start = text.char_indices().nth(skip).map_or(0, |(i, _)| i);
    format!("[... earlier output truncated ...]\n{}", &text[start..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::FakeSandbox;

    fn call(name: &str, arguments: Value) -> FunctionCall {
        FunctionCall {
            call_id: "c".into(),
            name: name.into(),
            arguments,
        }
    }

    fn fixed_report(test: Option<&str>) -> Value {
        json!({
            "status": "fixed", "root_cause": "rc", "explanation": "ex", "confidence": "high",
            "regression_test_name": test
        })
    }

    #[test]
    fn every_tool_has_a_flat_schema() {
        let defs = definitions();
        let names: Vec<_> = defs.iter().map(|d| d["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            [
                "list_files",
                "read_file",
                "write_file",
                "edit_file",
                "run_tests",
                "write_report"
            ]
        );
        for d in &defs {
            assert_eq!(d["type"], "function");
            assert_eq!(d["parameters"]["type"], "object");
        }
    }

    #[tokio::test]
    async fn read_file_redacts_secrets_before_the_model_sees_them() {
        let sb = FakeSandbox::with_files(&[(
            "src/config.rs",
            "const API_KEY: &str = \"x\"; // mail ops@example.com\npassword=hunter2\n",
        )]);
        let mut s = Session::new(&sb);
        let out = s
            .execute(&call("read_file", json!({ "path": "src/config.rs" })))
            .await
            .unwrap();
        assert!(!out.contains("hunter2") && !out.contains("ops@"), "{out}");
        assert!(out.contains("[REDACTED"));
    }

    #[tokio::test]
    async fn bad_arguments_and_unknown_tools_become_error_text() {
        let sb = FakeSandbox::with_files(&[]);
        let mut s = Session::new(&sb);
        let e = s
            .execute(&call("read_file", json!({ "nope": 1 })))
            .await
            .unwrap_err();
        assert!(
            e.starts_with("error:") && e.contains("invalid arguments"),
            "{e}"
        );
        let e = s.execute(&call("rm_rf", json!({}))).await.unwrap_err();
        assert!(e.contains("unknown tool"), "{e}");
    }

    #[tokio::test]
    async fn fixed_report_needs_red_then_green_on_the_final_code() {
        let sb = FakeSandbox::with_files(&[("src/lib.rs", "fn f() { a - b }\n")]);
        sb.script_runs(&[TestOutcome::TestsFailed, TestOutcome::Passed]);
        let mut s = Session::new(&sb);
        let report = || call("write_report", fixed_report(Some("regression_test")));

        // Nothing changed yet.
        assert!(
            s.execute(&report())
                .await
                .unwrap_err()
                .contains("no changes")
        );

        // Add the regression test, but never ran it red.
        s.execute(&call(
            "write_file",
            json!({ "path": "tests/r.rs", "content": "fn regression_test() {}\n" }),
        ))
        .await
        .unwrap();
        assert!(
            s.execute(&report())
                .await
                .unwrap_err()
                .contains("failing test")
        );

        // Red, then fix, but no green run yet.
        let red = s
            .execute(&call("run_tests", json!({ "filter": "regression_test" })))
            .await
            .unwrap();
        assert!(red.contains("TestsFailed"));
        s.execute(&call(
            "edit_file",
            json!({ "path": "src/lib.rs", "old": "a - b", "new": "a + b" }),
        ))
        .await
        .unwrap();
        assert!(
            s.execute(&report())
                .await
                .unwrap_err()
                .contains("without a filter")
        );

        // Green on the full suite: accepted.
        s.execute(&call("run_tests", json!({}))).await.unwrap();
        assert!(s.execute(&report()).await.unwrap().contains("accepted"));
        assert!(s.verified);
    }

    #[tokio::test]
    async fn an_edit_after_the_green_run_invalidates_it() {
        let sb = FakeSandbox::with_files(&[("src/lib.rs", "x\n")]);
        sb.script_runs(&[TestOutcome::TestsFailed, TestOutcome::Passed]);
        let mut s = Session::new(&sb);
        s.execute(&call(
            "write_file",
            json!({ "path": "t.rs", "content": "regression_test\n" }),
        ))
        .await
        .unwrap();
        s.execute(&call("run_tests", json!({}))).await.unwrap();
        s.execute(&call(
            "edit_file",
            json!({ "path": "src/lib.rs", "old": "x", "new": "y" }),
        ))
        .await
        .unwrap();
        s.execute(&call("run_tests", json!({}))).await.unwrap();
        s.execute(&call(
            "edit_file",
            json!({ "path": "src/lib.rs", "old": "y", "new": "z" }),
        ))
        .await
        .unwrap();
        let e = s
            .execute(&call("write_report", fixed_report(Some("regression_test"))))
            .await
            .unwrap_err();
        assert!(e.contains("without a filter"), "{e}");
        assert!(!s.verified);
    }

    #[tokio::test]
    async fn build_failure_is_not_a_red_run() {
        let sb = FakeSandbox::with_files(&[("src/lib.rs", "x\n")]);
        sb.script_runs(&[TestOutcome::BuildFailed, TestOutcome::Passed]);
        let mut s = Session::new(&sb);
        s.execute(&call(
            "write_file",
            json!({ "path": "t.rs", "content": "regression_test\n" }),
        ))
        .await
        .unwrap();
        let out = s.execute(&call("run_tests", json!({}))).await.unwrap();
        assert!(out.contains("NOT a failing regression test"));
        s.execute(&call(
            "edit_file",
            json!({ "path": "src/lib.rs", "old": "x", "new": "y" }),
        ))
        .await
        .unwrap();
        s.execute(&call("run_tests", json!({}))).await.unwrap();
        let e = s
            .execute(&call("write_report", fixed_report(Some("regression_test"))))
            .await
            .unwrap_err();
        assert!(e.contains("failing test"), "{e}");
    }

    #[tokio::test]
    async fn report_must_name_a_test_that_is_in_the_patch() {
        let sb = FakeSandbox::with_files(&[("src/lib.rs", "x\n")]);
        sb.script_runs(&[TestOutcome::TestsFailed, TestOutcome::Passed]);
        let mut s = Session::new(&sb);
        s.execute(&call(
            "write_file",
            json!({ "path": "t.rs", "content": "fn real_test() {}\n" }),
        ))
        .await
        .unwrap();
        s.execute(&call("run_tests", json!({}))).await.unwrap();
        s.execute(&call(
            "edit_file",
            json!({ "path": "src/lib.rs", "old": "x", "new": "y" }),
        ))
        .await
        .unwrap();
        s.execute(&call("run_tests", json!({}))).await.unwrap();
        assert!(
            s.execute(&call("write_report", fixed_report(None)))
                .await
                .unwrap_err()
                .contains("required")
        );
        assert!(
            s.execute(&call("write_report", fixed_report(Some("made_up"))))
                .await
                .unwrap_err()
                .contains("not part")
        );
        assert!(
            s.execute(&call("write_report", fixed_report(Some("real_test"))))
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn unable_to_fix_needs_no_proof_but_is_never_verified() {
        let sb = FakeSandbox::with_files(&[]);
        let mut s = Session::new(&sb);
        let args = json!({ "status": "unable_to_fix", "root_cause": "?", "explanation": "gave up", "confidence": "low" });
        assert!(s.execute(&call("write_report", args)).await.is_ok());
        assert!(!s.verified);
        assert_eq!(s.report.unwrap().status, Status::UnableToFix);
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        assert_eq!(truncate_head("héllo", 10), "héllo");
        assert!(truncate_head("ééééé", 2).starts_with("éé\n"));
        assert!(truncate_tail("ééééé", 2).ends_with("éé"));
        assert_eq!(truncate_tail("abc", 5), "abc");
    }
}
