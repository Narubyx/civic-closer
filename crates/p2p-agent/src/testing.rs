//! Test doubles: an in-memory sandbox with scripted test results, and a scripted model.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::client::{Model, ModelResponse};
use crate::sandbox::{Sandbox, TestOptions, TestOutcome, TestRun};

/// Files live in a map; `run_tests` returns the outcomes queued with [`script_runs`](Self::script_runs).
pub struct FakeSandbox {
    files: Mutex<BTreeMap<String, String>>,
    originals: Mutex<BTreeMap<String, Option<String>>>,
    runs: Mutex<VecDeque<TestOutcome>>,
}

impl FakeSandbox {
    pub fn with_files(files: &[(&str, &str)]) -> Self {
        Self {
            files: Mutex::new(
                files
                    .iter()
                    .map(|(p, c)| (p.to_string(), c.to_string()))
                    .collect(),
            ),
            originals: Mutex::default(),
            runs: Mutex::default(),
        }
    }

    pub fn script_runs(&self, outcomes: &[TestOutcome]) {
        self.runs.lock().unwrap().extend(outcomes.iter().copied());
    }
}

impl Sandbox for FakeSandbox {
    fn read_file(&self, path: &str) -> Result<String> {
        self.files
            .lock()
            .unwrap()
            .get(path)
            .cloned()
            .with_context(|| format!("no such file `{path}`"))
    }

    fn write_file(&self, path: &str, content: &str) -> Result<()> {
        let mut files = self.files.lock().unwrap();
        self.originals
            .lock()
            .unwrap()
            .entry(path.into())
            .or_insert_with(|| files.get(path).cloned());
        files.insert(path.into(), content.into());
        Ok(())
    }

    fn list_files(&self, dir: &str) -> Result<Vec<String>> {
        Ok(self
            .files
            .lock()
            .unwrap()
            .keys()
            .filter(|p| p.starts_with(dir))
            .cloned()
            .collect())
    }

    /// Not a real diff: every changed file's new content as `+` lines. Enough for the tests.
    fn diff(&self) -> Result<String> {
        let files = self.files.lock().unwrap();
        let mut out = String::new();
        for (path, original) in self.originals.lock().unwrap().iter() {
            let now = &files[path];
            if original.as_ref() != Some(now) {
                out.push_str(&format!("diff --git a/{path} b/{path}\n"));
                out.extend(now.lines().map(|l| format!("+{l}\n")));
            }
        }
        Ok(out)
    }

    async fn run_tests(&self, _opts: &TestOptions) -> Result<TestRun> {
        let outcome = self
            .runs
            .lock()
            .unwrap()
            .pop_front()
            .context("no scripted test run left")?;
        Ok(TestRun {
            outcome,
            exit_code: Some(if outcome == TestOutcome::Passed {
                0
            } else {
                101
            }),
            output: format!("fake output for {outcome:?}"),
            duration: Duration::from_millis(5),
        })
    }
}

/// Replays canned responses in order.
pub struct ScriptedModel {
    responses: Mutex<VecDeque<ModelResponse>>,
    /// How many `input` items each request carried (to check the conversation grows).
    pub input_sizes: Mutex<Vec<usize>>,
}

impl ScriptedModel {
    pub fn new(responses: Vec<Value>) -> Self {
        Self {
            responses: Mutex::new(
                responses
                    .into_iter()
                    .map(|r| serde_json::from_value(r).unwrap())
                    .collect(),
            ),
            input_sizes: Mutex::default(),
        }
    }
}

/// A response in which the model calls the given tools (all in the same turn).
pub fn calls(id: &str, tool_calls: &[(&str, Value)]) -> Value {
    let output: Vec<Value> = tool_calls
        .iter()
        .enumerate()
        .map(|(i, (name, args))| {
            json!({ "type": "function_call", "call_id": format!("{id}_{i}"), "name": name,
                    "arguments": args.to_string() })
        })
        .collect();
    json!({ "id": id, "output": output })
}

/// A response with only text (no tool calls).
pub fn text(id: &str, text: &str) -> Value {
    json!({ "id": id, "output": [{ "type": "message",
        "content": [{ "type": "output_text", "text": text }] }] })
}

impl Model for ScriptedModel {
    async fn create(&self, input: &[Value], _tools: &[Value]) -> Result<ModelResponse> {
        self.input_sizes.lock().unwrap().push(input.len());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .context("model script exhausted")
    }
}
