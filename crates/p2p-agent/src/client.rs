use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::Config;

/// Client for the Responses API (`POST {base}/responses`).
///
/// Conversation items are kept as raw `serde_json::Value`. That lets us echo the
/// model's output items (including `reasoning` ones) back unchanged on the next turn.
pub struct ResponsesClient {
    http: reqwest::Client,
    cfg: Config,
}

/// A tool call requested by the model. `arguments` is already parsed from the JSON string.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
}

/// The parts of a response we care about.
#[derive(Debug, Deserialize)]
pub struct ModelResponse {
    pub id: String,
    /// Raw output items: `message`, `reasoning`, `function_call`, ...
    pub output: Vec<Value>,
}

impl ModelResponse {
    /// All `function_call` items, in order. There can be several per turn (parallel calls).
    pub fn function_calls(&self) -> Result<Vec<FunctionCall>> {
        self.output
            .iter()
            .filter(|item| item["type"] == "function_call")
            .map(|item| {
                let name = item["name"].as_str().context("function_call without name")?;
                let call_id = item["call_id"].as_str().context("function_call without call_id")?;
                // `arguments` is a JSON *string*, so it needs a second parse.
                let raw = item["arguments"].as_str().unwrap_or("{}");
                let arguments = serde_json::from_str(raw)
                    .with_context(|| format!("bad arguments for tool `{name}`: {raw}"))?;
                Ok(FunctionCall { call_id: call_id.into(), name: name.into(), arguments })
            })
            .collect()
    }

    /// Concatenated assistant text from `message` items.
    pub fn text(&self) -> String {
        self.output
            .iter()
            .filter(|item| item["type"] == "message")
            .flat_map(|item| item["content"].as_array().into_iter().flatten())
            .filter(|part| part["type"] == "output_text")
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Build a `function_call_output` item answering one tool call.
pub fn function_output(call_id: &str, output: &str) -> Value {
    json!({ "type": "function_call_output", "call_id": call_id, "output": output })
}

impl ResponsesClient {
    pub fn new(cfg: Config) -> Result<Self> {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(120)).build()?;
        Ok(Self { http, cfg })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Send the whole conversation so far (`input`) plus tool definitions.
    pub async fn create(&self, input: &[Value], tools: &[Value]) -> Result<ModelResponse> {
        let body = json!({
            "model": self.cfg.model,
            "input": input,
            "tools": tools,
            "parallel_tool_calls": true,
        });
        let url = format!("{}/responses", self.cfg.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("request to {url} failed"))?;

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("model API returned {status}: {text}");
        }
        serde_json::from_str(&text).with_context(|| format!("unexpected response shape: {text}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shape taken from the Meta docs: reasoning item, two parallel calls, then text.
    fn sample() -> ModelResponse {
        serde_json::from_value(json!({
            "id": "resp_1",
            "output": [
                { "type": "reasoning", "summary": [] },
                { "type": "function_call", "call_id": "call_a", "name": "add",
                  "arguments": "{\"a\":17,\"b\":25}", "status": "completed" },
                { "type": "function_call", "call_id": "call_b", "name": "add",
                  "arguments": "{\"a\":3,\"b\":4}", "status": "completed" },
                { "type": "message", "content": [
                    { "type": "output_text", "text": "Hello" },
                    { "type": "output_text", "text": " world" } ] }
            ]
        }))
        .unwrap()
    }

    #[test]
    fn parses_parallel_function_calls() {
        let calls = sample().function_calls().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].call_id, "call_a");
        assert_eq!(calls[1].arguments["a"], 3);
    }

    #[test]
    fn joins_message_text() {
        assert_eq!(sample().text(), "Hello world");
    }

    #[test]
    fn bad_arguments_json_is_an_error() {
        let resp: ModelResponse = serde_json::from_value(json!({
            "id": "r",
            "output": [{ "type": "function_call", "call_id": "c", "name": "add", "arguments": "{oops" }]
        }))
        .unwrap();
        assert!(resp.function_calls().is_err());
    }

    #[test]
    fn function_output_has_matching_call_id() {
        let item = function_output("call_a", "42");
        assert_eq!(item["type"], "function_call_output");
        assert_eq!(item["call_id"], "call_a");
    }
}
