//! Day-1 smoke test: one trivial tool, full tool-call round trip, printed to stdout.

use anyhow::{Result, bail};
use serde_json::{Value, json};

use crate::client::{FunctionCall, ResponsesClient, function_output};

const MAX_TURNS: usize = 5;

/// Tool definitions use the Responses API's *flat* shape (no nested `function` key).
fn tools() -> Vec<Value> {
    vec![json!({
        "type": "function",
        "name": "add",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a", "b"]
        }
    })]
}

/// Run one tool. Errors are returned as text so the model can see and react to them.
fn run_tool(call: &FunctionCall) -> String {
    match call.name.as_str() {
        "add" => match (call.arguments["a"].as_i64(), call.arguments["b"].as_i64()) {
            (Some(a), Some(b)) => (a + b).to_string(),
            _ => "error: `a` and `b` must be integers".into(),
        },
        other => format!("error: unknown tool `{other}`"),
    }
}

pub async fn run(client: &ResponsesClient) -> Result<()> {
    println!(
        "model: {}  base: {}",
        client.config().model,
        client.config().base_url
    );

    // Two independent sums nudge the model into parallel tool calls.
    let mut input = vec![json!({
        "role": "user",
        "content": "Use the add tool to compute 17+25 and 3+4, then tell me both results."
    })];

    for turn in 1..=MAX_TURNS {
        let resp = client.create(&input, &tools()).await?;
        let calls = resp.function_calls()?;
        println!("\n--- turn {turn} (response {}) ---", resp.id);

        if calls.is_empty() {
            println!("assistant: {}", resp.text());
            return Ok(());
        }

        println!("model requested {} tool call(s) in one turn", calls.len());
        // Echo the model's output items back, then append one result per call.
        input.extend(resp.output.iter().cloned());
        for call in &calls {
            let result = run_tool(call);
            println!(
                "  {}({}) -> {}  [call_id {}]",
                call.name, call.arguments, result, call.call_id
            );
            input.push(function_output(&call.call_id, &result));
        }
    }
    bail!("model still calling tools after {MAX_TURNS} turns")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, arguments: Value) -> FunctionCall {
        FunctionCall {
            call_id: "c".into(),
            name: name.into(),
            arguments,
        }
    }

    #[test]
    fn add_tool_sums() {
        assert_eq!(run_tool(&call("add", json!({ "a": 17, "b": 25 }))), "42");
    }

    #[test]
    fn add_tool_rejects_bad_args() {
        assert!(run_tool(&call("add", json!({ "a": "x" }))).starts_with("error"));
    }

    #[test]
    fn unknown_tool_is_reported() {
        assert!(run_tool(&call("nope", json!({}))).contains("unknown tool"));
    }
}
