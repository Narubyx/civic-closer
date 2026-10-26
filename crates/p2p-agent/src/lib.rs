//! Model client and tool-calling loop for Panic-to-Patch.

pub mod agent;
pub mod client;
pub mod config;
pub mod report;
pub mod sandbox;
pub mod smoke;
pub mod tools;

#[cfg(test)]
mod testing;

pub use agent::{AgentEvent, AgentOutcome, run_agent};
pub use client::{FunctionCall, Model, ModelResponse, ResponsesClient};
pub use config::Config;
pub use sandbox::{LocalSandbox, Sandbox, TestOptions, TestOutcome, TestRun};
