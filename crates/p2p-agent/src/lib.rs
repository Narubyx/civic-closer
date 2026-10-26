//! Model client and tool-calling loop for Panic-to-Patch.

pub mod client;
pub mod config;
pub mod smoke;

pub use client::{FunctionCall, ModelResponse, ResponsesClient};
pub use config::Config;
