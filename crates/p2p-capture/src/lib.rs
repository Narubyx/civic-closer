//! Panic capture for Panic-to-Patch: structured incidents with secrets redacted.

pub mod hook;
pub mod incident;
pub mod redact;

pub use hook::{install, set_context};
pub use incident::{Incident, Location, short_hash};
pub use redact::{redact, redact_value};
