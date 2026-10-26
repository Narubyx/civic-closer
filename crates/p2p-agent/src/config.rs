use anyhow::{Context, Result};

pub const DEFAULT_MODEL: &str = "muse-spark-1.3";
pub const DEFAULT_BASE: &str = "https://api.meta.ai/v1";

/// Connection settings, read from the environment so no secret ever lives in the repo.
#[derive(Clone)]
pub struct Config {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("MODEL_API_KEY")
            .context("MODEL_API_KEY is not set (PowerShell: $env:MODEL_API_KEY = \"...\")")?;
        Ok(Self {
            api_key,
            model: std::env::var("MUSE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into()),
            base_url: std::env::var("META_API_BASE").unwrap_or_else(|_| DEFAULT_BASE.into()),
        })
    }
}

// Hand-written so `{:?}` can never print the API key into a log.
impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}
