# Panic-to-Patch

Panic-to-Patch is a Rust crate and agent that catches crashes, reproduces the bug with a new regression test, writes a patch, and verifies it before filing an incident report.

> Entry for the **Meta Global AI Developer Hackathon** (Indie track, theme "Push the Frontier").

## How it works

1. **Capture** - a panic hook (plus `tracing`) saves a structured incident with a backtrace.
2. **Redact** - secrets (keys, tokens, emails) are scrubbed *before* anything leaves the machine.
3. **Dedupe** - incidents are grouped by error fingerprint.
4. **Fix loop** - a Meta Muse Spark agent reads the code, forms a root-cause hypothesis, writes a failing regression test, proposes a patch, and verifies it with `cargo test` in a sandbox *copy* of the project. Your working tree is never touched; the output is a `.patch` file.
5. **Report** - a Markdown incident report with log, error, root cause, patch, regression test, test result and confidence.

## Configuration

| Variable        | Purpose                               | Default                   |
|-----------------|---------------------------------------|---------------------------|
| `MODEL_API_KEY` | API key (never commit it)             | required                  |
| `MUSE_MODEL`    | Model name                            | `muse-spark-1.3`          |
| `META_API_BASE` | Base URL of the OpenAI-compatible API | `https://api.meta.ai/v1`  |

PowerShell: `$env:MODEL_API_KEY = "..."` - bash: `export MODEL_API_KEY=...`

## Workspace layout

- `crates/p2p-capture` - panic hook, incident JSON, redaction
- `crates/p2p-agent` - model client and tool-calling loop
- `crates/p2p-cli` - command-line entry point

## What I built during the hackathon

Everything in this repository was written during the event (Oct 26 - Nov 4, 2026), starting from an empty repo. No prior code was imported. All example data is synthetic.

| Date       | Log |
|------------|-----|
| 2026-10-26 | Day 1: Cargo workspace skeleton (`p2p-capture`, `p2p-agent`, `p2p-cli`), MIT license, `.gitignore`, README. |

## License

MIT - see [LICENSE](LICENSE).
