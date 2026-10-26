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

Try the model round trip (PowerShell shown; bash uses `export`):

```powershell
$env:MODEL_API_KEY = "<your key>"
cargo run -p p2p-cli -- smoke
```

## Workspace layout

- `crates/p2p-capture` - panic hook, incident JSON, redaction
- `crates/p2p-agent` - model client and tool-calling loop
- `crates/p2p-cli` - command-line entry point

## What I built during the hackathon

Everything in this repository was written during the event (Oct 26 - Nov 4, 2026), starting from an empty repo. No prior code was imported. All example data is synthetic.

| Date       | Log |
|------------|-----|
| 2026-10-26 | Day 1: Cargo workspace skeleton (`p2p-capture`, `p2p-agent`, `p2p-cli`), MIT license, `.gitignore`, README. |
| 2026-10-26 | Day 1: `p2p-agent` Responses API client (`POST /v1/responses`) with flat tool schemas, parallel `function_call` handling and a `p2p smoke` command that runs a one-tool round trip against Muse Spark. Verified live: `muse-spark-1.3` issues parallel tool calls and accepts its own output items echoed back. |
| 2026-10-26 | Day 1: `p2p-capture` panic hook that saves a structured incident as JSON (message, location, backtrace, context, SHA-256 fingerprint). Redaction (keys, tokens, JWTs, private keys, emails, passwords, home-directory user names) runs before anything is saved; unit and end-to-end tests included. |
| 2026-10-26 | Day 1: `.gitattributes` (LF everywhere). Local sandbox in `p2p-agent`: a temp-directory copy of the project (no `.env`, keys, `target/` or `.git`), path-escape guard, exact search/replace edits, `cargo test` with a scrubbed environment (no `MODEL_API_KEY`), a timeout that kills the process tree, a red/green-aware outcome (`TestsFailed` vs `BuildFailed`), a git-style `.patch` generated from the sandbox and checked with `git apply --check` against the untouched original. |

## License

MIT - see [LICENSE](LICENSE).
