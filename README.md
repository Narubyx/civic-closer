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

Fix a captured panic (uses a small synthetic project with a seeded bug; your folder is only copied, never modified):

```powershell
cargo run -p p2p-cli -- fix --incident fixtures/inventory-bug.incident.json --project fixtures/inventory-bug
```

The result is `.p2p/reports/<fingerprint>/report.md` and `fix.patch`. A fix is only labelled *verified* if the harness itself saw a failing regression test, then a passing full test suite on the final code.

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
| 2026-10-26 | Day 1: the agent loop. Muse Spark gets tools (`list_files`, `read_file`, `write_file`, `edit_file`, `run_tests`, `write_report`) and works inside the sandbox; the loop is generic over a `Model` trait so it is tested offline with a scripted model. Source is redacted before the model sees it. The harness records every test run and rejects a `fixed` report unless it saw a failing regression test and then a passing full suite on the final code. Markdown incident report + `fix.patch` output, `p2p fix` command, and a synthetic fixture project with a seeded bug. |

## License

MIT - see [LICENSE](LICENSE).
