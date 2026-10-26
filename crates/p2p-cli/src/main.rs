use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use p2p_agent::sandbox::check_patch_applies;
use p2p_agent::{AgentEvent, Config, LocalSandbox, ResponsesClient, report, run_agent, smoke};
use p2p_capture::Incident;

#[derive(Parser)]
#[command(name = "p2p", about = "Panic-to-Patch: crash in, verified patch out")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Call the model with one trivial tool and print the tool-call round trip.
    Smoke,
    /// Fix a captured panic: reproduce it with a test, patch it, verify, and write a report.
    Fix {
        /// Incident JSON saved by the panic hook.
        #[arg(long)]
        incident: PathBuf,
        /// The project (folder with Cargo.toml). It is only copied, never modified.
        #[arg(long)]
        project: PathBuf,
        /// Where `<fingerprint>/report.md` and `fix.patch` are written.
        #[arg(long, default_value = ".p2p/reports")]
        out: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Smoke => {
            let client = ResponsesClient::new(Config::from_env()?)?;
            smoke::run(&client).await
        }
        Command::Fix {
            incident,
            project,
            out,
        } => fix(incident, project, out).await,
    }
}

async fn fix(incident_path: PathBuf, project: PathBuf, out: PathBuf) -> Result<()> {
    let json = std::fs::read_to_string(&incident_path)
        .with_context(|| format!("cannot read {}", incident_path.display()))?;
    let incident: Incident =
        serde_json::from_str(&json).context("not a valid incident JSON file")?;

    let client = ResponsesClient::new(Config::from_env()?)?;
    let sandbox = LocalSandbox::create(&project)?;
    println!(
        "incident {} ({}) - model {} - sandbox copy of {}",
        incident.fingerprint,
        incident.message,
        client.config().model,
        project.display()
    );

    let outcome = run_agent(&client, &sandbox, &incident, &mut print_event).await?;

    let dir = report::save(&out, &incident, &outcome)?;
    println!("\n== done in {} turn(s) ==", outcome.turns);
    match (&outcome.report, outcome.verified) {
        (Some(_), true) => println!("verified fix (failing test -> passing suite)"),
        (Some(_), false) => println!("the agent did NOT produce a verified fix; see the report"),
        (None, _) => println!("the agent stopped without a report"),
    }
    if outcome.verified {
        match check_patch_applies(&project, &outcome.patch) {
            Ok(()) => println!("fix.patch applies cleanly to your project (not applied)"),
            Err(e) => println!("warning: {e}"),
        }
    }
    println!("report: {}", dir.join("report.md").display());
    Ok(())
}

fn print_event(event: AgentEvent) {
    match event {
        AgentEvent::Turn(n) => println!("\n-- turn {n} --"),
        AgentEvent::Text(text) => println!("model: {}", text.trim()),
        AgentEvent::ToolCall { name, summary } => println!("  > {name} {summary}"),
        AgentEvent::ToolResult { ok, summary, .. } => {
            println!("    {} {summary}", if ok { "ok" } else { "ERR" })
        }
    }
}
