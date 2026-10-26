use clap::{Parser, Subcommand};
use p2p_agent::{Config, ResponsesClient, smoke};

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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Smoke => {
            let client = ResponsesClient::new(Config::from_env()?)?;
            smoke::run(&client).await
        }
    }
}
