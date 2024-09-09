#![feature(const_trait_impl)]
#![feature(iter_array_chunks)]

/// This is a work-in-progress LLM built from scratch to learn about how transformers work.

use clap::Command;
use lazy_static::lazy_static;
use std::{env, path::PathBuf};

use dotenv::dotenv;

mod downloader;
mod tokenizer;

lazy_static! {
    pub static ref DATA_PATH: PathBuf = PathBuf::from(env::var("DATA_DIRECTORY").expect("DATA_DIRECTORY must be set with a .env file or in the environment"));
}

fn cli() -> Command {
    Command::new("LLMFromScratch")
        .version("0.1.0")
        .about("A work-in-progress LLM built from scratch to learn about how transformers work.")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(downloader::cli())
        .subcommand(tokenizer::generate_vocab_cli())
        .subcommand(tokenizer::tokenize_cli())
        .subcommand(tokenizer::detokenize_cli())
}

#[tokio::main]
async fn main() {
    dotenv().ok();
    
    let app = cli();
    let matches = app.get_matches();

    match matches.subcommand() {
        Some((downloader::DOWNLOAD_SUBCOMMAND, sub_matches)) => downloader::download(*sub_matches.get_one::<usize>("subsetCount").unwrap()).await,
        Some((tokenizer::GENERATE_VOCAB_SUBCOMMAND, sub_matches)) => _ = tokenizer::Tokenizer::build_token_map(
            *sub_matches.get_one::<bool>("continue").unwrap(),
            *sub_matches.get_one::<f64>("data_ratio").unwrap(),
        ).await,
        Some((tokenizer::TOKENIZE_SUBCOMMAND, sub_matches)) => _ = tokenizer::tokenize(sub_matches.get_one::<String>("text").unwrap().clone()).await,
        Some((tokenizer::DETOKENIZE_SUBCOMMAND, sub_matches)) => _ = tokenizer::detokenize(sub_matches.get_one::<String>("tokens").unwrap().clone()).await,
        _ => unreachable!(),
    }
}
