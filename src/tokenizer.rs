/// This is a basic byte-pair encoding tokenizer. It is used to tokenize the input text into subwords.

use std::{fs, path::PathBuf};
use clap::Command;
use lazy_static::lazy_static;
use crate::DATA_PATH;

pub const GENERATE_VOCAB_SUBCOMMAND: &str = "build_token_map";
pub fn cli() -> Command {
    Command::new(GENERATE_VOCAB_SUBCOMMAND)
        .about("Build the token map from the vocabulary file.")
}

const UNKNOWN_TOKEN: &str = "<|unk|>";
const ENDOFTEXT_TOKEN: &str = "<|endoftext|>";
const VOCAB_SIZE: usize = 30_000;

lazy_static! {
    pub static ref VOCAB_FILE: PathBuf = DATA_PATH.join("vocab.txt");
}

pub fn build_token_map() {
}

pub struct Tokenizer {
    vocab: Vec<String>,
}
impl Tokenizer {
    pub fn new() -> Self {
        if !VOCAB_FILE.exists() {
            panic!("Vocabulary file not found; cannot run tokenizer. Please run `download` and `{}` first.", GENERATE_VOCAB_SUBCOMMAND); // TODO: Proper error handling
        }

        let vocab = fs::read_to_string(VOCAB_FILE.clone()).unwrap().lines().map(|line| line.to_string()).collect();
        Self { vocab }
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = vec![];
        for token in text.split_whitespace() {
            let mut token = token.to_string();
            if token == " " {
                continue;
            }
            if !self.vocab.contains(&token) {
                token = UNKNOWN_TOKEN.to_string();
            }
            tokens.push(token);
        }
        tokens
    }

    pub fn detokenize(&self, tokens: &[String]) -> String {
        let mut text = String::new();
        for token in tokens {
            if text.is_empty() {
                text.push_str(token);
            } else {
                let first_char = token.chars().next().unwrap();
                if first_char.is_alphanumeric() {
                    text.push(' ');
                }
                text.push_str(token);
            }
        }
        text
    }
}