/// This is a basic byte-pair encoding tokenizer. It is used to tokenize the input text into subwords.

use std::{fs, path::PathBuf};
use clap::Command;
use lazy_static::lazy_static;
use hashbrown::{HashMap, HashSet};
use crate::{downloader::DOWNLOAD_SUBCOMMAND, DATA_PATH};

pub const GENERATE_VOCAB_SUBCOMMAND: &str = "build_token_map";
pub fn cli() -> Command {
    Command::new(GENERATE_VOCAB_SUBCOMMAND)
        .about("Build the token map for the tokenizer from the training data.")
}

const UNKNOWN_TOKEN: &str = "<|unk|>";
const ENDOFTEXT_TOKEN: &str = "<|endoftext|>";
const VOCAB_SIZE: usize = 30_000;

lazy_static! {
    pub static ref VOCAB_FILE: PathBuf = DATA_PATH.join("vocab.txt");
    pub static ref MERGES_FILE: PathBuf = DATA_PATH.join("merges.txt");
}

pub async fn build_token_map() {
    let subsets_folder = DATA_PATH.join("data/extracted");
    let subsets_available = fs::read_dir(subsets_folder.clone()).unwrap();
    if subsets_available.count() == 0 {
        panic!("No subsets found in the data folder. Please run the `download` command first.");
    }

    let mut visited_chars = HashSet::<char>::new();
    let mut token_map = HashMap::<String, usize>::new();
    token_map.insert(UNKNOWN_TOKEN.to_string(), 0);
    token_map.insert(ENDOFTEXT_TOKEN.to_string(), 1);

    while token_map.len() < VOCAB_SIZE {
        // We need to read the dataset and count the frequency of each token.
        // We will then merge the least frequent tokens into a single token until we reach the desired vocabulary size.
        // We will also keep track of the token pairs that we merge, as they will be useful for detokenization.
        // We will save the token map and the merges to disk.

        let mut frequencies = HashMap::<(usize, usize), usize>::new();
        
        // We run the process one file at a time since it's unreasonable to load the 50GB dataset into memory all at once on most machines.
        let subsets_available = fs::read_dir(subsets_folder.clone()).unwrap();
        for subset in subsets_available.enumerate() {
            // TEMPORARY: only use the first subset for testing purposes
            if subset.0 > 0 {
                break;
            }

            let (index, subset) = subset;
            println!("Processing subset {}...", index);

            let subset = subset.unwrap();
            let files = fs::read_dir(subset.path()).unwrap();

            for file in files.enumerate() {
                let (index, file) = file;
                if index % 100 == 0 {
                    println!("Processing file {}...", index);
                }
                if index > 10_000 {
                    break; // TEMPORARY: only process the first 10,000 files for testing purposes
                }

                let file = file.unwrap();
                let path = file.path();
                let text: String = fs::read_to_string(path).unwrap();

                // Tokenize the text
                let mut tokens = vec![];
                for char in text.chars() {
                    // For now, just use the character as a token
                    tokens.push(char as usize);
                }
                
                // Find all adjacent token pairs and increment their frequency
                for i in 0..tokens.len() - 1 {
                    let pair = (tokens[i], tokens[i + 1]);
                    *frequencies.entry(pair).or_insert(0) += 1;
                }
            }
        }

        // Print the frequencies, sorted from low to high
        let mut frequencies: Vec<_> = frequencies.into_iter().collect();
        frequencies.sort_by_key(|(_, freq)| *freq);
        for (pair, freq) in frequencies.iter() {
            println!("{:?}: {}", pair, freq);
        }

        // For now, stop looping
        break;
    }
}

pub struct Tokenizer {
    vocab: Vec<String>,
}
impl Tokenizer {
    pub fn new() -> Self {
        if !VOCAB_FILE.exists() {
            panic!("Vocabulary file not found; cannot run tokenizer. Please run `{}` and `{}` first.", DOWNLOAD_SUBCOMMAND, GENERATE_VOCAB_SUBCOMMAND); // TODO: Proper error handling
        }

        let vocab = fs::read_to_string(VOCAB_FILE.clone()).unwrap().lines().map(|line| line.to_string()).collect();
        Self { vocab }
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![];
        tokens
    }

    pub fn detokenize(&self, tokens: &[usize]) -> String {
        let mut text = String::new();
        text
    }
}