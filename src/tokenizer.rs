/// This is a basic byte-pair encoding tokenizer. It is used to tokenize the input text into subwords.

use std::{fs, path::PathBuf};
use clap::Command;
use indicatif::ProgressBar;
use lazy_static::lazy_static;
use hashbrown::HashMap;
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

pub struct Tokenizer {
    merges: HashMap<(usize, usize), usize>,

    token_map: HashMap<String, usize>,
    reverse_token_map: HashMap<usize, String>,
}

impl Tokenizer {
    pub fn load() -> Self {
        if !VOCAB_FILE.exists() || !MERGES_FILE.exists() {
            panic!("Vocabulary file not found; cannot run tokenizer. Please run `{}` and `{}` first.", DOWNLOAD_SUBCOMMAND, GENERATE_VOCAB_SUBCOMMAND); // TODO: Proper error handling
        }

        let tokenizer = Tokenizer {
            merges: HashMap::new(),
            token_map: HashMap::new(),
            reverse_token_map: HashMap::new(),
        };
        // TODO: Load with bincode
        tokenizer
    }

    fn add_token(&mut self, token: String) {
        let value = self.token_map.len();
        self.token_map.insert(token.clone(), value);
        self.reverse_token_map.insert(value, token);
    }
    
    pub async fn build_token_map() -> Self {
        let subsets_folder = DATA_PATH.join("data/extracted");
        let subsets_available = fs::read_dir(subsets_folder.clone()).unwrap().count();
        if subsets_available == 0 {
            panic!("No subsets found in the data folder. Please run the `download` command first.");
        }

        let mut tokenizer = Tokenizer {
            merges: HashMap::new(),
            token_map: HashMap::new(),
            reverse_token_map: HashMap::new(),
        };

        tokenizer.add_token(UNKNOWN_TOKEN.to_string());
        tokenizer.add_token(ENDOFTEXT_TOKEN.to_string());

        println!("Running preliminary pass to create individual character tokens...");

        // First, we run a preliminary pass to create individual character tokens.
        for subset in fs::read_dir(subsets_folder.clone()).unwrap().enumerate() {
            // TEMPORARY: only use the first subset for testing purposes
            if subset.0 > 0 {
                break;
            }

            let subset = subset.1.unwrap();
            let progress = ProgressBar::new(fs::read_dir(subset.path()).unwrap().count() as u64)
                .with_prefix("Processing subset")
                .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len} | {msg}").unwrap());

            let files = fs::read_dir(subset.path()).unwrap();
            for file in files {
                let file = file.unwrap();
                let path = file.path();
                let text: String = fs::read_to_string(path).unwrap();

                for c in text.chars() {
                    let token = c.to_string();
                    if !tokenizer.token_map.contains_key(&token) {
                        tokenizer.add_token(token);
                    }
                }
                
                progress.inc(1);
            }

            progress.finish();
        }

        println!("Preprocessing complete. Found {} single-character tokens.", tokenizer.token_map.len());

        // Next, we run the BPE algorithm to create subword tokens until we reach the desired vocabulary size.
        while tokenizer.token_map.len() < VOCAB_SIZE {
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
                    let tokens = tokenizer.tokenize(&text);
                    
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
        
        println!("Tokenizer built with {} tokens.", tokenizer.token_map.len());

        tokenizer
    }

    /// Converts a text into a list of token IDs.
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![];

        // Run a preliminary pass to tokenize individual characters
        for c in text.chars() {
            let token = c.to_string();
            let token_id = self.token_map.get(&token).unwrap_or_else(|| self.token_map.get(UNKNOWN_TOKEN).unwrap());
            tokens.push(*token_id);
        }

        // Merge the tokens into subwords
        let mut i = 0;
        while i < tokens.len() {
            let j = tokens.len();
            while i < j {
                let mut found = false;
                for k in (i + 1..j).rev() {
                    let pair = (tokens[i], tokens[k]);
                    if let Some(merged) = self.merges.get(&pair) {
                        tokens[i] = *merged;
                        tokens.remove(k);
                        found = true;
                        break;
                    }
                }

                if !found {
                    i += 1;
                }
            }
        }

        tokens
    }

    /// Converts a list of token IDs back into text.
    pub fn detokenize(&self, tokens: &[usize]) -> String {
        let mut text = String::new();

        for token in tokens {
            let token = self.reverse_token_map.get(token).unwrap();
            text.push_str(token);
        }

        text
    }
}