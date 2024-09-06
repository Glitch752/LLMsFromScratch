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
const VOCAB_SIZE: usize = 50_000;

lazy_static! {
    pub static ref VOCAB_FILE: PathBuf = DATA_PATH.join("vocab.txt");
    pub static ref MERGES_FILE: PathBuf = DATA_PATH.join("merges.txt");
}

#[derive(Clone, Debug)]
pub struct Tokenizer {
    /// Map from pairs of tokens to the BPE rank. Lower ranks indicate more frequent pairs.
    merges: HashMap<(String, String), usize>,

    /// Map from token string to token ID.
    token_map: HashMap<String, usize>,
    /// Map from token ID to token string.
    reverse_token_map: HashMap<usize, String>,

    /// A cache for the BPE algorithm. This is used to speed up tokenization.
    bpe_cache: HashMap<String, Vec<String>>,
}

fn min_by_key<T, K: Ord>(iter: impl IntoIterator<Item = T>, key: impl Fn(&T) -> K) -> Option<T> {
    let mut iter = iter.into_iter();
    let mut min = iter.next()?;
    let mut min_key = key(&min);

    for item in iter {
        let item_key = key(&item);
        if item_key < min_key {
            min = item;
            min_key = item_key;
        }
    }

    Some(min)
}

impl Tokenizer {
    fn new() -> Self {
        Tokenizer {
            merges: HashMap::new(),
            token_map: HashMap::new(),
            reverse_token_map: HashMap::new(),
            bpe_cache: HashMap::new(),
        }
    }

    pub fn load() -> Self {
        if !VOCAB_FILE.exists() || !MERGES_FILE.exists() {
            panic!("Vocabulary file not found; cannot run tokenizer. Please run `{}` and `{}` first.", DOWNLOAD_SUBCOMMAND, GENERATE_VOCAB_SUBCOMMAND); // TODO: Proper error handling
        }

        let tokenizer = Tokenizer::new();
        // TODO: Load with bincode
        tokenizer
    }

    fn add_token(&mut self, token: String) {
        let value = self.token_map.len();
        self.token_map.insert(token.clone(), value);
        self.reverse_token_map.insert(value, token);
        self.bpe_cache.clear(); // Clear the cache since the token map has changed
    }
    
    pub async fn build_token_map() -> Self {
        let subsets_folder = DATA_PATH.join("data/extracted");
        let subsets_available = fs::read_dir(subsets_folder.clone()).unwrap().count();
        if subsets_available == 0 {
            panic!("No subsets found in the data folder. Please run the `download` command first.");
        }

        let mut tokenizer = Tokenizer::new();
        tokenizer.add_token(UNKNOWN_TOKEN.to_string());
        tokenizer.add_token(ENDOFTEXT_TOKEN.to_string());

        // Add every possible byte as a token
        for i in 0..=255 {
            let c = char::from_u32(i).unwrap();
            tokenizer.add_token(c.to_string());
        }

        // This is mostly temporary since it doesn't scale to large datasets. However, I think it's okay if we
        // only build the tokenizer based on the data that can fit in memory.
        let subset_count = 1; // For now, we just analyze the first subset
        println!("Loading {} subsets into memory...", subset_count);
        let mut inputs: Vec<String> = vec![];
        let subsets = fs::read_dir(subsets_folder.clone())
            .unwrap()
            .take(subset_count)
            .map(|subset| subset.unwrap());
        for subset in subsets {
            let i = 0;
            let progress = ProgressBar::new(fs::read_dir(subset.path()).unwrap().count() as u64)
                .with_prefix("Loading subset")
                .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len} | {msg}").unwrap());

            let files = fs::read_dir(subset.path()).unwrap();
            for file in files {
                let file = file.unwrap();
                let path = file.path();
                let contents = fs::read_to_string(path).unwrap();
                inputs.push(contents);

                progress.inc(1);
            }

            progress.finish();
        }
        println!("Loaded {} subsets into memory.", subset_count);

        // Next, we run the BPE algorithm to create subword tokens until we reach the desired vocabulary size.
        while tokenizer.token_map.len() < VOCAB_SIZE {
            let mut frequencies = HashMap::<(usize, usize), usize>::new();
            
            // Count the frequencies of all pairs of tokens
            let progress = ProgressBar::new(inputs.len() as u64)
                .with_prefix("Counting token pairs")
                .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len} | {msg}").unwrap());
            for input in inputs {
                if tokenizer.bpe_cache.len() > 50_000 {
                    tokenizer.bpe_cache.clear(); // Clear the cache periodically so it doesn't get ridiculously large
                }

                let tokens = tokenizer.tokenize(&input);
                for i in 0..tokens.len() - 1 {
                    *frequencies.entry((tokens[i], tokens[i + 1])).or_insert(0) += 1;
                }
                progress.inc(1);
            }
            progress.finish();

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

    /// Gets all possible pairs of tokens in a string.
    fn get_pairs(&self, values: &Vec<String>) -> Vec<(String, String)> {
        let mut pairs = vec![];
        for i in 0..values.len() - 1 {
            pairs.push((values[i].clone(), values[i + 1].clone()));
        }

        pairs
    }

    /// Applies the Byte-Pair Encoding algorithm to a string.
    fn bpe(&mut self, string: &str) -> Vec<String> {
        if self.bpe_cache.contains_key(string) {
            return self.bpe_cache.get(string).unwrap().to_vec();
        }

        let mut tokens: Vec<String> = string.bytes().map(|b| char::from_u32(b as u32).unwrap().to_string()).collect();
        let mut pairs: Vec<(String, String)> = self.get_pairs(&tokens);

        if pairs.is_empty() {
            return vec![string.to_string()];
        }

        loop {
            // First, we find the most important pair to merge (meaning the pair that occurs most frequently)
            let most_important_pair = min_by_key(pairs, |pair| self.merges.get(pair).unwrap_or(&usize::MAX));
            if most_important_pair.is_none() { // If we didn't find a pair to merge, we're done
                break;
            }
            let (first, second) = most_important_pair.unwrap();

            // Next, we merge the pair
            let mut new_tokens = vec![];
            let mut i = 0;
            while i < tokens.len() {
                let j = tokens.iter().skip(i).position(|token| *token == first);
                if let Some(j) = j {
                    new_tokens.extend(tokens[i..i + j].iter().cloned());
                    i += j;
                }

                if i < tokens.len() - 1 && tokens[i + 1] == second {
                    new_tokens.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }

            tokens = new_tokens;

            // If the new string is only one character long, we're done
            if tokens.len() == 1 {
                break;
            }
            
            // Otherwise, we update the pairs and continue
            pairs = self.get_pairs(&tokens);
        }

        // Finally, we update the cache and return the tokens
        self.bpe_cache.insert(string.to_string(), tokens.clone());

        tokens
    }

    /// Converts a text into a list of token IDs.
    pub fn tokenize(&mut self, text: &str) -> Vec<usize> {
        let preliminary_tokens = split_into_preliminary_tokens(text);

        let mut bytepair_tokens: Vec<usize> = vec![];
        for preliminary_token in preliminary_tokens {
            bytepair_tokens.extend(self.bpe(preliminary_token).iter().map(|token| *self.token_map.get(token).unwrap_or(self.token_map.get(UNKNOWN_TOKEN).unwrap())));
        }

        bytepair_tokens
    }

    /// Converts a list of token IDs back into text.
    pub fn detokenize(&self, tokens: &[usize]) -> String {
        let mut text = String::new();

        let unknown = UNKNOWN_TOKEN.to_string();
        for token in tokens {
            let token = self.reverse_token_map.get(token).unwrap_or(&unknown);
            text.push_str(token);
        }

        text
    }
}

// This effectively finds matches for the regex "'\p{L}+| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
/// adapted from the GPT-2 tokenizer: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
/// This is used to split text into "preliminary tokens" before applying the BPE algorithm.
/// We do this so we can manually control token boundaries. For example, generated text is much more accurate if we
/// tokenize punctuation separately, always split at word boundaries, etc.
fn split_into_preliminary_tokens(text: &str) -> Vec<&str> {
    let mut tokens = vec![];

    let mut i = 0;
    while i < text.len() {
        // "'\p{L}+| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        // Highest priority match: contractions (an apostrophe followed by at least one letter)
        if text[i..].starts_with('\'') {
            let mut j = i + 1;
            while j < text.len() && text[j..].starts_with(char::is_alphabetic) {
                j += text[j..].chars().next().unwrap().len_utf8();
            }
            if j > i + 1 {
                tokens.push(&text[i..j]);
                i = j;
                continue;
            }

            // If we didn't find a contraction, fall through to the next match
        }

        // " ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        // Next highest priority match: a possible space followed by at least one letter
        let has_space = text[i..].starts_with(' ');
        let mut j = i + has_space as usize;
        while j < text.len() && text[j..].starts_with(char::is_alphabetic) {
            j += text[j..].chars().next().unwrap().len_utf8();
        }
        if j > i + has_space as usize {
            tokens.push(&text[i..j]);
            i = j;
            continue;
        }

        // " ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        // Next highest priority match: a possible space followed by at least one number
        let mut j = i + has_space as usize;
        while j < text.len() && text[j..].starts_with(char::is_numeric) {
            j += text[j..].chars().next().unwrap().len_utf8();
        }
        if j > i + has_space as usize {
            tokens.push(&text[i..j]);
            i = j;
            continue;
        }

        // " ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        // Next highest priority match: a possible space followed by at least one non-alphanumeric and non-whitespace character
        let mut j = i + has_space as usize;
        while j < text.len() && !text[j..].starts_with(char::is_alphanumeric) && !text[j..].starts_with(char::is_whitespace) {
            j += text[j..].chars().next().unwrap().len_utf8();
        }
        if j > i + has_space as usize {
            tokens.push(&text[i..j]);
            i = j;
            continue;
        }

        // "\s+(?!\S)|\s+"
        // And now the reason we're not using a simple regex: one or more whitespace characters that aren't followed by a non-whitespace character.
        // (The Rust regex crate doesn't support negative lookahead, and this is a pretty simple matching rule anyway.)
        let mut j = i;
        while j < text.len() && text[j..].starts_with(char::is_whitespace) {
            j += text[j..].chars().next().unwrap().len_utf8();
        }
        j -= 1; // We don't include the last whitespace character
        if j > i {
            tokens.push(&text[i..j]);
            i = j;
            continue;
        }

        // "\s+"
        // Finally, one or more whitespace characters
        let mut j = i;
        while j < text.len() && text[j..].starts_with(char::is_whitespace) {
            j += text[j..].chars().next().unwrap().len_utf8();
        }
        if j > i {
            tokens.push(&text[i..j]);
            i = j;
            continue;
        }

        // If we didn't match anything, just skip the character
        i += text[i..].chars().next().unwrap().len_utf8();
    }

    tokens
}