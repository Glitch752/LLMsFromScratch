/// This is a basic byte-pair encoding tokenizer. It is used to tokenize the input text into subwords.

use std::{fs, io::Write, path::PathBuf, sync::Arc};
use clap::{Arg, Command};
use indicatif::{MultiProgress, ProgressBar};
use lazy_static::lazy_static;
use hashbrown::{Equivalent, HashMap};
use serde::{Serialize, Deserialize};
use tokio::task::JoinSet;
use crate::{downloader::DOWNLOAD_SUBCOMMAND, DATA_PATH};

pub const GENERATE_VOCAB_SUBCOMMAND: &str = "build_token_map";
pub fn generate_vocab_cli() -> Command {
    Command::new(GENERATE_VOCAB_SUBCOMMAND)
        .about("Build the token map for the tokenizer from the training data.")
        .arg(Arg::new("continue").help("Continue building the token map from the last saved state.").long("continue").action(clap::ArgAction::SetTrue))
}
pub const TOKENIZE_SUBCOMMAND: &str = "tokenize";
pub fn tokenize_cli() -> Command {
    Command::new(TOKENIZE_SUBCOMMAND)
        .about("Tokenize a string using the tokenizer. Input is a string of text.")
        .arg(Arg::new("text").help("The text to tokenize.").required(true))
}
pub const DETOKENIZE_SUBCOMMAND: &str = "detokenize";
pub fn detokenize_cli() -> Command {
    Command::new(DETOKENIZE_SUBCOMMAND)
        .about("Detokenize a list of token IDs using the tokenizer. Input is a list of integers separated by spaces.")
        .arg(Arg::new("tokens").help("The list of token IDs to detokenize.").required(true))
}

const UNKNOWN_TOKEN: &str = "<|unk|>";
const ENDOFTEXT_TOKEN: &str = "<|endoftext|>";
const VOCAB_SIZE: usize = 50_000;
const SIMULTANEOUS_FILE_LOADS: usize = 32;

lazy_static! {
    pub static ref VOCAB_FILE: PathBuf = DATA_PATH.join("vocab.bin");
    pub static ref MERGES_FILE: PathBuf = DATA_PATH.join("merges.bin");

    pub static ref TEMP_VOCAB_FILE: PathBuf = DATA_PATH.join("vocab.tmp");
    pub static ref TEMP_MERGES_FILE: PathBuf = DATA_PATH.join("merges.tmp");
}

/// A map from pairs of tokens to the BPE rank. Lower ranks indicate more frequent pairs.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Merges(HashMap<(String, String), usize>);

/// A map from token string to token ID.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct TokenMap(HashMap<String, usize>);

#[derive(Clone, Debug)]
pub struct Tokenizer {
    /// Map from pairs of tokens to the BPE rank. Lower ranks indicate more frequent pairs.
    merges: Merges,

    /// Map from token string to token ID.
    token_map: TokenMap,
    /// Map from token ID to token string.
    reverse_token_map: HashMap<usize, String>
}

#[derive(Hash, Eq, PartialEq)]
struct StringPair(std::string::String, std::string::String);
impl Equivalent<StringPair> for (&std::string::String, &std::string::String) {
    fn equivalent(&self, other: &StringPair) -> bool {
        *self.0 == other.0 && *self.1 == other.1
    }
}

impl Tokenizer {
    fn new() -> Self {
        Tokenizer {
            merges: Merges(HashMap::new()),
            token_map: TokenMap(HashMap::new()),
            reverse_token_map: HashMap::new()
        }
    }

    pub fn load(use_temporary_files: bool) -> Self {
        if use_temporary_files {
            if !TEMP_VOCAB_FILE.exists() || !TEMP_MERGES_FILE.exists() {
                panic!("Temporary files not found; cannot run tokenizer. Please run `{}` and `{}` (without --continue) first.", DOWNLOAD_SUBCOMMAND, GENERATE_VOCAB_SUBCOMMAND); // TODO: Proper error handling
            }
        } else {
            if !VOCAB_FILE.exists() || !MERGES_FILE.exists() {
                panic!("Vocabulary and merge files not found; cannot run tokenizer. Please run `{}` and `{}` first.", DOWNLOAD_SUBCOMMAND, GENERATE_VOCAB_SUBCOMMAND); // TODO: Proper error handling
            }
        }

        let mut tokenizer = Tokenizer::new();
        
        let vocab_file = fs::File::open(if use_temporary_files { TEMP_VOCAB_FILE.clone() } else { VOCAB_FILE.clone() }).unwrap();
        let merges_file = fs::File::open(if use_temporary_files { TEMP_MERGES_FILE.clone() } else { MERGES_FILE.clone() }).unwrap();
        let token_map: TokenMap = bincode::deserialize_from(vocab_file).unwrap();
        let merges: Merges = bincode::deserialize_from(merges_file).unwrap();
        
        tokenizer.token_map = token_map;
        tokenizer.merges = merges;
        tokenizer.reverse_token_map = tokenizer.token_map.0.iter().map(|(k, v)| (*v, k.clone())).collect();

        tokenizer
    }

    fn add_token(&mut self, token: String) -> usize {
        let value = self.token_map.0.len();
        self.token_map.0.insert(token.clone(), value);
        self.reverse_token_map.insert(value, token);
        value
    }
    
    pub async fn build_token_map(load_previous: bool) -> Self {
        let subsets_folder = DATA_PATH.join("data/extracted");
        let subsets_available = fs::read_dir(subsets_folder.clone()).unwrap().count();
        if subsets_available == 0 {
            panic!("No subsets found in the data folder. Please run the `download` command first.");
        }

        let mut tokenizer = Tokenizer::new();
        if load_previous {
            tokenizer = Tokenizer::load(true);
        } else {
            tokenizer.add_token(UNKNOWN_TOKEN.to_string());
            tokenizer.add_token(ENDOFTEXT_TOKEN.to_string());
        }

        // Add every possible byte as a token
        for i in 0..=255 {
            let c = char::from_u32(i).unwrap();
            tokenizer.add_token(c.to_string());
        }

        // This is mostly temporary since it doesn't scale to large datasets. However, I think it's okay if we
        // only build the tokenizer based on the data that can fit in memory.
        let max_subset_count = 2;
        println!("Loading {} subsets into memory...", max_subset_count.min(subsets_available));
        // We need to split the inputs into chunks so we can process them in parallel, but we can't clone the data.
        let threads = num_cpus::get();
        let mut inputs: Vec<Vec<String>> = Vec::with_capacity(threads);
        for _ in 0..threads {
            inputs.push(vec![]);
        }
        let mut thread_distribute_index = 0;

        let subsets = fs::read_dir(subsets_folder.clone())
            .unwrap()
            .take(max_subset_count)
            .map(|subset| subset.unwrap());
        for (index, subset) in subsets.enumerate() {
            let file_count = fs::read_dir(subset.path()).unwrap().count();
            let progress = ProgressBar::new(file_count as u64)
                .with_prefix(format!("Loading subset {:02}", index))
                .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:90.cyan/blue} {pos}/{len} {msg}").unwrap());
            inputs.iter_mut().for_each(|input| input.reserve(file_count / threads));

            let files = fs::read_dir(subset.path()).unwrap();
            let mut chunks = files.array_chunks::<SIMULTANEOUS_FILE_LOADS>();
            for files in chunks.by_ref() {
                let content_futures = files.map(|file| tokio::fs::read_to_string(file.unwrap().path()));
                let contents = futures_util::future::join_all(content_futures).await.into_iter().collect::<Result<Vec<String>, _>>().unwrap();
                for content in contents.into_iter() {
                    inputs.get_mut(thread_distribute_index % threads).unwrap().push(content);
                    thread_distribute_index += 1;
                }

                progress.inc(SIMULTANEOUS_FILE_LOADS as u64);
            }
            for file in chunks.into_remainder().unwrap() {
                let content = tokio::fs::read_to_string(file.unwrap().path()).await.unwrap();
                
                inputs.get_mut(thread_distribute_index % threads).unwrap().push(content);
                thread_distribute_index += 1;

                progress.inc(1);
            }

            progress.finish();
        }
        let inputs = inputs.into_iter().map(Arc::new).collect::<Vec<_>>();
        println!("Loaded {} subsets into memory.", max_subset_count);

        // Next, we run the BPE algorithm to create subword tokens until we reach the desired vocabulary size.
        while tokenizer.token_map.0.len() < VOCAB_SIZE {
            // Count the frequencies of all pairs of tokens
            let multi_progress = MultiProgress::new();
            
            let mut thread_join = JoinSet::new();
            for (index, chunk) in inputs.iter().enumerate() {
                let chunk = chunk.clone();
                let progress = multi_progress.clone();
                let mut tokenizer = tokenizer.clone();
                thread_join.spawn(async move {
                    let mut thread_frequencies = HashMap::<StringPair, usize>::new();
            
                    let progress = progress.add(ProgressBar::new(chunk.len() as u64)
                        .with_prefix(format!("Counting token pairs; thread {:02}/{:02}", index, threads))
                        .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len} | {msg}").unwrap()));
                    
                    for input in chunk.iter() {
                        let preliminary_tokens = split_into_preliminary_tokens(input);
                        for preliminary_token in preliminary_tokens {
                            let tokens = tokenizer.bpe(preliminary_token);
                            for i in 0..tokens.len() - 1 {
                                thread_frequencies.get_mut(&(&tokens[i], &tokens[i + 1]))
                                    .map(|count| { *count += 1; })
                                    .unwrap_or_else(|| { thread_frequencies.insert(StringPair(tokens[i].to_owned(), tokens[i + 1].to_owned()), 1); });
                            }
                        }
                
                        progress.inc(1);
                    }
                    progress.finish();

                    thread_frequencies
                });
            }
            let values = thread_join.join_all().await;
            let mut frequencies = HashMap::<(String, String), usize>::new();
            for value in values {
                for (pair, count) in value {
                    *frequencies.entry((pair.0, pair.1)).or_insert(0) += count;
                }
            }

            // Find the most frequent pair of tokens
            let most_frequent_pair = frequencies.iter().max_by_key(|(_, &count)| count);
            if most_frequent_pair.is_none() {
                break; // If we didn't find a pair, we're done
            }

            let pair = most_frequent_pair.unwrap();
            let (first, second) = pair.0;
            let id = tokenizer.add_token(format!("{}{}", first, second));
            tokenizer.merges.0.insert((pair.0.0.clone(), pair.0.1.clone()), id);

            println!("Merged pair {:?} into token #{}.", pair, id);

            println!("Saving temporary tokenizer state...");
            fs::File::create(TEMP_VOCAB_FILE.clone()).unwrap().write_all(bincode::serialize(&tokenizer.token_map).unwrap().as_slice()).unwrap();
            fs::File::create(TEMP_MERGES_FILE.clone()).unwrap().write_all(bincode::serialize(&tokenizer.merges).unwrap().as_slice()).unwrap();
        }

        println!("Tokenizer built with {} tokens. Saving to disk...", tokenizer.token_map.0.len());

        // Write the token map and merges to disk
        let mut vocab_file = fs::File::create(VOCAB_FILE.clone()).unwrap();
        let mut merges_file = fs::File::create(MERGES_FILE.clone()).unwrap();
        vocab_file.write_all(bincode::serialize(&tokenizer.token_map).unwrap().as_slice()).unwrap();
        merges_file.write_all(bincode::serialize(&tokenizer.merges).unwrap().as_slice()).unwrap();
        
        println!("Saved tokenizer vocab and merges.");

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
        let mut tokens: Vec<String> = string.bytes().map(|b| char::from_u32(b as u32).unwrap().to_string()).collect();
        let mut pairs: Vec<(String, String)> = self.get_pairs(&tokens);

        if pairs.is_empty() {
            return vec![string.to_string()];
        }

        loop {
            // First, we find the most important pair to merge (meaning the pair that occurs most frequently)
            let most_important_pair = pairs.iter().map(|pair| (pair, self.merges.0.get(pair).unwrap_or(&usize::MAX))).min_by_key(|(_, index)| **index);
            // If we didn't find a pair to merge, we're done
            if most_important_pair.is_none() || *most_important_pair.unwrap().1 == usize::MAX {
                break;
            }
            let (first, second) = most_important_pair.unwrap().0;

            // Next, we merge the pair
            let mut new_tokens = vec![];
            let mut i = 0;
            while i < tokens.len() {
                let j = tokens.iter().skip(i).position(|token| token == first);
                if let Some(j) = j {
                    new_tokens.extend(tokens[i..i + j].iter().cloned());
                    i += j;
                }

                if i < tokens.len() - 1 && tokens[i + 1] == *second {
                    new_tokens.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }

            tokens = new_tokens;

            // If we only have one token left, we're done
            if tokens.len() == 1 {
                break;
            }
            
            // Otherwise, we update the pairs and continue
            pairs = self.get_pairs(&tokens);
        }

        tokens
    }

    /// Converts a text into a list of token IDs.
    pub fn tokenize(&mut self, text: &str) -> Vec<usize> {
        let preliminary_tokens = split_into_preliminary_tokens(text);

        let mut bytepair_tokens: Vec<usize> = vec![];
        for preliminary_token in preliminary_tokens {
            bytepair_tokens.extend(self.bpe(preliminary_token).iter().map(|token| *self.token_map.0.get(token).unwrap_or(self.token_map.0.get(UNKNOWN_TOKEN).unwrap())));
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

pub async fn tokenize(text: String) {
    let mut tokenizer = Tokenizer::load(false);
    let tokens = tokenizer.tokenize(&text);
    println!("Token IDs: {:?}", tokens);
    println!("Token literals: {:?}", tokens.iter().map(|token| tokenizer.reverse_token_map.get(token).unwrap()).collect::<Vec<&String>>());
}

pub async fn detokenize(tokens: String) {
    let tokenizer = Tokenizer::load(false);
    let tokens: Vec<usize> = tokens.split_whitespace().map(|token| token.replace(",", "").parse().unwrap()).collect();
    let text = tokenizer.detokenize(&tokens);
    println!("{}", text);
}