/// This is a basic byte-pair encoding tokenizer. It is used to tokenize the input text into subwords.

use std::{fs, io::Write, path::PathBuf, sync::{Arc, RwLock}};
use clap::{Arg, Command};
use indexmap::IndexMap;
use indicatif::{MultiProgress, ProgressBar};
use lazy_static::lazy_static;
use hashbrown::{Equivalent, HashMap};
use serde::{Serialize, Deserialize};
use tokio::task::JoinSet;
use compact_str::{format_compact, CompactString, ToCompactString};
use crate::{downloader::DOWNLOAD_SUBCOMMAND, DATA_PATH};

pub const GENERATE_VOCAB_SUBCOMMAND: &str = "build_token_map";
pub fn generate_vocab_cli() -> Command {
    Command::new(GENERATE_VOCAB_SUBCOMMAND)
        .about("Build the token map for the tokenizer from the training data.")
        .arg(Arg::new("continue").help("Continue building the token map from the last saved state.").long("continue").action(clap::ArgAction::SetTrue))
        .arg(Arg::new("data_ratio").help("The ratio of the data to use for building the token map.").long("data-ratio").default_value("0.02").value_parser(|v: &str| v.parse::<f64>()))
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
struct Merges(HashMap<StringPair, usize>);

/// A map from token string to token ID.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct TokenMap(HashMap<CompactString, usize>);

#[derive(Clone, Debug)]
pub struct Tokenizer {
    /// Map from pairs of tokens to the BPE rank. Lower ranks indicate more frequent pairs.
    merges: Merges,

    /// Map from token string to token ID.
    token_map: TokenMap,
    /// Map from token ID to token string.
    reverse_token_map: HashMap<usize, CompactString>
}

#[derive(Serialize, Deserialize, Debug, Clone, Hash, Eq, PartialEq)]
struct StringPair(CompactString, CompactString);
impl Equivalent<StringPair> for (CompactString, CompactString) {
    fn equivalent(&self, other: &StringPair) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Equivalent<StringPair> for (&CompactString, &CompactString) {
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

    fn add_token(&mut self, token: CompactString) -> usize {
        let value = self.token_map.0.len();
        self.token_map.0.insert(token.clone(), value);
        self.reverse_token_map.insert(value, token);
        value
    }
    
    pub async fn build_token_map(load_previous: bool, data_ratio: f64) -> Self {
        let subsets_folder = DATA_PATH.join("data/extracted");
        let subsets_available = fs::read_dir(subsets_folder.clone()).unwrap().count();
        if subsets_available == 0 {
            panic!("No subsets found in the data folder. Please run the `download` command first.");
        }

        let mut tokenizer = Tokenizer::new();
        if load_previous {
            tokenizer = Tokenizer::load(true);
            println!("Loaded previous tokenizer state with {} tokens and {} merges.", tokenizer.token_map.0.len(), tokenizer.merges.0.len());
        } else {
            tokenizer.add_token(UNKNOWN_TOKEN.to_compact_string());
            tokenizer.add_token(ENDOFTEXT_TOKEN.to_compact_string());
            // Add every possible byte as a token
            for i in 0..=255 {
                let c = char::from_u32(i).unwrap();
                tokenizer.add_token(c.to_compact_string());
            }

            println!("Initialized blank tokenizer state with {} tokens.", tokenizer.token_map.0.len());
        }

        // This is mostly temporary since it doesn't scale to large datasets. However, I think it's okay if we
        // only build the tokenizer based on the data that can fit in memory.
        let max_subset_count = (subsets_available as f64 * data_ratio).ceil() as usize;
        println!("Loading {} subsets into memory...", max_subset_count.min(subsets_available));
        // We need to split the inputs into chunks so we can process them in parallel, but we can't clone the data.
        let threads = num_cpus::get();
        let mut token_chunks: Vec<Vec<String>> = Vec::with_capacity(threads);
        for _ in 0..threads {
            token_chunks.push(vec![]);
        }
        let mut thread_distribute_index = 0;

        let subsets = fs::read_dir(subsets_folder.clone())
            .unwrap()
            .take(max_subset_count)
            .map(|subset| subset.unwrap());
        for (index, subset) in subsets.enumerate() {
            let count = fs::read_dir(subset.path()).unwrap().count();
            let max_take = if index == max_subset_count - 1 { ((data_ratio * 21.).fract() * count as f64).floor() as usize / SIMULTANEOUS_FILE_LOADS } else { usize::MAX };
            let file_count = count.min(max_take * SIMULTANEOUS_FILE_LOADS);

            let progress = ProgressBar::new(file_count as u64)
                .with_prefix(format!("Loading subset {:02}", index))
                .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:90.cyan/blue} {pos}/{len} {msg}").unwrap());
            token_chunks.iter_mut().for_each(|input| input.reserve(file_count / threads));

            let files = fs::read_dir(subset.path()).unwrap();
            let mut chunks = files.array_chunks::<SIMULTANEOUS_FILE_LOADS>();
            for files in chunks.by_ref().take(max_take) {
                let content_futures = files.map(|file| tokio::fs::read_to_string(file.unwrap().path()));
                let contents = futures_util::future::join_all(content_futures).await.into_iter().collect::<Result<Vec<String>, _>>().unwrap();
                for content in contents.into_iter() {
                    token_chunks.get_mut(thread_distribute_index % threads).unwrap().push(content);

                    thread_distribute_index += 1;
                }

                progress.inc(SIMULTANEOUS_FILE_LOADS as u64);
            }
            for file in chunks.into_remainder().unwrap() {
                let content = tokio::fs::read_to_string(file.unwrap().path()).await.unwrap();
                token_chunks.get_mut(thread_distribute_index % threads).unwrap().push(content);
                thread_distribute_index += 1;

                progress.inc(1);
            }

            progress.finish_and_clear();
        }
        let token_chunks = token_chunks.into_iter().map(Arc::new).collect::<Vec<_>>();

        println!("Loaded data from {} subsets into memory.", max_subset_count);

        // Next, run the BPE algorithm in parallel on the chunks of data.
        let multi_progress = MultiProgress::new();
        let mut thread_join: JoinSet<String> = JoinSet::new();
        for (index, chunk) in token_chunks.iter().enumerate() {
            let chunk = chunk.clone();
            let mut tokenizer = tokenizer.clone();
            let progress = multi_progress.clone();
            thread_join.spawn(async move {
                let progress = progress.add(ProgressBar::new(chunk.len() as u64)
                    .with_prefix(format!("Tokenizing data; thread {:02}/{:02}", index, threads))
                    .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len}").unwrap()));
                
                let mut new_chunk: String = String::new();
                for input in chunk.iter() {
                    let preliminary_tokens = Tokenizer::split_into_preliminary_tokens(input);
                    for preliminary_token in preliminary_tokens {
                        // We use a null byte as a separator, so we need to remove it from the input. I could probably replace it with something else, but we realistically don't need to generate null bytes.
                        let tokens = tokenizer.bpe(preliminary_token).into_iter().map(|s| s.replace("\0", "")).map(Some);
                        for token in tokens {
                            new_chunk.push_str(&token.unwrap());
                            new_chunk.push('\0'); // We use a null byte as a separator
                        }
                        // Add an extra null token to the end of this split token
                        new_chunk.push('\0');
                    }

                    // The null separators indicate boundaries as follows:
                    // - 1 null byte: End of a token (so this is the boundary of a pair)
                    // - 2 null bytes: End of a token and end of a preliminary token (so don't count a pair across this boundary)

                    progress.inc(1);
                }
                new_chunk.shrink_to_fit();

                progress.finish_and_clear();
                
                new_chunk
            });
        }
        let token_chunks = thread_join.join_all().await.into_iter().map(RwLock::new).map(Arc::new).collect::<Vec<_>>();

        println!("Finished tokenizing data.");

        // Next, we run the BPE algorithm to create subword tokens until we reach the desired vocabulary size.
        while tokenizer.token_map.0.len() < VOCAB_SIZE {
            println!("Vocab size is now {}/{}. Finding next token pair...", tokenizer.token_map.0.len(), VOCAB_SIZE);

            let start_time = std::time::Instant::now();

            // Count the frequencies of all pairs of tokens
            let multi_progress = MultiProgress::new();
            let mut thread_join = JoinSet::new();
            for (index, chunk) in token_chunks.iter().enumerate() {
                let chunk = chunk.clone();
                let progress = multi_progress.clone();
                thread_join.spawn(async move {
                    let progress_message = format!("Counting token pairs; thread {:02}/{:02}", index + 1, threads);
                    Tokenizer::get_pair_frequencies(chunk, progress, progress_message)
                });
            }
            let values = thread_join.join_all().await;

            println!("Joining thread frequencies...");
            let mut frequencies = IndexMap::<(CompactString, CompactString), usize>::new();
            for value in values {
                for (pair, count) in value {
                    *frequencies.entry((pair.0, pair.1)).or_insert(0) += count;
                }
            }

            let pair_count = frequencies.len();
            println!("Found {} pairs in {}s. Finding most frequent pair...", pair_count, start_time.elapsed().as_secs());

            // Find the most frequent pair of tokens
            let bin_size = pair_count.div_ceil(threads);
            let frequencies = Arc::new(RwLock::new(frequencies));

            // Parallelize the search for the most frequent pair
            let mut thread_join = JoinSet::new();
            for i in 0..threads {
                let frequencies = frequencies.clone();
                thread_join.spawn(async move {
                    let frequencies = frequencies.try_read().expect("Somehow the RwLock is already locked");
                    let mut most_frequent_pair = None;
                    let mut most_frequent_count = 0;
                    for (pair, count) in frequencies.iter().skip(i * bin_size).take(bin_size) {
                        if *count > most_frequent_count {
                            most_frequent_pair = Some(pair.clone());
                            most_frequent_count = *count;
                        }
                    }

                    (most_frequent_pair, most_frequent_count)
                });
            }
            let most_frequent_pair = thread_join.join_all().await.into_iter().max_by_key(|(_, count)| *count);

            if most_frequent_pair.is_none() {
                println!("No pair found. Ending tokenizer build.");
                break; // If we didn't find a pair, we're done
            }

            let pair = most_frequent_pair.unwrap().0.clone();
            if pair.is_none() {
                println!("No pair found. Ending tokenizer build.");
                break; // If we didn't find a pair, we're done
            }
            let pair = pair.unwrap();
            let (first, second) = pair.clone();
            let mut new_token = format_compact!("{}{}", first, second);
            new_token.shrink_to_fit();

            let id = tokenizer.add_token(new_token.clone());
            tokenizer.merges.0.insert(StringPair(first, second), id);

            println!("Found pair {:?} to merge into token #{} in {}s.", pair, id, start_time.elapsed().as_secs());

            println!("Saving temporary tokenizer state...");
            fs::File::create(TEMP_VOCAB_FILE.clone()).unwrap().write_all(bincode::serialize(&tokenizer.token_map).unwrap().as_slice()).unwrap();
            fs::File::create(TEMP_MERGES_FILE.clone()).unwrap().write_all(bincode::serialize(&tokenizer.merges).unwrap().as_slice()).unwrap();

            println!("Merging tokens in memory...");
            
            let multi_progress = MultiProgress::new();

            let mut thread_join: JoinSet<()> = JoinSet::new();
            for (index, chunk) in token_chunks.iter().enumerate() {
                let chunk = chunk.clone();
                let progress = multi_progress.clone();
                let pair = pair.clone();
                thread_join.spawn(async move {
                    let mut chunk = chunk.try_write().expect("Somehow the RwLock is already locked");
                    let progress_message = format!("Merging token pair {:?}; thread {:02}/{:02}", pair, index + 1, threads);
                    Tokenizer::merge_tokens(&mut chunk, pair, progress_message, progress);
                });
            }
            thread_join.join_all().await;

            println!("Finished merging tokens in memory in {}s.", start_time.elapsed().as_secs());
        }

        println!("Tokenizer built with {} tokens. Saving to disk...", tokenizer.token_map.0.len());

        // Write the token map and merges to disk
        let mut vocab_file = fs::File::create(VOCAB_FILE.clone()).unwrap();
        let mut merges_file = fs::File::create(MERGES_FILE.clone()).unwrap();
        vocab_file.write_all(bincode::serialize(&tokenizer.token_map).unwrap().as_slice()).unwrap();
        merges_file.write_all(bincode::serialize(&tokenizer.merges).unwrap().as_slice()).unwrap();

        // Delete the temporary files
        fs::remove_file(TEMP_VOCAB_FILE.clone()).unwrap();
        fs::remove_file(TEMP_MERGES_FILE.clone()).unwrap();
        
        println!("Saved tokenizer vocab and merges.");

        tokenizer
    }

    fn merge_tokens(chunk: &mut String, pair: (CompactString, CompactString), progress_message: String, progress: MultiProgress) {
        let (first, second) = pair.clone();
        let new_token = format!("{}{}", first, second);
        
        const PROGRESS_STEP: usize = 10_000;
        let progress = progress.add(ProgressBar::new((chunk.len() / PROGRESS_STEP) as u64)
            .with_prefix(progress_message)
            .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len} | {msg}").unwrap()));

        /// The number of tokens to modify in a single batch.
        /// This is a tradeoff between memory usage and speed (since updating the original string is O(n))
        const CHUNK_MODIFICATION_CHARACTER_BATCH: usize = 200_000_000;
        let mut new_chunk = String::new();
        let mut last_token = String::new();
        let mut current_token = String::new();
        // In order to avoid doubling the memory usage, we modify the chunk (somewhat) in-place:
        // Every CHUNK_MODIFICATION_TOKEN_BATCH tokens, we clear the start of the chunk.
        let mut i = 0;
        while chunk.len() > 0 {
            let mut length_taken = 0;
            for char in chunk.chars() {
                length_taken += char.len_utf8();

                i += 1;
                if i % PROGRESS_STEP == 0 {
                    progress.inc(1);
                }

                if char == '\0' {
                    if current_token.is_empty() {
                        // This is the end of a preliminary token; don't count a pair across this boundary.
                        new_chunk.push_str(&last_token);
                        new_chunk.push('\0');
                        new_chunk.push('\0');
                        last_token.clear();
                        current_token.clear();
                        continue;
                    }

                    let should_merge = last_token == first && current_token == second;
                    if !last_token.is_empty() && !should_merge {
                        new_chunk.push_str(&last_token);
                        new_chunk.push('\0');
                    }
                    if length_taken >= CHUNK_MODIFICATION_CHARACTER_BATCH {
                        break;
                    }

                    if should_merge {
                        last_token = new_token.clone();
                    } else {
                        last_token = current_token.clone();
                    }
                    current_token.clear();
                    continue;
                }

                current_token.push(char);
            }
            if !last_token.is_empty() {
                if !current_token.is_empty() && last_token == first && current_token == second {
                    new_chunk.push_str(&new_token);
                } else {
                    new_chunk.push_str(&last_token);
                    if !current_token.is_empty() {
                        new_chunk.push('\0');
                        new_chunk.push_str(&current_token);
                    }
                }
            } else {
                new_chunk.push_str(&current_token);
            }

            chunk.drain(..length_taken);
            chunk.shrink_to_fit();
        }

        new_chunk.shrink_to_fit();
        *chunk = new_chunk;
        
        progress.finish_and_clear();
    }

    fn get_pair_frequencies(chunk: Arc<RwLock<String>>, progress: MultiProgress, progress_message: String) -> HashMap<StringPair, usize> {
        let mut thread_frequencies = HashMap::<StringPair, usize>::new();
            
        let chunk = &chunk.try_read().expect("Somehow the RwLock is messed up");

        const PROGRESS_STEP: usize = 10_000;
        let progress = progress.add(ProgressBar::new((chunk.len() / PROGRESS_STEP) as u64)
            .with_prefix(progress_message)
            .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len}").unwrap()));
        
        let mut last_token = String::new();
        let mut current_token = String::new();
        for (i, char) in chunk.chars().enumerate() {
            if i % PROGRESS_STEP == 0 {
                progress.inc(1);
            }

            if char == '\0' {
                if current_token.is_empty() {
                    // This is the end of a preliminary token; don't count a pair across this boundary.
                    last_token.clear();
                    current_token.clear();
                    continue;
                }

                if !last_token.is_empty() {
                    // This is the end of a token; add the pair to the frequency map.
                    let pair = (last_token.to_compact_string(), current_token.to_compact_string());
                    thread_frequencies.get_mut(&pair)
                        .map(|count| { *count += 1; })
                        .unwrap_or_else(|| {
                            thread_frequencies.insert(StringPair(pair.0.clone(), pair.1.clone()), 1);
                        });
                }

                last_token = current_token.clone();
                current_token.clear();
                continue;
            }

            current_token.push(char);
        }
        if !last_token.is_empty() {
            let pair = (last_token.to_compact_string(), current_token.to_compact_string());
            thread_frequencies.get_mut(&pair)
                .map(|count| { *count += 1; })
                .unwrap_or_else(|| {
                    thread_frequencies.insert(StringPair(pair.0.clone(), pair.1.clone()), 1);
                });
        }

        progress.finish_and_clear();

        thread_frequencies
    }

    /// Gets all possible pairs of tokens in a string.
    fn get_pairs<'a>(values: &'a Vec<CompactString>) -> Vec<(CompactString, CompactString)> {
        let mut pairs = vec![];
        for i in 0..values.len() - 1 {
            pairs.push((values[i].to_owned(), values[i + 1].to_owned()));
        }

        pairs
    }

    /// Applies the Byte-Pair Encoding algorithm to a string.
    fn bpe(&mut self, string: &str) -> Vec<CompactString> {
        let mut tokens = string.bytes().map(|b| char::from_u32(b as u32).unwrap().to_compact_string()).collect();
        let mut pairs = Tokenizer::get_pairs(&tokens);

        if pairs.is_empty() {
            return vec![string.to_compact_string()];
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
                    new_tokens.push(format_compact!("{}{}", first, second));
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
            pairs = Tokenizer::get_pairs(&tokens);
        }

        for token in &mut tokens { token.shrink_to_fit(); }

        tokens
    }

    /// Converts a text into a list of token IDs.
    pub fn tokenize(&mut self, text: &str) -> Vec<usize> {
        let preliminary_tokens = Tokenizer::split_into_preliminary_tokens(text);

        let mut bytepair_tokens: Vec<usize> = vec![];
        for preliminary_token in preliminary_tokens {
            bytepair_tokens.extend(self.bpe(preliminary_token).iter().map(|token| *self.token_map.0.get(token).unwrap_or(self.token_map.0.get(UNKNOWN_TOKEN).unwrap())));
        }

        bytepair_tokens
    }

    /// Converts a list of token IDs back into text.
    pub fn detokenize(&self, tokens: &[usize]) -> String {
        let mut text = String::new();

        for token in tokens {
            let token = self.reverse_token_map.get(token).map(|v| v.to_string()).unwrap_or(UNKNOWN_TOKEN.to_string());
            text.push_str(&token);
        }

        text
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
}

pub async fn tokenize(text: String) {
    let mut tokenizer = Tokenizer::load(false);
    let tokens = tokenizer.tokenize(&text);
    println!("Token IDs: {:?}", tokens);
    println!("Token literals: {:?}", tokens.iter().map(|token| tokenizer.reverse_token_map.get(token).unwrap()).collect::<Vec<&CompactString>>());
    println!("Detokenized: {}", tokenizer.detokenize(&tokens));
}

pub async fn detokenize(tokens: String) {
    let tokenizer = Tokenizer::load(false);
    let tokens: Vec<usize> = tokens.split_whitespace().map(|token| token.replace(",", "").parse().unwrap()).collect();
    let text = tokenizer.detokenize(&tokens);
    println!("{}", text);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_tokens() {
        let mut chunk = "h\0e\0l\0l\0o\0\0 \0w\0o\0r\0l\0d\0\0 \0h\0e\0l\0l\0o".to_string();
        let pair = (CompactString::from("l"), CompactString::from("l"));
        Tokenizer::merge_tokens(&mut chunk, pair, "Merging tokens".to_string(), MultiProgress::new());
        assert_eq!(chunk, "h\0e\0ll\0o\0\0 \0w\0o\0r\0l\0d\0\0 \0h\0e\0ll\0o");

        let pair = (CompactString::from("o"), CompactString::from("w"));
        Tokenizer::merge_tokens(&mut chunk, pair, "Merging tokens".to_string(), MultiProgress::new());
        assert_eq!(chunk, "h\0e\0ll\0o\0\0 \0w\0o\0r\0l\0d\0\0 \0h\0e\0ll\0o");

        let pair = (CompactString::from("ll"), CompactString::from("o"));
        Tokenizer::merge_tokens(&mut chunk, pair, "Merging tokens".to_string(), MultiProgress::new());
        assert_eq!(chunk, "h\0e\0llo\0\0 \0w\0o\0r\0l\0d\0\0 \0h\0e\0llo");

        // Test with a bunch of garbage generated data and a merge that doesn't exist
        let mut chunk = String::new();
        for _ in 0..1000 {
            chunk.push_str("a\0b\0c\0d\0e\0f\0g\0h\0i\0j\0k\0l\0m\0n\0o\0p\0q\0r\0s\0t\0u\0v\0w\0x\0y\0z\0\0 ");
        }
        let original_chunk = chunk.clone();
        let pair = (CompactString::from("z"), CompactString::from("z"));
        Tokenizer::merge_tokens(&mut chunk, pair, "Merging tokens".to_string(), MultiProgress::new());
        assert_eq!(chunk, original_chunk);
    }
    
    #[test]
    fn get_thread_frequencies() {
        let chunk = Arc::new(RwLock::new("Token1\0Token2\0Token3\0\0Token1\0Token2".to_string()));
        let frequencies = Tokenizer::get_pair_frequencies(chunk, MultiProgress::new(), "Counting token pairs".to_string());
        assert_eq!(frequencies.get(&StringPair("Token1".to_compact_string(), "Token2".to_compact_string())).unwrap(), &2);
        assert_eq!(frequencies.get(&StringPair("Token2".to_compact_string(), "Token3".to_compact_string())).unwrap(), &1);
        assert_eq!(frequencies.len(), 2);
    }
}