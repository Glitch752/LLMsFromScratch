/// This is a basic byte-pair encoding tokenizer. It is used to tokenize the input text into subwords.

use std::{fs, io::Write, path::PathBuf, sync::{Arc, RwLock}};
use clap::{Arg, Command};
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
        .arg(Arg::new("use_temporary_files").help("Use the temporary files generated during the tokenizer build process.").long("use-temporary-files").action(clap::ArgAction::SetTrue))
}
pub const DETOKENIZE_SUBCOMMAND: &str = "detokenize";
pub fn detokenize_cli() -> Command {
    Command::new(DETOKENIZE_SUBCOMMAND)
        .about("Detokenize a list of token IDs using the tokenizer. Input is a list of integers separated by spaces.")
        .arg(Arg::new("tokens").help("The list of token IDs to detokenize.").required(true))
        .arg(Arg::new("use_temporary_files").help("Use the temporary files generated during the tokenizer build process.").long("use-temporary-files").action(clap::ArgAction::SetTrue))
}
pub const FIX_TOKENIZER_SUBCOMMAND: &str = "fix_tokenizer";
pub fn fix_tokenizer_cli() -> Command {
    Command::new(FIX_TOKENIZER_SUBCOMMAND)
        .about("Fix the tokenizer result by removing invalid merges. This shouldn't be necessary, but the tokenizer is somewhat buggy.")
}

const UNKNOWN_TOKEN: &str = "<|unk|>";
const ENDOFTEXT_TOKEN: &str = "<|endoftext|>";
const SEPARATOR_TOKEN: &str = "<|sep|>";

type TokenId = u16;
// Maximum value is 65536 because I'm currenting using an unsigned 16-bit integer for token IDs. If we ever need more tokens, I can change this to u32 and make a small migration script.
const VOCAB_SIZE: TokenId = 50_000;
const SIMULTANEOUS_FILE_LOADS: usize = 32;

lazy_static! {
    pub static ref VOCAB_FILE: PathBuf = DATA_PATH.join("vocab.bin");
    pub static ref MERGES_FILE: PathBuf = DATA_PATH.join("merges.bin");

    pub static ref TEMP_VOCAB_FILE: PathBuf = DATA_PATH.join("vocab.tmp");
    pub static ref TEMP_MERGES_FILE: PathBuf = DATA_PATH.join("merges.tmp");
}

/// A map from pairs of tokens to the BPE rank. Lower ranks indicate more frequent pairs.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Merges(HashMap<StringPair, TokenId>);

/// A map from token string to token ID.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct TokenMap(HashMap<CompactString, TokenId>);

#[derive(Clone, Debug)]
pub struct Tokenizer {
    /// Map from pairs of tokens to the BPE rank. Lower ranks indicate more frequent pairs.
    merges: Merges,

    /// Map from token string to token ID.
    token_map: TokenMap,
    /// Map from token ID to token string.
    reverse_token_map: HashMap<TokenId, CompactString>
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
        tokenizer.regenerate_reverse_token_map();

        tokenizer
    }

    pub fn regenerate_reverse_token_map(&mut self) {
        self.reverse_token_map = self.token_map.0.iter().map(|(k, v)| (*v, k.clone())).collect();
    }

    fn add_token(&mut self, token: CompactString) -> TokenId {
        let value = self.token_map.0.len() as TokenId;
        if self.reverse_token_map.contains_key(&value) {
            panic!("Token ID collision! Token ID {} already exists for token \"{}\".", value, self.reverse_token_map.get(&value).unwrap());
        }
        if self.token_map.0.contains_key(&token) {
            panic!("Token already exists! Token \"{}\" already has ID {}.", token, self.token_map.0.get(&token).unwrap());
        }

        self.token_map.0.insert(token.clone(), value);
        self.reverse_token_map.insert(value, token);

        if self.reverse_token_map.len() != self.token_map.0.len() {
            panic!("Token map and reverse token map are out of sync! Token map has {} entries, reverse token map has {} entries.", self.token_map.0.len(), self.reverse_token_map.len());
        }

        value
    }

    fn add_merge(&mut self, pair: (CompactString, CompactString)) -> TokenId {
        let new_token = format_compact!("{}{}", pair.0, pair.1);
        if !self.token_map.0.contains_key(&new_token) {
            self.add_token(new_token.clone());
        }
        // Finding the maximum value is a bit of a hack maintained since I originally was using token IDs as the rank, which didn't account for edge cases where multiple merges
        // could form the same token.
        self.merges.0.insert(StringPair(pair.0, pair.1), self.merges.0.values().max().unwrap_or(&0) + 1);
        self.token_map.0.get(&new_token).unwrap().clone()
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
            tokenizer.add_token(SEPARATOR_TOKEN.to_compact_string());
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
        let token_chunks = token_chunks.into_iter().collect::<Vec<_>>();

        println!("Loaded data from {} subsets into memory.", max_subset_count);

        // Next, run the BPE algorithm in parallel on the chunks of data.
        let multi_progress = MultiProgress::new();
        let mut thread_join: JoinSet<Vec<TokenId>> = JoinSet::new();
        for (index, chunk) in token_chunks.into_iter().enumerate() {
            let mut tokenizer = tokenizer.clone();
            let progress = multi_progress.clone();
            thread_join.spawn(async move {
                tokenizer.tokenize_chunk(chunk, format!("Tokenizing data; thread {:02}/{:02}", index, threads), progress)
            });
        }
        let token_chunks = thread_join.join_all().await.into_iter().map(RwLock::new).map(Arc::new).collect::<Vec<_>>();

        println!("Finished tokenizing data.");

        let mut pair_count_guess = 0;

        // Next, we run the BPE algorithm to create subword tokens until we reach the desired vocabulary size.
        while (tokenizer.token_map.0.len() as TokenId) < VOCAB_SIZE {
            println!("Vocab size is now {}/{}. Finding next token pair...", tokenizer.token_map.0.len(), VOCAB_SIZE);

            let start_time = std::time::Instant::now();

            // Count the frequencies of all pairs of tokens
            let multi_progress = MultiProgress::new();
            let mut thread_join = JoinSet::new();
            let separater_token_id = *tokenizer.token_map.0.get(&SEPARATOR_TOKEN.to_compact_string()).unwrap();
            for (index, chunk) in token_chunks.iter().enumerate() {
                let chunk = chunk.clone();
                let progress = multi_progress.clone();
                thread_join.spawn(async move {
                    let progress_message = format!("Counting token pairs; thread {:02}/{:02}", index + 1, threads);
                    Tokenizer::get_pair_frequencies(separater_token_id, chunk, progress, progress_message, pair_count_guess)
                });
            }
            let values = thread_join.join_all().await;

            println!("Joining thread frequencies...");
            let mut frequencies = HashMap::<(TokenId, TokenId), usize>::new();
            let mut pair_counts = vec![];
            for value in values {
                pair_counts.push(value.len());
                for (pair, count) in value {
                    *frequencies.entry(pair).or_insert(0) += count;
                }
            }
            pair_count_guess = pair_counts.iter().sum::<usize>() / threads;
            let frequencies = frequencies.into_iter().collect::<Vec<_>>();

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
                    for i in (i * bin_size)..((i + 1) * bin_size).min(frequencies.len()) {
                        let (pair, count) = frequencies.get(i).unwrap();
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
            if most_frequent_pair.unwrap().1 == 0 {
                println!("No pair found with a count greater than 0. Ending tokenizer build.");
                break; // If we didn't find a pair with a count greater than 0, we're done
            }

            let pair = most_frequent_pair.clone().unwrap().0;
            if pair.is_none() {
                println!("No pair found. Ending tokenizer build.");
                break; // If we didn't find a pair, we're done
            }
            let id_pair = pair.unwrap();
            let pair = (tokenizer.reverse_token_map.get(&id_pair.0).unwrap().clone(), tokenizer.reverse_token_map.get(&id_pair.1).unwrap().clone());
            let (first, second) = pair.clone();
            let mut new_token = format_compact!("{}{}", first, second);
            new_token.shrink_to_fit();

            let id = tokenizer.add_merge(pair.clone());

            println!("Found pair {:?} with {} matches to merge into token #{} in {}s.", pair, most_frequent_pair.unwrap().1, id, start_time.elapsed().as_secs());

            println!("Saving temporary tokenizer state...");
            tokenizer.save(true);

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
                    Tokenizer::merge_tokens(&mut chunk, id_pair, id, progress_message, progress);
                });
            }
            thread_join.join_all().await;

            println!("Finished merging tokens in memory in {}s.", start_time.elapsed().as_secs());
        }

        println!("Tokenizer built with {} tokens. Saving to disk...", tokenizer.token_map.0.len());

        tokenizer.save(false);

        // Delete the temporary files
        fs::remove_file(TEMP_VOCAB_FILE.clone()).unwrap();
        fs::remove_file(TEMP_MERGES_FILE.clone()).unwrap();
        
        println!("Saved tokenizer vocab and merges.");

        tokenizer
    }

    pub fn tokenize_chunk(&mut self, mut chunk: Vec<String>, progress_message: String, progress: MultiProgress) -> Vec<TokenId> {
        const PROGRESS_STEP: usize = 10;
        let progress = progress.add(ProgressBar::new((chunk.len() / PROGRESS_STEP) as u64)
            .with_prefix(progress_message)
            .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len}").unwrap()));
        
        /// The number of bytes to modify in a single batch.
        /// This is a tradeoff between memory usage and speed (since updating the original string is O(n))
        const CHUNK_MODIFICATION_CHARACTER_BATCH: usize = 50_000_000;

        let separator_token_id = *self.token_map.0.get(&SEPARATOR_TOKEN.to_compact_string()).unwrap();

        let mut tokenized_data: Vec<TokenId> = vec![];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        let mut i = 0;
        while chunk.len() > 0 {
            let mut length_taken = 0;
            let mut inputs_taken = 0;
            for input in chunk.iter() {
                length_taken += input.len();
                inputs_taken += 1;

                i += 1;
                if i % PROGRESS_STEP == 0 {
                    progress.inc(1);
                }

                let preliminary_tokens = Tokenizer::split_into_preliminary_tokens(input);
                for preliminary_token in preliminary_tokens {
                    let tokens = self.bpe(preliminary_token);
                    tokenized_data.extend(tokens.iter().map(|token| *self.token_map.0.get(token).expect("BPE returned an invalid token!")));
                    tokenized_data.push(separator_token_id);
                }

                // The null separators indicate boundaries as follows:
                // - 1 null byte: End of a token (so this is the boundary of a pair)
                // - 2 null bytes: End of a token and end of a preliminary token (so don't count a pair across this boundary)
                
                if length_taken >= CHUNK_MODIFICATION_CHARACTER_BATCH {
                    break;
                }
            }

            chunk.drain(..inputs_taken);
            chunk.shrink_to_fit();
        }

        tokenized_data.shrink_to_fit();

        progress.finish_and_clear();

        tokenized_data
    }

    pub fn save(&self, temporary: bool) {
        let mut vocab_file = if temporary {
            fs::File::create(TEMP_VOCAB_FILE.clone()).unwrap()
        } else {
            fs::File::create(VOCAB_FILE.clone()).unwrap()
        };
        let mut merges_file = if temporary {
            fs::File::create(TEMP_MERGES_FILE.clone()).unwrap()
        } else {
            fs::File::create(MERGES_FILE.clone()).unwrap()
        };
        vocab_file.write_all(bincode::serialize(&self.token_map).unwrap().as_slice()).unwrap();
        merges_file.write_all(bincode::serialize(&self.merges).unwrap().as_slice()).unwrap();
    }

    fn merge_tokens(chunk: &mut Vec<TokenId>, pair: (TokenId, TokenId), new_id: TokenId, progress_message: String, progress: MultiProgress) {
        const PROGRESS_STEP: usize = 10_000;
        let progress = progress.add(ProgressBar::new((chunk.len() / PROGRESS_STEP) as u64)
            .with_prefix(progress_message)
            .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len} | {msg}").unwrap()));

        const UNKNOWN_TOKEN: TokenId = 0;
        
        let mut i = 0;
        while i < chunk.len() - 1 {
            if i % PROGRESS_STEP == 0 {
                progress.inc(1);
            }

            if *chunk.get(i).unwrap() == pair.0 && *chunk.get(i + 1).unwrap() == pair.1 {
                *chunk.get_mut(i).unwrap() = new_id;
                *chunk.get_mut(i + 1).unwrap() = UNKNOWN_TOKEN;
                i += 1;
            }
            i += 1;
        }

        chunk.retain(|x| *x != UNKNOWN_TOKEN);

        progress.finish_and_clear();
    }

    fn get_pair_frequencies(separater_token_id: TokenId, chunk: Arc<RwLock<Vec<TokenId>>>, progress: MultiProgress, progress_message: String, pair_count_guess: usize) -> Vec<((TokenId, TokenId), usize)> {
        let mut thread_frequencies = HashMap::with_capacity(pair_count_guess);
        
        let chunk = &chunk.try_read().expect("Somehow the RwLock is messed up");

        const PROGRESS_STEP: usize = 10_000;
        let progress = progress.add(ProgressBar::new((chunk.len() / PROGRESS_STEP) as u64)
            .with_prefix(progress_message)
            .with_style(indicatif::ProgressStyle::default_bar().template("{prefix} {bar:60.cyan/blue} {pos}/{len}").unwrap()));
        
        let mut i = 0;
        let mut last_token: TokenId = separater_token_id;
        for token in chunk.iter() {
            i += 1;
            if i % PROGRESS_STEP == 0 {
                progress.inc(1);
            }

            if *token != separater_token_id && last_token != separater_token_id {
                *thread_frequencies.entry((last_token as u32) | ((*token as u32) << 16)).or_insert(0) += 1;
            }
            last_token = *token;
        }

        progress.finish_and_clear();

        let result = thread_frequencies.into_iter().map(|(pair, count)| (((pair & 0xFFFF) as TokenId, (pair >> 16) as TokenId), count)).collect();
        result
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
            let most_important_pair = pairs.into_iter().min_by_key(|pair| self.merges.0.get(pair).unwrap_or(&TokenId::MAX));
            // If we didn't find a pair to merge, we're done
            if most_important_pair.is_none() {
                break;
            }
            let most_important_pair = most_important_pair.unwrap();
            if self.merges.0.get(&most_important_pair).is_none() {
                break;
            }

            let (first, second) = most_important_pair;

            // Next, we merge the pair
            let mut new_tokens = vec![];
            let mut i = 0;
            while i < tokens.len() {
                let mut j = i;
                while j < tokens.len() {
                    if tokens[j] == first && j < tokens.len() - 1 && tokens[j + 1] == second {
                        new_tokens.push(format_compact!("{}{}", first, second));
                        i = j + 2;
                        break;
                    }

                    new_tokens.push(tokens[j].clone());
                    j += 1;
                }

                if j == tokens.len() {
                    break;
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
    pub fn tokenize(&mut self, text: &str) -> Vec<TokenId> {
        let preliminary_tokens = Tokenizer::split_into_preliminary_tokens(text);

        let mut bytepair_tokens: Vec<TokenId> = vec![];
        for preliminary_token in preliminary_tokens {
            bytepair_tokens.extend(self.bpe(preliminary_token).iter().map(|token| *self.token_map.0.get(token).unwrap_or_else(|| {
                println!("Warning: Unknown token found: \"{}\"", token);
                self.token_map.0.get(UNKNOWN_TOKEN).unwrap()
            })));
        }

        bytepair_tokens
    }

    /// Converts a list of token IDs back into text.
    pub fn detokenize(&self, tokens: &[TokenId]) -> String {
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

pub async fn tokenize(text: String, use_temporary_files: bool) {
    let mut tokenizer = Tokenizer::load(use_temporary_files);
    println!("Preliminary tokens: {:?}", Tokenizer::split_into_preliminary_tokens(&text));
    let tokens = tokenizer.tokenize(&text);
    println!("Token IDs: {:?}", tokens);
    println!("Token literals: {:?}", tokens.iter().map(|token| tokenizer.reverse_token_map.get(token).unwrap()).collect::<Vec<&CompactString>>());
    println!("Detokenized: \"{}\"", tokenizer.detokenize(&tokens));
}

pub async fn detokenize(tokens: String, use_temporary_files: bool) {
    let tokenizer = Tokenizer::load(use_temporary_files);
    let tokens: Vec<TokenId> = tokens.split_whitespace().map(|token| token.replace(",", "").parse::<TokenId>().unwrap()).collect();
    let text = tokenizer.detokenize(&tokens);
    println!("{}", text);
}

pub async fn fix_tokenizer() {
    // Load the tokenizer. If the final files don't exist, use the temporary files.
    let mut tokenizer = if MERGES_FILE.exists() && VOCAB_FILE.exists() {
        // Back up the final files
        let mut backup_vocab_file = VOCAB_FILE.clone().with_extension("bin.bak");
        let mut i = 0;
        while backup_vocab_file.exists() {
            i += 1;
            backup_vocab_file = VOCAB_FILE.clone().with_extension(format!("bin.bak.{}", i));
        }
        fs::copy(VOCAB_FILE.clone(), backup_vocab_file).unwrap();

        let mut backup_merges_file = MERGES_FILE.clone().with_extension("bin.bak");
        let mut i = 0;
        while backup_merges_file.exists() {
            i += 1;
            backup_merges_file = MERGES_FILE.clone().with_extension(format!("bin.bak.{}", i));
        }
        fs::copy(MERGES_FILE.clone(), backup_merges_file).unwrap();

        Tokenizer::load(false)
    } else if !TEMP_MERGES_FILE.exists() || !TEMP_VOCAB_FILE.exists() {
        panic!("fix_tokenizer requires either temporary intermediate tokenizer state stored or a finished tokenizer vocab and merges file.");
    } else {
        // If the final files don't exist, use the temporary files
        // Back up the temporary files
        let mut backup_vocab_file = TEMP_VOCAB_FILE.clone().with_extension("tmp.bak");
        let mut i = 0;
        while backup_vocab_file.exists() {
            i += 1;
            backup_vocab_file = TEMP_VOCAB_FILE.clone().with_extension(format!("tmp.bak.{}", i));
        }
        fs::copy(TEMP_VOCAB_FILE.clone(), backup_vocab_file).unwrap();

        let mut backup_merges_file = TEMP_MERGES_FILE.clone().with_extension("tmp.bak");
        let mut i = 0;
        while backup_merges_file.exists() {
            i += 1;
            backup_merges_file = TEMP_MERGES_FILE.clone().with_extension(format!("tmp.bak.{}", i));
        }
        fs::copy(TEMP_MERGES_FILE.clone(), backup_merges_file).unwrap();

        Tokenizer::load(true)
    };

    // If SEPARATOR_TOKEN doesn't exist in the token map, shift all tokens up by one and add it
    if !tokenizer.token_map.0.contains_key(SEPARATOR_TOKEN) {
        println!("Adding separator token to tokenizer.");
        let mut new_token_map = HashMap::<CompactString, TokenId>::new();
        const SEPARATOR_ID: TokenId = 2;
        for (token, id) in tokenizer.token_map.0.iter() {
            if *id >= SEPARATOR_ID {
                new_token_map.insert(token.clone(), id - 1);
            } else {
                new_token_map.insert(token.clone(), *id);
            }
        }
        new_token_map.insert(SEPARATOR_TOKEN.to_compact_string(), SEPARATOR_ID);
        tokenizer.token_map = TokenMap(new_token_map);
        tokenizer.regenerate_reverse_token_map();
    }

    // Find all the tokens with a space in them (that isn't at the start, discluding tokens that are entirely whitespace)
    let mut tokens_to_fix = vec![];
    for (token, id) in tokenizer.token_map.0.iter() {
        if !token.trim().is_empty() && token.chars().count() > 2 {
            let sub_token: String = token.chars().skip(1).take(token.chars().count() - 2).collect();
            if sub_token.contains(' ') {
                tokens_to_fix.push((token.clone(), *id));
            }
        }
        if token.contains("<|unk|>") && token != UNKNOWN_TOKEN {
            tokens_to_fix.push((token.clone(), *id));
        }
    }

    println!("Identified {} tokens requiring fixing: {}", tokens_to_fix.len(), tokens_to_fix.clone().into_iter().map(|e| { format!("\"{}\"", e.0) }).collect::<Vec<_>>().join(", "));

    // Remove all tokens in tokens_to_fix and all merges that contain them
    for (token, id) in tokens_to_fix {
        tokenizer.token_map.0.remove(&token);
        tokenizer.reverse_token_map.remove(&id);
    }

    // Remove all merges that contain or result in tokens that don't exist
    let mut merges_to_remove = vec![];
    for (pair, _) in tokenizer.merges.0.iter() {
        if !tokenizer.token_map.0.contains_key(&pair.0) || !tokenizer.token_map.0.contains_key(&pair.1) || !tokenizer.token_map.0.contains_key(&format_compact!("{}{}", pair.0, pair.1)) {
            merges_to_remove.push(pair.clone());
        }
    }

    println!("Identified {} merges to remove: {}", merges_to_remove.len(), merges_to_remove.clone().into_iter().map(|e| { format!("{:?}", e) }).collect::<Vec<_>>().join(", "));

    for pair in merges_to_remove {
        tokenizer.merges.0.remove(&pair);
    }

    // Ensure no tokens share the same ID
    let mut id_map = HashMap::<TokenId, CompactString>::new();
    let mut conflicting_tokens = vec![];
    for (token, id) in tokenizer.token_map.0.iter() {
        if id_map.contains_key(id) {
            conflicting_tokens.push((id_map.get(id).unwrap().clone(), token.clone()));
        }
        id_map.insert(*id, token.clone());
    }

    println!("Identified {} conflicting tokens: {}", conflicting_tokens.len(), conflicting_tokens.clone().into_iter().map(|e| { format!("\"{}\" and \"{}\"", e.0, e.1) }).collect::<Vec<_>>().join(", "));

    // Fix the conflicting tokens
    for (_, token) in &conflicting_tokens {
        tokenizer.token_map.0.remove(token);
    }
    for (_, token) in &conflicting_tokens {
        tokenizer.add_token(token.clone());
    }
    if conflicting_tokens.len() > 0 {
        tokenizer.regenerate_reverse_token_map();
    }

    // Check if there are any skipped IDs
    let mut skipped_ids = vec![];
    for i in 0..tokenizer.token_map.0.len() as TokenId {
        if !tokenizer.reverse_token_map.contains_key(&i) {
            skipped_ids.push(i);
        }
    }

    println!("Identified {} skipped IDs: {}", skipped_ids.len(), skipped_ids.clone().into_iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "));

    // Fix the skipped IDs
    for (idx, skipped_id) in skipped_ids.iter().enumerate() {
        for(token, token_id) in tokenizer.token_map.0.iter_mut() {
            if *token_id > *skipped_id - idx as TokenId {
                *token_id -= 1;
                tokenizer.reverse_token_map.remove(token_id);
                tokenizer.reverse_token_map.insert(*token_id, token.clone());
            }
        }
    }
    if skipped_ids.len() > 0 {
        tokenizer.regenerate_reverse_token_map();
    }

    println!("New token count: {}.", tokenizer.token_map.0.len());

    tokenizer.save(!MERGES_FILE.exists() || !VOCAB_FILE.exists());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_tokens() {
        let mut chunk = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        Tokenizer::merge_tokens(&mut chunk, (3, 4), 11, "Merging tokens".to_string(), MultiProgress::new());
        assert_eq!(chunk, vec![1, 2, 11, 5, 6, 7, 8, 9, 10]);

        let mut chunk = vec![1, 2, 1, 2, 2, 1, 3, 2, 1, 2];
        Tokenizer::merge_tokens(&mut chunk, (1, 2), 4, "Merging tokens".to_string(), MultiProgress::new());
        assert_eq!(chunk, vec![4, 4, 2, 1, 3, 2, 4]);
    }
    
    #[test]
    fn get_pair_frequencies() {
        let chunk = vec![1, 2, 0, 3, 4, 5, 0, 6];
        let frequencies = Tokenizer::get_pair_frequencies(0, Arc::new(RwLock::new(chunk)), MultiProgress::new(), "Counting token pairs".to_string(), 0);
        assert_eq!(frequencies.len(), 3);
        assert!(frequencies.contains(&((1, 2), 1)));
        assert!(frequencies.contains(&((3, 4), 1)));
        assert!(frequencies.contains(&((4, 5), 1)));
    }
}