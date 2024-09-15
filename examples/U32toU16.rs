use std::{env, fs, path::PathBuf};

use compact_str::CompactString;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use dotenv::dotenv;

/// A map from token string to token ID.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct OldTokenMap(HashMap<CompactString, u32>);

/// A map from pairs of tokens to the BPE rank. Lower ranks indicate more frequent pairs.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct OldMerges(HashMap<StringPair, u32>);

/// A map from token string to token ID.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct TokenMap(HashMap<CompactString, u16>);

/// A map from pairs of tokens to the BPE rank. Lower ranks indicate more frequent pairs.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Merges(HashMap<StringPair, u16>);

#[derive(Serialize, Deserialize, Debug, Clone, Hash, Eq, PartialEq)]
struct StringPair(CompactString, CompactString);

pub fn main() {
    dotenv().ok();
    let data_path = PathBuf::from(env::var("DATA_DIRECTORY").expect("DATA_DIRECTORY must be set with a .env file or in the environment"));

    let vocab_file = fs::File::open(data_path.join("vocab.tmp")).unwrap();
    let merges_file = fs::File::open(data_path.join("merges.tmp")).unwrap();
    
    let token_map: OldTokenMap = bincode::deserialize_from(vocab_file).unwrap();
    let merges: OldMerges = bincode::deserialize_from(merges_file).unwrap();
    
    let mut new_token_map = HashMap::new();
    for (token, id) in token_map.0 {
        new_token_map.insert(token, id as u16);
    }

    let mut new_merges = HashMap::new();
    for (pair, rank) in merges.0 {
        new_merges.insert(pair, rank as u16);
    }

    let new_token_map_file = fs::File::create("G:\\LLMsFromScratch\\vocab.tmp.new").unwrap();
    let new_merges_file = fs::File::create("G:\\LLMsFromScratch\\merges.tmp.new").unwrap();

    bincode::serialize_into(new_token_map_file, &TokenMap(new_token_map)).unwrap();
    bincode::serialize_into(new_merges_file, &Merges(new_merges)).unwrap();
}