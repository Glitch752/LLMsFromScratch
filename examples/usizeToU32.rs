use std::fs;

use compact_str::CompactString;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// A map from token string to token ID.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct OldTokenMap(HashMap<CompactString, usize>);

/// A map from pairs of tokens to the BPE rank. Lower ranks indicate more frequent pairs.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct OldMerges(HashMap<StringPair, usize>);

/// A map from token string to token ID.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct TokenMap(HashMap<CompactString, u32>);

/// A map from pairs of tokens to the BPE rank. Lower ranks indicate more frequent pairs.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Merges(HashMap<StringPair, u32>);

#[derive(Serialize, Deserialize, Debug, Clone, Hash, Eq, PartialEq)]
struct StringPair(CompactString, CompactString);

pub fn main() {
    let vocab_file = fs::File::open("G:\\LLMsFromScratch\\vocab.tmp").unwrap();
    let merges_file = fs::File::open("G:\\LLMsFromScratch\\merges.tmp").unwrap();
    
    let token_map: OldTokenMap = bincode::deserialize_from(vocab_file).unwrap();
    let merges: OldMerges = bincode::deserialize_from(merges_file).unwrap();
    
    let mut new_token_map = HashMap::new();
    for (token, id) in token_map.0 {
        new_token_map.insert(token, id as u32);
    }

    let mut new_merges = HashMap::new();
    for (pair, rank) in merges.0 {
        new_merges.insert(pair, rank as u32);
    }

    let new_token_map_file = fs::File::create("G:\\LLMsFromScratch\\vocab.bin").unwrap();
    let new_merges_file = fs::File::create("G:\\LLMsFromScratch\\merges.bin").unwrap();

    bincode::serialize_into(new_token_map_file, &TokenMap(new_token_map)).unwrap();
    bincode::serialize_into(new_merges_file, &Merges(new_merges)).unwrap();
}