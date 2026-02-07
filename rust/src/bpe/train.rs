//! BPE Training.

use fancy_regex::Regex as FancyRegex;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use regex::Regex;
use std::collections::{HashMap, HashSet};

const GPT2_PAT: &str = r"'(?:[sdmt]|ll|ve|re)| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+(?!\S)|\s+";

#[pyfunction]
pub fn run_train_bpe<'py>(
    py: Python<'py>,
    input_path: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyList>)> {
    let text = std::fs::read_to_string(input_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))?;

    // Split on special tokens first
    let text_segments: Vec<&str> = if !special_tokens.is_empty() {
        let mut sorted_specials = special_tokens.clone();
        sorted_specials.sort_by(|a, b| b.len().cmp(&a.len()));
        let escaped: Vec<String> = sorted_specials.iter().map(|s| regex::escape(s)).collect();
        let special_pat = Regex::new(&escaped.join("|")).unwrap();
        special_pat.split(&text).collect()
    } else {
        vec![text.as_str()]
    };

    let pattern = FancyRegex::new(GPT2_PAT).unwrap();

    // Pre-tokenize and count
    let mut pretoken_counts: HashMap<Vec<Vec<u8>>, u32> = HashMap::new();
    for segment in &text_segments {
        if segment.is_empty() {
            continue;
        }
        for m in pattern.find_iter(segment) {
            let m = match m {
                Ok(m) => m,
                Err(_) => continue,
            };
            let byte_seq: Vec<Vec<u8>> = m.as_str().as_bytes().iter().map(|&b| vec![b]).collect();
            *pretoken_counts.entry(byte_seq).or_insert(0) += 1;
        }
    }

    // Initialize vocab
    let mut vocab: HashMap<u32, Vec<u8>> = HashMap::new();
    for i in 0u32..256 {
        vocab.insert(i, vec![i as u8]);
    }
    let mut next_id = 256u32;
    for st in &special_tokens {
        vocab.insert(next_id, st.as_bytes().to_vec());
        next_id += 1;
    }

    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let num_merges = vocab_size.saturating_sub(vocab.len());

    // Build word list and initial pair counts
    let mut words: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut word_counts: Vec<u32> = Vec::new();
    let mut pair_counts: HashMap<(Vec<u8>, Vec<u8>), i64> = HashMap::new();
    let mut pair_to_words: HashMap<(Vec<u8>, Vec<u8>), HashSet<usize>> = HashMap::new();

    for (token_seq, count) in &pretoken_counts {
        let word_idx = words.len();
        words.push(token_seq.clone());
        word_counts.push(*count);
        for i in 0..token_seq.len().saturating_sub(1) {
            let pair = (token_seq[i].clone(), token_seq[i + 1].clone());
            *pair_counts.entry(pair.clone()).or_insert(0) += *count as i64;
            pair_to_words.entry(pair).or_default().insert(word_idx);
        }
    }

    for _ in 0..num_merges {
        if pair_counts.is_empty() {
            break;
        }

        // Find best pair
        let best_pair = pair_counts
            .iter()
            .max_by(|(a, &ca), (b, &cb)| {
                ca.cmp(&cb).then_with(|| a.cmp(b))
            })
            .map(|(k, _)| k.clone());

        let Some(best_pair) = best_pair else { break };
        let best_count = pair_counts[&best_pair];
        if best_count < 1 {
            break;
        }

        let mut new_token = best_pair.0.clone();
        new_token.extend_from_slice(&best_pair.1);
        vocab.insert(next_id, new_token.clone());
        next_id += 1;
        merges.push(best_pair.clone());

        // Update affected words
        let affected: Vec<usize> = pair_to_words
            .get(&best_pair)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();

        for word_idx in affected {
            let word = &words[word_idx];
            let count = word_counts[word_idx] as i64;
            let mut new_word: Vec<Vec<u8>> = Vec::new();
            let mut i = 0;

            while i < word.len() {
                if i < word.len() - 1 && word[i] == best_pair.0 && word[i + 1] == best_pair.1 {
                    // Remove old neighbor pairs
                    if !new_word.is_empty() {
                        let old_left = (new_word.last().unwrap().clone(), best_pair.0.clone());
                        if let Some(c) = pair_counts.get_mut(&old_left) {
                            *c -= count;
                            if *c <= 0 {
                                pair_counts.remove(&old_left);
                                pair_to_words.remove(&old_left);
                            }
                        }
                    }
                    if i + 2 < word.len() {
                        let old_right = (best_pair.1.clone(), word[i + 2].clone());
                        if let Some(c) = pair_counts.get_mut(&old_right) {
                            *c -= count;
                            if *c <= 0 {
                                pair_counts.remove(&old_right);
                                pair_to_words.remove(&old_right);
                            }
                        }
                    }

                    new_word.push(new_token.clone());

                    // Add new neighbor pairs
                    if new_word.len() >= 2 {
                        let new_left = (new_word[new_word.len() - 2].clone(), new_token.clone());
                        *pair_counts.entry(new_left.clone()).or_insert(0) += count;
                        pair_to_words.entry(new_left).or_default().insert(word_idx);
                    }
                    if i + 2 < word.len() {
                        let new_right = (new_token.clone(), word[i + 2].clone());
                        *pair_counts.entry(new_right.clone()).or_insert(0) += count;
                        pair_to_words.entry(new_right).or_default().insert(word_idx);
                    }

                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            words[word_idx] = new_word;
        }

        // Clean up merged pair
        pair_counts.remove(&best_pair);
        pair_to_words.remove(&best_pair);
    }

    // Convert to Python objects
    let py_vocab = PyDict::new(py);
    for (id, bytes) in &vocab {
        py_vocab.set_item(id, PyBytes::new(py, bytes))?;
    }

    let py_merges = PyList::empty(py);
    for (a, b) in &merges {
        let tuple = PyTuple::new(py, &[
            PyBytes::new(py, a).as_any(),
            PyBytes::new(py, b).as_any(),
        ])?;
        py_merges.append(tuple)?;
    }

    Ok((py_vocab.to_owned(), py_merges.to_owned()))
}
