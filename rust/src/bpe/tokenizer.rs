//! BPE Tokenizer encode/decode.

use fancy_regex::Regex as FancyRegex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use regex::Regex;
use std::collections::HashMap;

const GPT2_PAT: &str = r"'(?:[sdmt]|ll|ve|re)| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+(?!\S)|\s+";

#[pyclass]
pub struct RustTokenizer {
    vocab: HashMap<u32, Vec<u8>>,        // id -> bytes
    vocab_inv: HashMap<Vec<u8>, u32>,    // bytes -> id
    #[allow(dead_code)]
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    merge_priority: HashMap<(Vec<u8>, Vec<u8>), usize>,
    #[allow(dead_code)]
    special_tokens: Vec<String>,
    special_token_ids: HashMap<String, u32>,
    pattern: FancyRegex,
    special_pattern: Option<Regex>,
}

#[pymethods]
impl RustTokenizer {
    #[new]
    #[pyo3(signature = (vocab, merges, special_tokens=None))]
    fn new(
        vocab: &Bound<'_, PyDict>,
        merges: &Bound<'_, PyList>,
        special_tokens: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut vocab_map: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut vocab_inv: HashMap<Vec<u8>, u32> = HashMap::new();

        for (k, v) in vocab.iter() {
            let id: u32 = k.extract()?;
            let bytes: Vec<u8> = v.extract()?;
            vocab_inv.insert(bytes.clone(), id);
            vocab_map.insert(id, bytes);
        }

        let mut merge_list = Vec::new();
        let mut merge_priority = HashMap::new();
        for (i, item) in merges.iter().enumerate() {
            let pair: (Vec<u8>, Vec<u8>) = item.extract()?;
            merge_priority.insert(pair.clone(), i);
            merge_list.push(pair);
        }

        let special_tokens = special_tokens.unwrap_or_default();
        let mut sorted_specials = special_tokens.clone();
        sorted_specials.sort_by(|a, b| b.len().cmp(&a.len()));

        let mut special_token_ids = HashMap::new();
        for st in &sorted_specials {
            let st_bytes = st.as_bytes().to_vec();
            if let Some(&id) = vocab_inv.get(&st_bytes) {
                special_token_ids.insert(st.clone(), id);
            }
        }

        let special_pattern = if !sorted_specials.is_empty() {
            let escaped: Vec<String> = sorted_specials.iter().map(|s| regex::escape(s)).collect();
            Some(Regex::new(&escaped.join("|")).unwrap())
        } else {
            None
        };

        let pattern = FancyRegex::new(GPT2_PAT).unwrap();

        Ok(RustTokenizer {
            vocab: vocab_map,
            vocab_inv,
            merges: merge_list,
            merge_priority,
            special_tokens: sorted_specials,
            special_token_ids,
            pattern,
            special_pattern,
        })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        if self.special_pattern.is_none() {
            return self.encode_chunk(text);
        }

        let special_pat = self.special_pattern.as_ref().unwrap();
        let mut ids = Vec::new();
        let mut last_end = 0;

        for m in special_pat.find_iter(text) {
            if m.start() > last_end {
                ids.extend(self.encode_chunk(&text[last_end..m.start()]));
            }
            let st = m.as_str();
            if let Some(&id) = self.special_token_ids.get(st) {
                ids.push(id);
            }
            last_end = m.end();
        }

        if last_end < text.len() {
            ids.extend(self.encode_chunk(&text[last_end..]));
        }

        ids
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(token_bytes) = self.vocab.get(&id) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    fn encode_iterable(&self, _py: Python<'_>, iterable: &Bound<'_, PyAny>) -> PyResult<TokenIterator> {
        Ok(TokenIterator {
            tokenizer_vocab_inv: self.vocab_inv.clone(),
            tokenizer_merge_priority: self.merge_priority.clone(),
            tokenizer_special_token_ids: self.special_token_ids.clone(),
            tokenizer_pattern: FancyRegex::new(GPT2_PAT).unwrap(),
            tokenizer_special_pattern: self.special_pattern.as_ref().map(|p| Regex::new(p.as_str()).unwrap()),
            py_iter: iterable.call_method0("__iter__")?.unbind(),
            buffer: std::collections::VecDeque::new(),
        })
    }
}

impl RustTokenizer {
    fn encode_chunk(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut ids = Vec::new();
        for m in self.pattern.find_iter(text) {
            let m = match m {
                Ok(m) => m,
                Err(_) => continue,
            };
            let token_bytes: Vec<Vec<u8>> = m.as_str().as_bytes().iter().map(|&b| vec![b]).collect();
            let merged = self.apply_merges(token_bytes);
            for token in merged {
                if let Some(&id) = self.vocab_inv.get(&token) {
                    ids.push(id);
                } else {
                    for b in &token {
                        if let Some(&id) = self.vocab_inv.get(&vec![*b]) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        ids
    }

    fn apply_merges(&self, mut tokens: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        if tokens.len() <= 1 {
            return tokens;
        }

        loop {
            let mut best_pair: Option<(Vec<u8>, Vec<u8>)> = None;
            let mut best_priority = usize::MAX;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&priority) = self.merge_priority.get(&pair) {
                    if priority < best_priority {
                        best_priority = priority;
                        best_pair = Some(pair);
                    }
                }
            }

            let Some(pair) = best_pair else { break };

            let mut new_token = pair.0.clone();
            new_token.extend_from_slice(&pair.1);

            let mut new_tokens = Vec::new();
            let mut i = 0;
            while i < tokens.len() {
                if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    new_tokens.push(new_token.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;

            if tokens.len() <= 1 {
                break;
            }
        }

        tokens
    }
}

#[pyclass]
pub struct TokenIterator {
    tokenizer_vocab_inv: HashMap<Vec<u8>, u32>,
    tokenizer_merge_priority: HashMap<(Vec<u8>, Vec<u8>), usize>,
    tokenizer_special_token_ids: HashMap<String, u32>,
    tokenizer_pattern: FancyRegex,
    tokenizer_special_pattern: Option<Regex>,
    py_iter: PyObject,
    buffer: std::collections::VecDeque<u32>,
}

#[pymethods]
impl TokenIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<u32>> {
        while self.buffer.is_empty() {
            match self.py_iter.call_method0(py, "__next__") {
                Ok(line_obj) => {
                    let text: String = line_obj.extract(py)?;
                    let ids = self.encode_text(&text);
                    self.buffer.extend(ids);
                }
                Err(_) => return Ok(None),
            }
        }
        Ok(self.buffer.pop_front())
    }
}

impl TokenIterator {
    fn encode_text(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        if self.tokenizer_special_pattern.is_none() {
            return self.encode_chunk(text);
        }

        let special_pat = self.tokenizer_special_pattern.as_ref().unwrap();
        let mut ids = Vec::new();
        let mut last_end = 0;

        for m in special_pat.find_iter(text) {
            if m.start() > last_end {
                ids.extend(self.encode_chunk(&text[last_end..m.start()]));
            }
            let st = m.as_str();
            if let Some(&id) = self.tokenizer_special_token_ids.get(st) {
                ids.push(id);
            }
            last_end = m.end();
        }

        if last_end < text.len() {
            ids.extend(self.encode_chunk(&text[last_end..]));
        }

        ids
    }

    fn encode_chunk(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for m in self.tokenizer_pattern.find_iter(text) {
            let m = match m {
                Ok(m) => m,
                Err(_) => continue,
            };
            let token_bytes: Vec<Vec<u8>> = m.as_str().as_bytes().iter().map(|&b| vec![b]).collect();
            let merged = self.apply_merges(token_bytes);
            for token in merged {
                if let Some(&id) = self.tokenizer_vocab_inv.get(&token) {
                    ids.push(id);
                } else {
                    for b in &token {
                        if let Some(&id) = self.tokenizer_vocab_inv.get(&vec![*b]) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        ids
    }

    fn apply_merges(&self, mut tokens: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        if tokens.len() <= 1 {
            return tokens;
        }
        loop {
            let mut best_pair: Option<(Vec<u8>, Vec<u8>)> = None;
            let mut best_priority = usize::MAX;
            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&priority) = self.tokenizer_merge_priority.get(&pair) {
                    if priority < best_priority {
                        best_priority = priority;
                        best_pair = Some(pair);
                    }
                }
            }
            let Some(pair) = best_pair else { break };
            let mut new_token = pair.0.clone();
            new_token.extend_from_slice(&pair.1);
            let mut new_tokens = Vec::new();
            let mut i = 0;
            while i < tokens.len() {
                if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    new_tokens.push(new_token.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
            if tokens.len() <= 1 {
                break;
            }
        }
        tokens
    }
}
