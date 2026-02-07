"""BPE Tokenizer for CS336 Assignment 1."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterator

import regex


# GPT-2 pre-tokenization pattern
GPT2_PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab  # id -> bytes
        self.vocab_inv = {v: k for k, v in vocab.items()}  # bytes -> id
        self.merges = merges
        # Build merge priority: lower index = higher priority
        self.merge_priority = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        # Build special token ID mapping
        self.special_token_ids = {}
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes in self.vocab_inv:
                self.special_token_ids[st] = self.vocab_inv[st_bytes]

        # Build regex for splitting on special tokens (longest match first)
        if self.special_tokens:
            escaped = [regex.escape(st) for st in self.special_tokens]
            self.special_pattern = regex.compile("|".join(escaped))
        else:
            self.special_pattern = None

    def _apply_merges(self, token_list: list[bytes]) -> list[bytes]:
        """Apply BPE merges to a list of byte tokens."""
        if len(token_list) <= 1:
            return token_list

        while True:
            # Find the pair with highest priority (lowest index in merges)
            best_pair = None
            best_priority = float("inf")
            for i in range(len(token_list) - 1):
                pair = (token_list[i], token_list[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair

            if best_pair is None:
                break

            # Merge all occurrences of the best pair
            new_token = best_pair[0] + best_pair[1]
            new_list = []
            i = 0
            while i < len(token_list):
                if (
                    i < len(token_list) - 1
                    and token_list[i] == best_pair[0]
                    and token_list[i + 1] == best_pair[1]
                ):
                    new_list.append(new_token)
                    i += 2
                else:
                    new_list.append(token_list[i])
                    i += 1
            token_list = new_list

            if len(token_list) <= 1:
                break

        return token_list

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a text chunk (without special tokens) into token IDs."""
        if not text:
            return []

        ids = []
        # Pre-tokenize using GPT-2 pattern
        for match in GPT2_PAT.findall(text):
            # Convert to bytes
            byte_tokens = [bytes([b]) for b in match.encode("utf-8")]
            # Apply merges
            merged = self._apply_merges(byte_tokens)
            # Convert to IDs
            for token in merged:
                if token in self.vocab_inv:
                    ids.append(self.vocab_inv[token])
                else:
                    # Fallback: encode each byte individually
                    for b in token:
                        ids.append(self.vocab_inv.get(bytes([b]), 0))
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs, respecting special tokens."""
        if not text:
            return []

        if self.special_pattern is None:
            return self._encode_chunk(text)

        # Split on special tokens
        ids = []
        parts = self.special_pattern.split(text)
        specials = self.special_pattern.findall(text)

        for i, part in enumerate(parts):
            if part:
                ids.extend(self._encode_chunk(part))
            if i < len(specials):
                ids.append(self.special_token_ids[specials[i]])

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to a string."""
        byte_tokens = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_tokens.append(self.vocab[token_id])
            else:
                byte_tokens.append(b"")
        return b"".join(byte_tokens).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable) -> Iterator[int]:
        """Lazily encode an iterable of strings, yielding token IDs one at a time."""
        for line in iterable:
            for token_id in self.encode(line):
                yield token_id


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on a corpus.

    Returns (vocab, merges) where:
    - vocab maps token ID -> token bytes
    - merges is an ordered list of (token1_bytes, token2_bytes) pairs
    """
    # Read input
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on special tokens first so their bytes don't participate in merges
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        escaped = [regex.escape(st) for st in sorted_specials]
        special_pat = regex.compile("|".join(escaped))
        # Only keep the non-special-token parts
        text_segments = special_pat.split(text)
    else:
        text_segments = [text]

    # Pre-tokenize each segment and count pre-token frequencies
    pretoken_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    for segment in text_segments:
        if not segment:
            continue
        for pt in GPT2_PAT.findall(segment):
            byte_seq = tuple(bytes([b]) for b in pt.encode("utf-8"))
            pretoken_counts[byte_seq] += 1

    # Initialize vocab with 256 byte values + special tokens
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])

    # Add special tokens to vocab
    next_id = 256
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    # Build initial pair counts
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    # Track which pre-tokens contain which pairs (for efficient updates)
    # We use a list-based representation: each "word" is a mutable list
    words: list[list[bytes]] = []
    word_counts: list[int] = []
    # Build pair -> word indices mapping
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for token_seq, count in pretoken_counts.items():
        word_idx = len(words)
        word = list(token_seq)
        words.append(word)
        word_counts.append(count)
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_words[pair].add(word_idx)

    for _ in range(num_merges):
        if not pair_counts:
            break

        # Find most frequent pair (tiebreak: max pair lexicographically)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        best_count = pair_counts[best_pair]

        if best_count < 1:
            break

        # Create new token
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        merges.append(best_pair)

        # Update words that contain the best pair
        affected_words = list(pair_to_words.get(best_pair, set()))
        for word_idx in affected_words:
            word = words[word_idx]
            count = word_counts[word_idx]
            new_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    # Before merging, remove old pairs from counts
                    # Left neighbor pair
                    if new_word:
                        old_left_pair = (new_word[-1], best_pair[0])
                        pair_counts[old_left_pair] -= count
                        if pair_counts[old_left_pair] <= 0:
                            del pair_counts[old_left_pair]
                            pair_to_words[old_left_pair].discard(word_idx)
                    # Right neighbor pair
                    if i + 2 < len(word):
                        old_right_pair = (best_pair[1], word[i + 2])
                        pair_counts[old_right_pair] -= count
                        if pair_counts[old_right_pair] <= 0:
                            del pair_counts[old_right_pair]
                            pair_to_words[old_right_pair].discard(word_idx)

                    new_word.append(new_token)

                    # Add new pairs with neighbors
                    if len(new_word) >= 2:
                        new_left_pair = (new_word[-2], new_token)
                        pair_counts[new_left_pair] += count
                        pair_to_words[new_left_pair].add(word_idx)
                    if i + 2 < len(word):
                        new_right_pair = (new_token, word[i + 2])
                        pair_counts[new_right_pair] += count
                        pair_to_words[new_right_pair].add(word_idx)

                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[word_idx] = new_word

        # Clean up the merged pair
        if best_pair in pair_counts:
            del pair_counts[best_pair]
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]

    return vocab, merges
