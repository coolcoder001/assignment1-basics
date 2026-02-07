from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# ── Rust dispatch ────────────────────────────────────────────────────
try:
    import cs336_rust as _rust

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# ── Python fallbacks (always available) ──────────────────────────────
from cs336_basics.nn import (
    cross_entropy_forward,
    embedding_forward,
    linear_forward,
    multihead_self_attention,
    multihead_self_attention_with_rope,
    rmsnorm_forward,
    rope_forward,
    scaled_dot_product_attention,
    silu,
    softmax_forward,
    swiglu_forward,
    transformer_block_forward,
    transformer_lm_forward,
)
from cs336_basics.optimizer import AdamW, gradient_clipping
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.tokenizer import BPETokenizer, train_bpe
from cs336_basics.utils import get_batch, get_lr_cosine_schedule


# ── Conversion helpers ───────────────────────────────────────────────
def _to_np(t):
    """Convert a torch Tensor to a numpy array."""
    if isinstance(t, Tensor):
        return t.detach().cpu().contiguous().numpy()
    return t


def _to_torch(arr):
    """Convert a numpy array to a torch Tensor."""
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr.copy())
    return arr


def _dict_to_np(d: dict[str, Tensor]) -> dict[str, np.ndarray]:
    """Convert a dict of torch Tensors to a dict of numpy arrays."""
    return {k: _to_np(v) for k, v in d.items()}


# ── Adapter functions ────────────────────────────────────────────────

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    if _HAS_RUST:
        return _to_torch(_rust.run_linear(d_in, d_out, _to_np(weights), _to_np(in_features)))
    return linear_forward(d_in, d_out, weights, in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    if _HAS_RUST:
        return _to_torch(_rust.run_embedding(vocab_size, d_model, _to_np(weights), _to_np(token_ids).astype(np.int64)))
    return embedding_forward(vocab_size, d_model, weights, token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    if _HAS_RUST:
        return _to_torch(_rust.run_swiglu(
            d_model, d_ff,
            _to_np(w1_weight), _to_np(w2_weight), _to_np(w3_weight),
            _to_np(in_features),
        ))
    return swiglu_forward(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    if _HAS_RUST:
        np_mask = _to_np(mask) if mask is not None else None
        return _to_torch(_rust.run_scaled_dot_product_attention(
            _to_np(Q), _to_np(K), _to_np(V), np_mask,
        ))
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    if _HAS_RUST:
        return _to_torch(_rust.run_multihead_self_attention(
            d_model, num_heads,
            _to_np(q_proj_weight), _to_np(k_proj_weight),
            _to_np(v_proj_weight), _to_np(o_proj_weight),
            _to_np(in_features),
        ))
    return multihead_self_attention(
        d_model, num_heads, q_proj_weight, k_proj_weight,
        v_proj_weight, o_proj_weight, in_features,
    )


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    if _HAS_RUST:
        np_pos = _to_np(token_positions).astype(np.int64) if token_positions is not None else None
        return _to_torch(_rust.run_multihead_self_attention_with_rope(
            d_model, num_heads, max_seq_len, theta,
            _to_np(q_proj_weight), _to_np(k_proj_weight),
            _to_np(v_proj_weight), _to_np(o_proj_weight),
            _to_np(in_features), np_pos,
        ))
    return multihead_self_attention_with_rope(
        d_model, num_heads, max_seq_len, theta,
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        in_features, token_positions,
    )


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    if _HAS_RUST:
        return _to_torch(_rust.run_rope(
            d_k, theta, max_seq_len,
            _to_np(in_query_or_key),
            _to_np(token_positions).astype(np.int64),
        ))
    return rope_forward(d_k, theta, max_seq_len, in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    if _HAS_RUST:
        return _to_torch(_rust.run_transformer_block(
            d_model, num_heads, d_ff, max_seq_len, theta,
            _dict_to_np(weights), _to_np(in_features),
        ))
    return transformer_block_forward(
        d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features,
    )


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    if _HAS_RUST:
        return _to_torch(_rust.run_transformer_lm(
            vocab_size, context_length, d_model, num_layers, num_heads,
            d_ff, rope_theta, _dict_to_np(weights),
            _to_np(in_indices).astype(np.int64),
        ))
    return transformer_lm_forward(
        vocab_size, context_length, d_model, num_layers, num_heads,
        d_ff, rope_theta, weights, in_indices,
    )


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    if _HAS_RUST:
        return _to_torch(_rust.run_rmsnorm(d_model, eps, _to_np(weights), _to_np(in_features)))
    return rmsnorm_forward(d_model, eps, weights, in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    if _HAS_RUST:
        return _to_torch(_rust.run_silu(_to_np(in_features)))
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    if _HAS_RUST:
        return _to_torch(_rust.run_softmax(_to_np(in_features), dim))
    return softmax_forward(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    if _HAS_RUST:
        result = _rust.run_cross_entropy(_to_np(inputs), _to_np(targets).astype(np.int64))
        return _to_torch(result)
    return cross_entropy_forward(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if _HAS_RUST:
        return _rust.run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    return get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    if _HAS_RUST:
        return _rust.RustTokenizer(vocab, merges, special_tokens)
    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if _HAS_RUST:
        return _rust.run_train_bpe(str(input_path), vocab_size, special_tokens)
    return train_bpe(input_path, vocab_size, special_tokens, **kwargs)
