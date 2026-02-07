"""Neural network building blocks for CS336 Assignment 1."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def linear_forward(
    d_in: int,
    d_out: int,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    """Apply a linear transformation: output = in_features @ weights.T"""
    return in_features @ weights.T


def embedding_forward(
    vocab_size: int,
    d_model: int,
    weights: Tensor,
    token_ids: Tensor,
) -> Tensor:
    """Look up embeddings for token IDs."""
    return weights[token_ids]


def silu(in_features: Tensor) -> Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return in_features * torch.sigmoid(in_features)


def softmax_forward(in_features: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax."""
    max_val = in_features.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(in_features - max_val)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def rmsnorm_forward(
    d_model: int,
    eps: float,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    """RMS Layer Normalization with affine transform."""
    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    return weights * (in_features / rms)


def cross_entropy_forward(
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """Average cross-entropy loss using log-sum-exp for stability."""
    # log_softmax using log-sum-exp trick
    max_val = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_val
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    # Gather the logit for the correct class
    correct_logits = inputs[torch.arange(inputs.shape[0], device=inputs.device), targets]
    # cross entropy = -log_softmax(correct class)
    loss = -correct_logits + max_val.squeeze(-1) + log_sum_exp
    return loss.mean()


def swiglu_forward(
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """SwiGLU feed-forward: (SiLU(x @ W1.T) * (x @ W3.T)) @ W2.T"""
    gate = silu(in_features @ w1_weight.T)
    x = in_features @ w3_weight.T
    return (gate * x) @ w2_weight.T


def rope_forward(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """Apply Rotary Position Embeddings (RoPE).

    For each dimension pair (2i, 2i+1):
      x'_2i   = x_2i * cos(m * theta_i) - x_{2i+1} * sin(m * theta_i)
      x'_2i+1 = x_2i * sin(m * theta_i) + x_{2i+1} * cos(m * theta_i)
    where theta_i = theta^(-2i/d_k) and m = position
    """
    # Compute frequency for each dimension pair
    half_d = d_k // 2
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=in_query_or_key.device) / d_k))
    # freqs shape: (half_d,)

    # token_positions may be (seq_len,) or (batch, seq_len) or (..., seq_len)
    # We need angles of shape (..., seq_len, half_d)
    angles = token_positions.unsqueeze(-1).float() * freqs  # (..., seq_len, half_d)

    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Split input into even and odd dimensions
    x_even = in_query_or_key[..., 0::2]  # (..., seq_len, half_d)
    x_odd = in_query_or_key[..., 1::2]   # (..., seq_len, half_d)

    # Apply rotation
    out_even = x_even * cos_angles - x_odd * sin_angles
    out_odd = x_even * sin_angles + x_odd * cos_angles

    # Interleave back
    out = torch.stack([out_even, out_odd], dim=-1)  # (..., seq_len, half_d, 2)
    return out.reshape(in_query_or_key.shape)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Scaled dot-product attention.

    mask: True = attend, False = mask out (set to -inf)
    """
    d_k = Q.shape[-1]
    scale = math.sqrt(d_k)

    scores = Q @ K.transpose(-2, -1) / scale  # (..., queries, keys)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax_forward(scores, dim=-1)
    return attn_weights @ V


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """Multi-head self-attention (batched, no RoPE).

    Projects Q, K, V for all heads in a single matrix multiply.
    """
    d_head = d_model // num_heads
    batch_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    # Project all heads at once
    Q = in_features @ q_proj_weight.T  # (..., seq_len, d_model)
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    # Reshape to separate heads: (..., seq_len, num_heads, d_head) -> (..., num_heads, seq_len, d_head)
    Q = Q.view(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)

    # Causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))

    # Attention
    attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)  # (..., num_heads, seq_len, d_head)

    # Concatenate heads: (..., num_heads, seq_len, d_head) -> (..., seq_len, d_model)
    attn_out = attn_out.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_model)

    # Output projection
    return attn_out @ o_proj_weight.T


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
    """Multi-head self-attention with RoPE."""
    d_head = d_model // num_heads
    batch_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    # Default positions: 0, 1, ..., seq_len-1
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0)
        # Expand to match batch dims
        for _ in batch_dims:
            token_positions = token_positions.unsqueeze(0)
        token_positions = token_positions.expand(*batch_dims, -1, -1).squeeze(-2)

    # Project all heads at once
    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    # Reshape to separate heads
    Q = Q.view(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_head).transpose(-3, -2)

    # Apply RoPE to each head's Q and K
    # token_positions shape: (..., seq_len) — need to add num_heads dim
    # Q/K shape: (..., num_heads, seq_len, d_head)
    # We need positions with shape (..., num_heads, seq_len)
    pos_for_rope = token_positions.unsqueeze(-2).expand(*batch_dims, num_heads, seq_len)

    Q = rope_forward(d_head, theta, max_seq_len, Q, pos_for_rope)
    K = rope_forward(d_head, theta, max_seq_len, K, pos_for_rope)

    # Causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))

    # Attention
    attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    # Concatenate heads
    attn_out = attn_out.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, d_model)

    # Output projection
    return attn_out @ o_proj_weight.T


def transformer_block_forward(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    """Pre-norm Transformer block with RoPE.

    Architecture: LN1 -> MHA -> residual -> LN2 -> FFN -> residual
    """
    eps = 1e-5

    # Pre-norm attention
    h = rmsnorm_forward(d_model, eps, weights["ln1.weight"], in_features)
    h = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=h,
    )
    x = in_features + h

    # Pre-norm FFN
    h = rmsnorm_forward(d_model, eps, weights["ln2.weight"], x)
    h = swiglu_forward(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=h,
    )
    return x + h


def transformer_lm_forward(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    """Full Transformer LM forward pass."""
    # Token embeddings
    x = embedding_forward(vocab_size, d_model, weights["token_embeddings.weight"], in_indices)

    # Transformer blocks
    for i in range(num_layers):
        layer_weights = {
            k.replace(f"layers.{i}.", ""): v
            for k, v in weights.items()
            if k.startswith(f"layers.{i}.")
        }
        x = transformer_block_forward(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
        )

    # Final layer norm
    x = rmsnorm_forward(d_model, 1e-5, weights["ln_final.weight"], x)

    # LM head
    logits = x @ weights["lm_head.weight"].T

    return logits
