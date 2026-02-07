//! Scaled Dot-Product Attention and Multi-Head Self-Attention.

use crate::nn::rope::rope_candle;
use crate::tensor_convert::{
    batched_matmul, candle_to_numpy, numpy_bool_to_candle, numpy_f32_to_candle, numpy_i64_to_candle,
};
use candle_core::{DType, Device, Tensor};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

fn to_pyerr(e: candle_core::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// Scaled dot-product attention.
/// mask: True = attend, False = mask out (set to -inf)
pub fn sdpa_candle(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let d_k = q.dim(candle_core::D::Minus1)?;
    let scale = (d_k as f64).sqrt();

    let q = q.contiguous()?;
    let kt = k.transpose(k.dims().len() - 2, k.dims().len() - 1)?.contiguous()?;
    let scores = (q.matmul(&kt)? / scale)?;

    let scores = if let Some(mask) = mask {
        // mask is u8: 1=attend, 0=mask. Use where_cond to avoid -inf * 0 = NaN
        let mask_broadcast = mask.broadcast_as(scores.shape())?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?;
        mask_broadcast.where_cond(&scores, &neg_inf)?
    } else {
        scores
    };

    let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
    let v = v.contiguous()?;
    attn_weights.matmul(&v)
}

/// Multi-head self-attention (no RoPE).
pub fn mha_candle(
    d_model: usize,
    num_heads: usize,
    q_proj: &Tensor,
    k_proj: &Tensor,
    v_proj: &Tensor,
    o_proj: &Tensor,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let d_head = d_model / num_heads;
    let dims = x.dims();
    let ndim = dims.len();
    let seq_len = dims[ndim - 2];

    // Project Q, K, V
    let q = batched_matmul(x, &q_proj.t()?)?;
    let k = batched_matmul(x, &k_proj.t()?)?;
    let v = batched_matmul(x, &v_proj.t()?)?;

    // Reshape: (..., seq, d_model) -> (..., seq, heads, d_head) -> (..., heads, seq, d_head)
    let batch_dims = &dims[..ndim - 2];
    let mut qkv_shape: Vec<usize> = batch_dims.to_vec();
    qkv_shape.extend_from_slice(&[seq_len, num_heads, d_head]);

    let q = q.reshape(qkv_shape.as_slice())?;
    let k = k.reshape(qkv_shape.as_slice())?;
    let v = v.reshape(qkv_shape.as_slice())?;

    // Transpose seq and heads dims
    let seq_dim = batch_dims.len();
    let head_dim = seq_dim + 1;
    let q = q.transpose(seq_dim, head_dim)?.contiguous()?;
    let k = k.transpose(seq_dim, head_dim)?.contiguous()?;
    let v = v.transpose(seq_dim, head_dim)?.contiguous()?;

    // Causal mask
    let causal = causal_mask(seq_len, x.device())?;

    // Attention
    let attn_out = sdpa_candle(&q, &k, &v, Some(&causal))?;

    // Reshape back: (..., heads, seq, d_head) -> (..., seq, d_model)
    let attn_out = attn_out.transpose(seq_dim, head_dim)?.contiguous()?;
    let mut out_shape: Vec<usize> = batch_dims.to_vec();
    out_shape.extend_from_slice(&[seq_len, d_model]);
    let attn_out = attn_out.reshape(out_shape.as_slice())?;

    // Output projection
    batched_matmul(&attn_out, &o_proj.t()?)
}

/// Multi-head self-attention with RoPE.
pub fn mha_rope_candle(
    d_model: usize,
    num_heads: usize,
    max_seq_len: usize,
    theta: f64,
    q_proj: &Tensor,
    k_proj: &Tensor,
    v_proj: &Tensor,
    o_proj: &Tensor,
    x: &Tensor,
    token_positions: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let d_head = d_model / num_heads;
    let dims = x.dims();
    let ndim = dims.len();
    let seq_len = dims[ndim - 2];
    let batch_dims = &dims[..ndim - 2];

    // Default positions: 0..seq_len
    let default_pos;
    let positions = if let Some(p) = token_positions {
        p
    } else {
        let pos_data: Vec<u32> = (0..seq_len as u32).collect();
        default_pos = Tensor::from_vec(pos_data, &[1, seq_len], x.device())?;
        &default_pos
    };

    // Project Q, K, V
    let q = batched_matmul(x, &q_proj.t()?)?;
    let k = batched_matmul(x, &k_proj.t()?)?;
    let v = batched_matmul(x, &v_proj.t()?)?;

    // Reshape to heads
    let mut qkv_shape: Vec<usize> = batch_dims.to_vec();
    qkv_shape.extend_from_slice(&[seq_len, num_heads, d_head]);

    let q = q.reshape(qkv_shape.as_slice())?;
    let k = k.reshape(qkv_shape.as_slice())?;
    let v = v.reshape(qkv_shape.as_slice())?;

    let seq_dim = batch_dims.len();
    let head_dim = seq_dim + 1;
    let q = q.transpose(seq_dim, head_dim)?.contiguous()?; // (..., heads, seq, d_head)
    let k = k.transpose(seq_dim, head_dim)?.contiguous()?;
    let v = v.transpose(seq_dim, head_dim)?.contiguous()?;

    // Expand positions for each head: (..., seq) -> (..., heads, seq)
    let pos_for_heads = positions.unsqueeze(seq_dim)?; // (..., 1, seq)
    let mut pos_shape: Vec<usize> = positions.dims()[..seq_dim].to_vec();
    pos_shape.push(num_heads);
    pos_shape.push(seq_len);
    let pos_expanded = pos_for_heads.expand(pos_shape.as_slice())?.contiguous()?;

    // Apply RoPE
    let q = rope_candle(d_head, theta, &q, &pos_expanded)?;
    let k = rope_candle(d_head, theta, &k, &pos_expanded)?;

    // Causal mask
    let causal = causal_mask(seq_len, x.device())?;

    // Attention
    let attn_out = sdpa_candle(&q, &k, &v, Some(&causal))?;

    // Reshape back
    let attn_out = attn_out.transpose(seq_dim, head_dim)?.contiguous()?;
    let mut out_shape: Vec<usize> = batch_dims.to_vec();
    out_shape.extend_from_slice(&[seq_len, d_model]);
    let attn_out = attn_out.reshape(out_shape.as_slice())?;

    batched_matmul(&attn_out, &o_proj.t()?)
}

fn causal_mask(seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut mask_data = vec![0u8; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask_data[i * seq_len + j] = 1;
        }
    }
    Tensor::from_vec(mask_data, &[seq_len, seq_len], device)
}

// Python-facing functions

#[pyfunction]
#[pyo3(signature = (q, k, v, mask=None))]
pub fn run_scaled_dot_product_attention<'py>(
    py: Python<'py>,
    q: PyReadonlyArrayDyn<'py, f32>,
    k: PyReadonlyArrayDyn<'py, f32>,
    v: PyReadonlyArrayDyn<'py, f32>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let q_t = numpy_f32_to_candle(&q).map_err(to_pyerr)?;
    let k_t = numpy_f32_to_candle(&k).map_err(to_pyerr)?;
    let v_t = numpy_f32_to_candle(&v).map_err(to_pyerr)?;
    let mask_t = mask
        .as_ref()
        .map(|m| numpy_bool_to_candle(m))
        .transpose()
        .map_err(to_pyerr)?;
    let result = sdpa_candle(&q_t, &k_t, &v_t, mask_t.as_ref()).map_err(to_pyerr)?;
    candle_to_numpy(py, &result)
}

#[pyfunction]
pub fn run_multihead_self_attention<'py>(
    py: Python<'py>,
    d_model: usize,
    num_heads: usize,
    q_proj_weight: PyReadonlyArrayDyn<'py, f32>,
    k_proj_weight: PyReadonlyArrayDyn<'py, f32>,
    v_proj_weight: PyReadonlyArrayDyn<'py, f32>,
    o_proj_weight: PyReadonlyArrayDyn<'py, f32>,
    in_features: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let qw = numpy_f32_to_candle(&q_proj_weight).map_err(to_pyerr)?;
    let kw = numpy_f32_to_candle(&k_proj_weight).map_err(to_pyerr)?;
    let vw = numpy_f32_to_candle(&v_proj_weight).map_err(to_pyerr)?;
    let ow = numpy_f32_to_candle(&o_proj_weight).map_err(to_pyerr)?;
    let x = numpy_f32_to_candle(&in_features).map_err(to_pyerr)?;
    let result = mha_candle(d_model, num_heads, &qw, &kw, &vw, &ow, &x).map_err(to_pyerr)?;
    candle_to_numpy(py, &result)
}

#[pyfunction]
#[pyo3(signature = (d_model, num_heads, max_seq_len, theta, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features, token_positions=None))]
pub fn run_multihead_self_attention_with_rope<'py>(
    py: Python<'py>,
    d_model: usize,
    num_heads: usize,
    max_seq_len: usize,
    theta: f64,
    q_proj_weight: PyReadonlyArrayDyn<'py, f32>,
    k_proj_weight: PyReadonlyArrayDyn<'py, f32>,
    v_proj_weight: PyReadonlyArrayDyn<'py, f32>,
    o_proj_weight: PyReadonlyArrayDyn<'py, f32>,
    in_features: PyReadonlyArrayDyn<'py, f32>,
    token_positions: Option<PyReadonlyArrayDyn<'py, i64>>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let qw = numpy_f32_to_candle(&q_proj_weight).map_err(to_pyerr)?;
    let kw = numpy_f32_to_candle(&k_proj_weight).map_err(to_pyerr)?;
    let vw = numpy_f32_to_candle(&v_proj_weight).map_err(to_pyerr)?;
    let ow = numpy_f32_to_candle(&o_proj_weight).map_err(to_pyerr)?;
    let x = numpy_f32_to_candle(&in_features).map_err(to_pyerr)?;
    let pos = token_positions
        .as_ref()
        .map(|p| numpy_i64_to_candle(p))
        .transpose()
        .map_err(to_pyerr)?;
    let result = mha_rope_candle(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        &qw,
        &kw,
        &vw,
        &ow,
        &x,
        pos.as_ref(),
    )
    .map_err(to_pyerr)?;
    candle_to_numpy(py, &result)
}
