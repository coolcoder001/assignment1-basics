//! Transformer block and full language model.

use crate::nn::attention::mha_rope_candle;
use crate::nn::ffn::swiglu_candle;
use crate::tensor_convert::{batched_matmul, candle_to_numpy, dict_to_tensors, numpy_f32_to_candle, numpy_i64_to_candle};
use candle_core::Tensor;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

fn to_pyerr(e: candle_core::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let ndim = x.dims().len();
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(ndim - 1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    let normalized = x.broadcast_div(&rms)?;
    normalized.broadcast_mul(weight)
}

/// Transformer block forward pass.
pub fn transformer_block_candle(
    d_model: usize,
    num_heads: usize,
    _d_ff: usize,
    max_seq_len: usize,
    theta: f64,
    weights: &HashMap<String, Tensor>,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let eps = 1e-5;

    // Pre-norm attention
    let h = rmsnorm(x, &weights["ln1.weight"], eps)?;
    let h = mha_rope_candle(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        &weights["attn.q_proj.weight"],
        &weights["attn.k_proj.weight"],
        &weights["attn.v_proj.weight"],
        &weights["attn.output_proj.weight"],
        &h,
        None,
    )?;
    let x = (x + h)?;

    // Pre-norm FFN
    let h = rmsnorm(&x, &weights["ln2.weight"], eps)?;
    let h = swiglu_candle(
        &weights["ffn.w1.weight"],
        &weights["ffn.w2.weight"],
        &weights["ffn.w3.weight"],
        &h,
    )?;
    Ok((x + h)?)
}

#[pyfunction]
pub fn run_transformer_block<'py>(
    py: Python<'py>,
    d_model: usize,
    num_heads: usize,
    d_ff: usize,
    max_seq_len: usize,
    theta: f64,
    weights: &Bound<'py, PyDict>,
    in_features: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let w = dict_to_tensors(weights)?;
    let x = numpy_f32_to_candle(&in_features).map_err(to_pyerr)?;
    let result = transformer_block_candle(d_model, num_heads, d_ff, max_seq_len, theta, &w, &x)
        .map_err(to_pyerr)?;
    candle_to_numpy(py, &result)
}

#[pyfunction]
pub fn run_transformer_lm<'py>(
    py: Python<'py>,
    vocab_size: usize,
    context_length: usize,
    d_model: usize,
    num_layers: usize,
    num_heads: usize,
    d_ff: usize,
    rope_theta: f64,
    weights: &Bound<'py, PyDict>,
    in_indices: PyReadonlyArrayDyn<'py, i64>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let w = dict_to_tensors(weights)?;
    let ids = numpy_i64_to_candle(&in_indices).map_err(to_pyerr)?;

    // Token embeddings
    let emb_weight = &w["token_embeddings.weight"];
    let flat_ids = ids.flatten_all().map_err(to_pyerr)?;
    let mut x = emb_weight
        .index_select(&flat_ids, 0)
        .map_err(to_pyerr)?;
    let id_dims = ids.dims();
    let mut emb_shape: Vec<usize> = id_dims.to_vec();
    emb_shape.push(d_model);
    x = x.reshape(emb_shape.as_slice()).map_err(to_pyerr)?;

    // Transformer blocks
    for i in 0..num_layers {
        let prefix = format!("layers.{i}.");
        let layer_weights: HashMap<String, Tensor> = w
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(&prefix).map(|rest| (rest.to_string(), v.clone()))
            })
            .collect();
        x = transformer_block_candle(d_model, num_heads, d_ff, context_length, rope_theta, &layer_weights, &x)
            .map_err(to_pyerr)?;
    }

    // Final layer norm
    x = rmsnorm(&x, &w["ln_final.weight"], 1e-5).map_err(to_pyerr)?;

    // LM head
    let lm_head = &w["lm_head.weight"];
    let logits = batched_matmul(&x, &lm_head.t().map_err(to_pyerr)?).map_err(to_pyerr)?;

    candle_to_numpy(py, &logits)
}
