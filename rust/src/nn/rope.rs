//! Rotary Position Embeddings (RoPE).

use crate::tensor_convert::{candle_to_numpy, numpy_f32_to_candle, numpy_i64_to_candle};
use candle_core::{DType, Tensor};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

fn to_pyerr(e: candle_core::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// Apply RoPE to query or key tensor.
///
/// For each dimension pair (2i, 2i+1) at position m:
///   x'_2i   = x_2i   * cos(m * theta_i) - x_{2i+1} * sin(m * theta_i)
///   x'_2i+1 = x_2i   * sin(m * theta_i) + x_{2i+1} * cos(m * theta_i)
/// where theta_i = theta^(-2i/d_k)
pub fn rope_candle(
    d_k: usize,
    theta: f64,
    in_qk: &Tensor,
    positions: &Tensor,
) -> candle_core::Result<Tensor> {
    let in_qk = &in_qk.contiguous()?;
    let device = in_qk.device();
    let half_d = d_k / 2;

    // Compute frequencies: 1 / (theta^(2i/d_k)) for i in 0..half_d
    let freq_data: Vec<f32> = (0..half_d)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / d_k as f64) as f32)
        .collect();
    let freqs = Tensor::from_vec(freq_data, &[half_d], device)?;

    // positions: (..., seq_len), we need angles: (..., seq_len, half_d)
    let pos_f32 = positions.to_dtype(DType::F32)?;
    let pos_unsqueezed = pos_f32.unsqueeze(candle_core::D::Minus1)?; // (..., seq_len, 1)
    let angles = pos_unsqueezed.broadcast_mul(&freqs)?; // (..., seq_len, half_d)

    let cos_a = angles.cos()?;
    let sin_a = angles.sin()?;

    // Split input into even and odd: (..., seq_len, d_k) -> even/odd of shape (..., seq_len, half_d)
    let dims = in_qk.dims();
    let ndim = dims.len();

    // Reshape to (..., seq_len, half_d, 2) then extract even/odd
    let mut reshape_dims: Vec<usize> = dims[..ndim - 1].to_vec();
    reshape_dims.push(half_d);
    reshape_dims.push(2);
    let reshaped = in_qk.reshape(reshape_dims.as_slice())?;

    let x_even = reshaped.narrow(ndim, 0, 1)?.squeeze(ndim)?; // (..., seq_len, half_d)
    let x_odd = reshaped.narrow(ndim, 1, 1)?.squeeze(ndim)?;

    // Apply rotation
    let out_even = (x_even.broadcast_mul(&cos_a)? - x_odd.broadcast_mul(&sin_a)?)?;
    let out_odd = (x_even.broadcast_mul(&sin_a)? + x_odd.broadcast_mul(&cos_a)?)?;

    // Interleave back: stack on last dim then reshape
    let stacked = Tensor::stack(&[&out_even, &out_odd], ndim)?; // (..., seq_len, half_d, 2)
    let result = stacked.reshape(dims)?;

    Ok(result)
}

#[pyfunction]
pub fn run_rope<'py>(
    py: Python<'py>,
    d_k: usize,
    theta: f64,
    _max_seq_len: usize,
    in_query_or_key: PyReadonlyArrayDyn<'py, f32>,
    token_positions: PyReadonlyArrayDyn<'py, i64>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let qk = numpy_f32_to_candle(&in_query_or_key).map_err(to_pyerr)?;
    let pos = numpy_i64_to_candle(&token_positions).map_err(to_pyerr)?;
    let result = rope_candle(d_k, theta, &qk, &pos).map_err(to_pyerr)?;
    candle_to_numpy(py, &result)
}
