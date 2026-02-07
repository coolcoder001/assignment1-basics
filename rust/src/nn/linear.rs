//! Linear and Embedding operations.

use crate::tensor_convert::{batched_matmul, candle_to_numpy, numpy_f32_to_candle, numpy_i64_to_candle};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

/// Linear transformation: output = in_features @ weights.T
/// Supports batched inputs of shape (..., d_in) -> (..., d_out).
#[pyfunction]
pub fn run_linear<'py>(
    py: Python<'py>,
    _d_in: usize,
    _d_out: usize,
    weights: PyReadonlyArrayDyn<'py, f32>,
    in_features: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let w = numpy_f32_to_candle(&weights).map_err(to_pyerr)?;
    let x = numpy_f32_to_candle(&in_features).map_err(to_pyerr)?;
    let wt = w.t().map_err(to_pyerr)?;
    let result = batched_matmul(&x, &wt).map_err(to_pyerr)?;
    candle_to_numpy(py, &result)
}

/// Embedding lookup: output = weights[token_ids]
#[pyfunction]
pub fn run_embedding<'py>(
    py: Python<'py>,
    _vocab_size: usize,
    _d_model: usize,
    weights: PyReadonlyArrayDyn<'py, f32>,
    token_ids: PyReadonlyArrayDyn<'py, i64>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let w = numpy_f32_to_candle(&weights).map_err(to_pyerr)?;
    let ids = numpy_i64_to_candle(&token_ids).map_err(to_pyerr)?;
    let result = w.index_select(&ids.flatten_all().map_err(to_pyerr)?, 0).map_err(to_pyerr)?;
    // Reshape to (..., d_model)
    let mut out_shape: Vec<usize> = ids.dims().to_vec();
    out_shape.push(w.dim(1).map_err(to_pyerr)?);
    let result = result.reshape(out_shape.as_slice()).map_err(to_pyerr)?;
    candle_to_numpy(py, &result)
}

fn to_pyerr(e: candle_core::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}
