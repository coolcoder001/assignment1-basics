//! RMSNorm.

use crate::tensor_convert::{candle_to_numpy, numpy_f32_to_candle};

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

fn to_pyerr(e: candle_core::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// RMS Layer Normalization: weights * (x / sqrt(mean(x^2) + eps))
#[pyfunction]
pub fn run_rmsnorm<'py>(
    py: Python<'py>,
    _d_model: usize,
    eps: f64,
    weights: PyReadonlyArrayDyn<'py, f32>,
    in_features: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let w = numpy_f32_to_candle(&weights).map_err(to_pyerr)?;
    let x = numpy_f32_to_candle(&in_features).map_err(to_pyerr)?;

    let ndim = x.dims().len();
    let last_dim = ndim - 1;

    // mean(x^2) along last dim
    let x_sq = x.sqr().map_err(to_pyerr)?;
    let mean_sq = x_sq
        .mean_keepdim(last_dim)
        .map_err(to_pyerr)?;
    // sqrt(mean(x^2) + eps)
    let rms = (mean_sq + eps).map_err(to_pyerr)?.sqrt().map_err(to_pyerr)?;
    // Normalize and scale
    let normalized = x.broadcast_div(&rms).map_err(to_pyerr)?;
    let result = normalized.broadcast_mul(&w).map_err(to_pyerr)?;

    candle_to_numpy(py, &result)
}
