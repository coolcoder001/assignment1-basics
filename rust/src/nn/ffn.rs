//! SwiGLU Feed-Forward Network.

use crate::tensor_convert::{batched_matmul, candle_to_numpy, numpy_f32_to_candle};
use candle_core::Tensor;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

fn to_pyerr(e: candle_core::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// SwiGLU: (SiLU(x @ W1.T) * (x @ W3.T)) @ W2.T
pub fn swiglu_candle(
    w1: &Tensor,
    w2: &Tensor,
    w3: &Tensor,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let gate = batched_matmul(x, &w1.t()?)?;
    let gate = candle_nn::ops::silu(&gate)?;
    let up = batched_matmul(x, &w3.t()?)?;
    let gated = (gate * up)?;
    let result = batched_matmul(&gated, &w2.t()?)?;
    Ok(result)
}

#[pyfunction]
pub fn run_swiglu<'py>(
    py: Python<'py>,
    _d_model: usize,
    _d_ff: usize,
    w1_weight: PyReadonlyArrayDyn<'py, f32>,
    w2_weight: PyReadonlyArrayDyn<'py, f32>,
    w3_weight: PyReadonlyArrayDyn<'py, f32>,
    in_features: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let w1 = numpy_f32_to_candle(&w1_weight).map_err(to_pyerr)?;
    let w2 = numpy_f32_to_candle(&w2_weight).map_err(to_pyerr)?;
    let w3 = numpy_f32_to_candle(&w3_weight).map_err(to_pyerr)?;
    let x = numpy_f32_to_candle(&in_features).map_err(to_pyerr)?;
    let result = swiglu_candle(&w1, &w2, &w3, &x).map_err(to_pyerr)?;
    candle_to_numpy(py, &result)
}
