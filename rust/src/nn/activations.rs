//! Activation functions: SiLU, Softmax.

use crate::tensor_convert::{candle_to_numpy, numpy_f32_to_candle};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

fn to_pyerr(e: candle_core::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// SiLU activation: x * sigmoid(x)
#[pyfunction]
pub fn run_silu<'py>(
    py: Python<'py>,
    in_features: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let x = numpy_f32_to_candle(&in_features).map_err(to_pyerr)?;
    let result = candle_nn::ops::silu(&x).map_err(to_pyerr)?;
    candle_to_numpy(py, &result)
}

/// Numerically stable softmax along a given dimension.
#[pyfunction]
pub fn run_softmax<'py>(
    py: Python<'py>,
    in_features: PyReadonlyArrayDyn<'py, f32>,
    dim: i64,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let x = numpy_f32_to_candle(&in_features).map_err(to_pyerr)?;
    let ndim = x.dims().len() as i64;
    let actual_dim = if dim < 0 { (ndim + dim) as usize } else { dim as usize };
    let result = candle_nn::ops::softmax(&x, actual_dim).map_err(to_pyerr)?;
    candle_to_numpy(py, &result)
}
