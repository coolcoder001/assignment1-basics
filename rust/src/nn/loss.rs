//! Cross-entropy loss.

use crate::tensor_convert::{candle_to_numpy, numpy_f32_to_candle, numpy_i64_to_candle};

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

fn to_pyerr(e: candle_core::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// Cross-entropy loss using log-sum-exp trick for numerical stability.
#[pyfunction]
pub fn run_cross_entropy<'py>(
    py: Python<'py>,
    inputs: PyReadonlyArrayDyn<'py, f32>,
    targets: PyReadonlyArrayDyn<'py, i64>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let logits = numpy_f32_to_candle(&inputs).map_err(to_pyerr)?; // (batch, vocab)
    let targets = numpy_i64_to_candle(&targets).map_err(to_pyerr)?; // (batch,)
    let _batch_size = logits.dim(0).map_err(to_pyerr)?;

    // log_softmax = x - max(x) - log(sum(exp(x - max(x))))
    let max_val = logits.max_keepdim(1).map_err(to_pyerr)?;
    let shifted = logits.broadcast_sub(&max_val).map_err(to_pyerr)?;
    let exp_shifted = shifted.exp().map_err(to_pyerr)?;
    let sum_exp = exp_shifted.sum_keepdim(1).map_err(to_pyerr)?;
    let log_sum_exp = sum_exp.log().map_err(to_pyerr)?;

    // Gather correct class logits
    let _targets_flat: Vec<u32> = targets
        .flatten_all()
        .map_err(to_pyerr)?
        .to_vec1()
        .map_err(to_pyerr)?;

    // Compute loss per example: -logit[target] + max + log_sum_exp
    let max_squeezed = max_val.squeeze(1).map_err(to_pyerr)?;
    let lse_squeezed = log_sum_exp.squeeze(1).map_err(to_pyerr)?;

    // Get correct logits by gathering
    let correct_logits = logits
        .gather(&targets.unsqueeze(1).map_err(to_pyerr)?, 1)
        .map_err(to_pyerr)?
        .squeeze(1)
        .map_err(to_pyerr)?;

    let loss = (correct_logits.neg().map_err(to_pyerr)?
        + max_squeezed)
        .map_err(to_pyerr)?;
    let loss = (loss + lse_squeezed).map_err(to_pyerr)?;
    let loss = loss
        .mean_all()
        .map_err(to_pyerr)?;

    candle_to_numpy(py, &loss)
}
