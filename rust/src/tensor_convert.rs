//! Conversion utilities between numpy arrays and candle tensors.

use candle_core::{DType, Device, Tensor};
use numpy::ndarray::IxDyn;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Convert a numpy f32 array to a candle Tensor.
pub fn numpy_f32_to_candle(arr: &PyReadonlyArrayDyn<f32>) -> candle_core::Result<Tensor> {
    let view = arr.as_array();
    let shape: Vec<usize> = view.shape().to_vec();
    let data: Vec<f32> = view.iter().cloned().collect();
    Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
}

/// Convert a numpy i64 array to a candle Tensor (as u32).
pub fn numpy_i64_to_candle(arr: &PyReadonlyArrayDyn<i64>) -> candle_core::Result<Tensor> {
    let view = arr.as_array();
    let shape: Vec<usize> = view.shape().to_vec();
    let data: Vec<u32> = view.iter().map(|&v| v as u32).collect();
    Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
}

/// Convert a numpy bool array to a candle Tensor (as u8).
pub fn numpy_bool_to_candle(arr: &PyReadonlyArrayDyn<bool>) -> candle_core::Result<Tensor> {
    let view = arr.as_array();
    let shape: Vec<usize> = view.shape().to_vec();
    let data: Vec<u8> = view.iter().map(|&v| v as u8).collect();
    Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
}

/// Convert a candle Tensor to a numpy f32 array.
pub fn candle_to_numpy<'py>(
    py: Python<'py>,
    tensor: &Tensor,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let tensor = tensor
        .to_dtype(DType::F32)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let shape: Vec<usize> = tensor.dims().to_vec();
    let data: Vec<f32> = tensor
        .flatten_all()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?
        .to_vec1()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let ndarray = numpy::ndarray::ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    Ok(ndarray.into_pyarray(py))
}

/// Batched matmul: handles (…, M, K) @ (K, N) -> (…, M, N).
/// candle's matmul requires same rank, so we flatten batch dims, matmul 2D, reshape back.
pub fn batched_matmul(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let a_dims = a.dims().to_vec();
    let b_dims = b.dims().to_vec();
    if a_dims.len() == b_dims.len() {
        // same rank → candle handles it
        return a.matmul(b);
    }
    // a is nD, b is 2D
    let k = *a_dims.last().unwrap();
    let n = b_dims[b_dims.len() - 1];
    let batch: usize = a_dims[..a_dims.len() - 1].iter().product();
    let a_flat = a.reshape(&[batch, k])?;
    let result = a_flat.matmul(b)?;
    let mut out_shape: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
    out_shape.push(n);
    result.reshape(out_shape.as_slice())
}

/// Convert a Python dict[str, numpy_array] to HashMap<String, Tensor>.
pub fn dict_to_tensors(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, Tensor>> {
    let mut result = HashMap::new();
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        let arr: PyReadonlyArrayDyn<f32> = value.extract()?;
        let tensor = numpy_f32_to_candle(&arr)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        result.insert(key, tensor);
    }
    Ok(result)
}
