use pyo3::prelude::*;

pub mod tensor_convert;
pub mod nn;
pub mod bpe;
pub mod schedule;

#[pymodule]
fn cs336_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // NN operations
    m.add_function(wrap_pyfunction!(nn::linear::run_linear, m)?)?;
    m.add_function(wrap_pyfunction!(nn::linear::run_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(nn::activations::run_silu, m)?)?;
    m.add_function(wrap_pyfunction!(nn::activations::run_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(nn::norm::run_rmsnorm, m)?)?;
    m.add_function(wrap_pyfunction!(nn::loss::run_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(nn::rope::run_rope, m)?)?;
    m.add_function(wrap_pyfunction!(nn::ffn::run_swiglu, m)?)?;
    m.add_function(wrap_pyfunction!(nn::attention::run_scaled_dot_product_attention, m)?)?;
    m.add_function(wrap_pyfunction!(nn::attention::run_multihead_self_attention, m)?)?;
    m.add_function(wrap_pyfunction!(nn::attention::run_multihead_self_attention_with_rope, m)?)?;
    m.add_function(wrap_pyfunction!(nn::transformer::run_transformer_block, m)?)?;
    m.add_function(wrap_pyfunction!(nn::transformer::run_transformer_lm, m)?)?;

    // Schedule
    m.add_function(wrap_pyfunction!(schedule::run_get_lr_cosine_schedule, m)?)?;

    // BPE
    m.add_function(wrap_pyfunction!(bpe::train::run_train_bpe, m)?)?;
    m.add_class::<bpe::tokenizer::RustTokenizer>()?;
    m.add_class::<bpe::tokenizer::TokenIterator>()?;

    Ok(())
}
