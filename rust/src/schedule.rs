//! Learning rate schedule.

use pyo3::prelude::*;
use std::f64::consts::PI;

/// Cosine annealing LR schedule with linear warmup.
#[pyfunction]
pub fn run_get_lr_cosine_schedule(
    it: usize,
    max_learning_rate: f64,
    min_learning_rate: f64,
    warmup_iters: usize,
    cosine_cycle_iters: usize,
) -> f64 {
    if it < warmup_iters {
        max_learning_rate * (it as f64) / (warmup_iters as f64)
    } else if it < cosine_cycle_iters {
        let progress = (it - warmup_iters) as f64 / (cosine_cycle_iters - warmup_iters) as f64;
        min_learning_rate
            + 0.5 * (max_learning_rate - min_learning_rate) * (1.0 + (PI * progress).cos())
    } else {
        min_learning_rate
    }
}
