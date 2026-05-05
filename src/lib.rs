// Ported from Java/src/pottslab/ (Storath & Weinmann)
// by Claude Sonnet coding agent, Anthropic, 2026.
//
// PyO3 module: exposes Rust implementations to Python.

mod l2potts;
mod l1potts;
mod processor;
mod admm;

use numpy::{
    PyArray1, PyArray2, PyArray3,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    IntoPyArray,
};
use pyo3::prelude::*;
use ndarray::{Array1, Array2, Array3};

/// Solve the 1D scalar L2-Potts problem in-place.
/// Returns the piecewise-constant solution as a 1-D numpy array.
#[pyfunction]
fn solve_l2_potts_1d<'py>(
    py: Python<'py>,
    f: PyReadonlyArray1<f64>,
    gamma: f64,
    weights: Option<PyReadonlyArray1<f64>>,
) -> Bound<'py, PyArray1<f64>> {
    let mut data: Vec<f64> = f.as_slice().unwrap().to_vec();
    let w: Option<Vec<f64>> = weights.map(|w| w.as_slice().unwrap().to_vec());
    l2potts::solve_l2_potts(&mut data, 1, w.as_deref(), gamma);
    Array1::from_vec(data).into_pyarray_bound(py)
}

/// Solve the 1D vector-valued L2-Potts problem.
/// f shape: (n, channels). Returns (n, channels).
#[pyfunction]
fn solve_l2_potts_vv<'py>(
    py: Python<'py>,
    f: PyReadonlyArray2<f64>,
    gamma: f64,
    weights: Option<PyReadonlyArray1<f64>>,
) -> Bound<'py, PyArray2<f64>> {
    let arr = f.as_array().to_owned();
    let (n, ch) = arr.dim();
    // Flatten row-major: data[i*ch + k] = arr[[i, k]]
    let mut data: Vec<f64> = arr.into_raw_vec_and_offset().0;
    let w: Option<Vec<f64>> = weights.map(|w| w.as_slice().unwrap().to_vec());
    l2potts::solve_l2_potts(&mut data, ch, w.as_deref(), gamma);
    let out = Array2::from_shape_vec((n, ch), data).unwrap();
    out.into_pyarray_bound(py)
}

/// Solve the 1D scalar L1-Potts problem in-place.
/// Returns the piecewise-constant solution as a 1-D numpy array.
#[pyfunction]
fn solve_l1_potts_1d<'py>(
    py: Python<'py>,
    f: PyReadonlyArray1<f64>,
    gamma: f64,
    weights: Option<PyReadonlyArray1<f64>>,
) -> Bound<'py, PyArray1<f64>> {
    let mut data: Vec<f64> = f.as_slice().unwrap().to_vec();
    let w: Option<Vec<f64>> = weights.map(|w| w.as_slice().unwrap().to_vec());
    l1potts::solve_l1_potts(&mut data, w.as_deref(), gamma);
    Array1::from_vec(data).into_pyarray_bound(py)
}

/// Compute the weighted median of a 1-D slice.
#[pyfunction]
fn weighted_median_rs(
    data: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
) -> f64 {
    let d = data.as_slice().unwrap();
    let w = weights.as_slice().unwrap();
    l1potts::weighted_median(d, w)
}

/// 2D ADMM with 4-connected (anisotropic) neighbourhood.
/// img shape: (rows, cols, channels).  weights shape: (rows, cols).
#[pyfunction]
fn admm_4connected<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<f64>,
    gamma: f64,
    weights: PyReadonlyArray2<f64>,
    mu_init: f64,
    mu_step: f64,
    stop_tol: f64,
    verbose: bool,
) -> Bound<'py, PyArray3<f64>> {
    let img_arr: Array3<f64> = img.as_array().to_owned();
    let w_arr: Array2<f64> = weights.as_array().to_owned();
    let result = admm::admm_4connected(img_arr, gamma, w_arr, mu_init, mu_step, stop_tol, verbose);
    result.into_pyarray_bound(py)
}

/// 2D ADMM with 8-connected (near-isotropic) neighbourhood.
/// img shape: (rows, cols, channels).  weights shape: (rows, cols).
/// omega: [omega_c, omega_d] neighbourhood weights.
#[pyfunction]
fn admm_8connected<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<f64>,
    gamma: f64,
    weights: PyReadonlyArray2<f64>,
    mu_init: f64,
    mu_step: f64,
    stop_tol: f64,
    verbose: bool,
    omega_c: f64,
    omega_d: f64,
) -> Bound<'py, PyArray3<f64>> {
    let img_arr: Array3<f64> = img.as_array().to_owned();
    let w_arr: Array2<f64> = weights.as_array().to_owned();
    let omega = [omega_c, omega_d];
    let result = admm::admm_8connected(img_arr, gamma, w_arr, mu_init, mu_step, stop_tol, verbose, omega);
    result.into_pyarray_bound(py)
}

#[pymodule]
#[pyo3(name = "_core")]
fn pottslab_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_l2_potts_1d, m)?)?;
    m.add_function(wrap_pyfunction!(solve_l2_potts_vv, m)?)?;
    m.add_function(wrap_pyfunction!(solve_l1_potts_1d, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_median_rs, m)?)?;
    m.add_function(wrap_pyfunction!(admm_4connected, m)?)?;
    m.add_function(wrap_pyfunction!(admm_8connected, m)?)?;
    Ok(())
}
