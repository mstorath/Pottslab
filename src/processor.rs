// Ported from Java/src/pottslab/PLProcessor.java (Storath & Weinmann)
// by Claude Sonnet coding agent, Anthropic, 2026.
//
// Applies the 1D L2-Potts solver in parallel along rows, columns, diagonals,
// and anti-diagonals of a 3-D image tensor (rows × cols × channels).
// Parallelism via Rayon (replacing Java's ExecutorService thread pool).

use ndarray::{Array3, Array2};
use rayon::prelude::*;

use crate::l2potts::solve_l2_potts;

/// Apply L2-Potts independently to each row of `img` in parallel.
/// `img`     — (rows, cols, channels), C-contiguous
/// `weights` — (rows, cols) per-pixel weights
/// `gamma`   — jump penalty
pub fn apply_horizontally(img: &mut Array3<f64>, weights: &Array2<f64>, gamma: f64) {
    let (rows, cols, ch) = img.dim();
    // Collect row data, solve in parallel, write back
    let results: Vec<(usize, Vec<f64>)> = (0..rows)
        .into_par_iter()
        .map(|i| {
            let mut buf = vec![0.0f64; cols * ch];
            let mut w_buf = vec![0.0f64; cols];
            for j in 0..cols {
                for k in 0..ch {
                    buf[j * ch + k] = img[[i, j, k]];
                }
                w_buf[j] = weights[[i, j]];
            }
            solve_l2_potts(&mut buf, ch, Some(&w_buf), gamma);
            (i, buf)
        })
        .collect();
    for (i, buf) in results {
        for j in 0..cols {
            for k in 0..ch {
                img[[i, j, k]] = buf[j * ch + k];
            }
        }
    }
}

/// Apply L2-Potts independently to each column of `img` in parallel.
pub fn apply_vertically(img: &mut Array3<f64>, weights: &Array2<f64>, gamma: f64) {
    let (rows, cols, ch) = img.dim();
    let results: Vec<(usize, Vec<f64>)> = (0..cols)
        .into_par_iter()
        .map(|j| {
            let mut buf = vec![0.0f64; rows * ch];
            let mut w_buf = vec![0.0f64; rows];
            for i in 0..rows {
                for k in 0..ch {
                    buf[i * ch + k] = img[[i, j, k]];
                }
                w_buf[i] = weights[[i, j]];
            }
            solve_l2_potts(&mut buf, ch, Some(&w_buf), gamma);
            (j, buf)
        })
        .collect();
    for (j, buf) in results {
        for i in 0..rows {
            for k in 0..ch {
                img[[i, j, k]] = buf[i * ch + k];
            }
        }
    }
}

/// Enumerate all diagonals (top-left to bottom-right) of an (rows × cols) image.
/// Returns a Vec of Vec<(row, col)> — one entry per diagonal.
fn diag_indices(rows: usize, cols: usize) -> Vec<Vec<(usize, usize)>> {
    let mut diags: Vec<Vec<(usize, usize)>> = Vec::new();
    // k = col offset of the diagonal starting at row 0
    for k in 0..cols {
        let len = rows.min(cols - k);
        let diag: Vec<(usize, usize)> = (0..len).map(|j| (j, j + k)).collect();
        diags.push(diag);
    }
    // k = row offset of the diagonal starting at col 0
    for k in 1..rows {
        let len = (rows - k).min(cols);
        let diag: Vec<(usize, usize)> = (0..len).map(|j| (j + k, j)).collect();
        diags.push(diag);
    }
    diags
}

/// Apply L2-Potts independently to each main diagonal in parallel.
pub fn apply_diag(img: &mut Array3<f64>, weights: &Array2<f64>, gamma: f64) {
    let (rows, cols, ch) = img.dim();
    let diags = diag_indices(rows, cols);

    let results: Vec<(usize, Vec<f64>, Vec<(usize, usize)>)> = diags
        .into_par_iter()
        .enumerate()
        .map(|(d_idx, diag)| {
            let len = diag.len();
            let mut buf = vec![0.0f64; len * ch];
            let mut w_buf = vec![0.0f64; len];
            for (s, &(i, j)) in diag.iter().enumerate() {
                for k in 0..ch {
                    buf[s * ch + k] = img[[i, j, k]];
                }
                w_buf[s] = weights[[i, j]];
            }
            solve_l2_potts(&mut buf, ch, Some(&w_buf), gamma);
            (d_idx, buf, diag)
        })
        .collect();
    for (_, buf, diag) in results {
        for (s, &(i, j)) in diag.iter().enumerate() {
            for k in 0..ch {
                img[[i, j, k]] = buf[s * ch + k];
            }
        }
    }
}

/// Enumerate all anti-diagonals (top-right to bottom-left) of an (rows × cols) image.
fn anti_diag_indices(rows: usize, cols: usize) -> Vec<Vec<(usize, usize)>> {
    let mut diags: Vec<Vec<(usize, usize)>> = Vec::new();
    // k = col offset from the right at row 0
    for k in 0..cols {
        let len = rows.min(cols - k);
        let diag: Vec<(usize, usize)> = (0..len).map(|j| (j, cols - 1 - (j + k))).collect();
        diags.push(diag);
    }
    for k in 1..rows {
        let len = (rows - k).min(cols);
        let diag: Vec<(usize, usize)> = (0..len).map(|j| (j + k, cols - 1 - j)).collect();
        diags.push(diag);
    }
    diags
}

/// Apply L2-Potts independently to each anti-diagonal in parallel.
pub fn apply_anti_diag(img: &mut Array3<f64>, weights: &Array2<f64>, gamma: f64) {
    let (rows, cols, ch) = img.dim();
    let diags = anti_diag_indices(rows, cols);

    let results: Vec<(usize, Vec<f64>, Vec<(usize, usize)>)> = diags
        .into_par_iter()
        .enumerate()
        .map(|(d_idx, diag)| {
            let len = diag.len();
            let mut buf = vec![0.0f64; len * ch];
            let mut w_buf = vec![0.0f64; len];
            for (s, &(i, j)) in diag.iter().enumerate() {
                for k in 0..ch {
                    buf[s * ch + k] = img[[i, j, k]];
                }
                w_buf[s] = weights[[i, j]];
            }
            solve_l2_potts(&mut buf, ch, Some(&w_buf), gamma);
            (d_idx, buf, diag)
        })
        .collect();
    for (_, buf, diag) in results {
        for (s, &(i, j)) in diag.iter().enumerate() {
            for k in 0..ch {
                img[[i, j, k]] = buf[s * ch + k];
            }
        }
    }
}
