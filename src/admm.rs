// Ported from Java/src/pottslab/JavaTools.java (Storath & Weinmann)
// by Claude Sonnet coding agent, Anthropic, 2026.
//
// ADMM splitting strategy for the 2D L2-Potts problem.
//
// 4-connected (anisotropic): alternating horizontal + vertical 1D-Potts steps.
// 8-connected (near-isotropic): four-direction (H+V+diag+antidiag) steps with
// six Lagrange multipliers.

use ndarray::{Array2, Array3};
use crate::processor::{apply_horizontally, apply_vertically, apply_diag, apply_anti_diag};

/// ADMM for the 2D L2-Potts problem with 4-connected (anisotropic) neighbourhood.
///
/// Minimises  gamma * ||Du||_0  +  <weights, ||u - f||_2^2>
/// where ||Du||_0 counts (axis-aligned) pairwise jumps.
///
/// Returns the minimising image u (same shape as `img`).
pub fn admm_4connected(
    img: Array3<f64>,
    gamma: f64,
    weights: Array2<f64>,
    mu_init: f64,
    mu_step: f64,
    stop_tol: f64,
    verbose: bool,
) -> Array3<f64> {
    let (m, n, l) = img.dim();

    let f_norm: f64 = img.iter().map(|&x| x * x).sum();
    if f_norm == 0.0 {
        return img;
    }

    let mut u = Array3::<f64>::zeros((m, n, l));
    let mut v = img.clone();
    let mut lam = Array3::<f64>::zeros((m, n, l));
    let mut mu = mu_init;
    let mut error = f64::INFINITY;
    let mut n_iter = 0usize;

    while error >= stop_tol * f_norm {
        let gamma_prime = 2.0 * gamma;

        // Build modified weights: w'[i,j] = w[i,j] + mu
        let mut weights_prime = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                weights_prime[[i, j]] = weights[[i, j]] + mu;
            }
        }

        // u step: combine data and v, then solve horizontal 1D Potts
        for i in 0..m {
            for j in 0..n {
                for k in 0..l {
                    u[[i, j, k]] = (img[[i, j, k]] * weights[[i, j]]
                        + v[[i, j, k]] * mu
                        - lam[[i, j, k]])
                        / weights_prime[[i, j]];
                }
            }
        }
        apply_horizontally(&mut u, &weights_prime, gamma_prime);

        // v step: combine data and u, then solve vertical 1D Potts
        for i in 0..m {
            for j in 0..n {
                for k in 0..l {
                    v[[i, j, k]] = (img[[i, j, k]] * weights[[i, j]]
                        + u[[i, j, k]] * mu
                        + lam[[i, j, k]])
                        / weights_prime[[i, j]];
                }
            }
        }
        apply_vertically(&mut v, &weights_prime, gamma_prime);

        // Update Lagrange multiplier and compute error
        error = 0.0;
        for i in 0..m {
            for j in 0..n {
                for k in 0..l {
                    let diff = u[[i, j, k]] - v[[i, j, k]];
                    lam[[i, j, k]] += diff * mu;
                    error += diff * diff;
                }
            }
        }

        mu *= mu_step;
        n_iter += 1;

        if verbose {
            eprint!("*");
            if n_iter % 50 == 0 {
                eprintln!();
            }
        }
    }

    if verbose {
        eprintln!("\nTotal iterations: {}", n_iter);
    }

    u
}

/// ADMM for the 2D L2-Potts problem with 8-connected (near-isotropic) neighbourhood.
///
/// Uses four splitting directions (H, V, diag, anti-diag) with six Lagrange multipliers.
/// `omega` = [omega_c, omega_d]: neighbourhood weights for axis-aligned vs diagonal.
/// Default: omega_c = sqrt(2) - 1,  omega_d = 1 - sqrt(2)/2.
pub fn admm_8connected(
    img: Array3<f64>,
    gamma: f64,
    weights: Array2<f64>,
    mu_init: f64,
    mu_step: f64,
    stop_tol: f64,
    verbose: bool,
    omega: [f64; 2],
) -> Array3<f64> {
    let (m, n, l) = img.dim();

    let f_norm: f64 = img.iter().map(|&x| x * x).sum();
    if f_norm == 0.0 {
        return img;
    }

    let omega_c = omega[0];
    let omega_d = omega[1];

    let mut u = Array3::<f64>::zeros((m, n, l));   // horizontal direction variable
    let mut v = img.clone();                         // vertical direction variable
    let mut w_img = img.clone();                     // diagonal direction variable
    let mut z = img.clone();                         // anti-diagonal direction variable
    // Six Lagrange multipliers (u-v, u-w, u-z, v-w, v-z, w-z)
    let mut lam1 = Array3::<f64>::zeros((m, n, l)); // u - v
    let mut lam2 = Array3::<f64>::zeros((m, n, l)); // u - w
    let mut lam3 = Array3::<f64>::zeros((m, n, l)); // u - z
    let mut lam4 = Array3::<f64>::zeros((m, n, l)); // v - w
    let mut lam5 = Array3::<f64>::zeros((m, n, l)); // v - z
    let mut lam6 = Array3::<f64>::zeros((m, n, l)); // w - z

    let mut mu = mu_init;
    let mut error = f64::INFINITY;
    let mut n_iter = 0usize;

    while error >= stop_tol * f_norm {
        let gamma_prime_c = 4.0 * omega_c * gamma;
        let gamma_prime_d = 4.0 * omega_d * gamma;

        let mut weights_prime = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                weights_prime[[i, j]] = weights[[i, j]] + 6.0 * mu;
            }
        }

        // u step (horizontal)
        for i in 0..m {
            for j in 0..n {
                for k in 0..l {
                    u[[i, j, k]] = (img[[i, j, k]] * weights[[i, j]]
                        + 2.0 * mu * (w_img[[i, j, k]] + v[[i, j, k]] + z[[i, j, k]])
                        + 2.0 * (-lam1[[i, j, k]] - lam2[[i, j, k]] - lam3[[i, j, k]]))
                        / weights_prime[[i, j]];
                }
            }
        }
        apply_horizontally(&mut u, &weights_prime, gamma_prime_c);

        // w step (diagonal)
        for i in 0..m {
            for j in 0..n {
                for k in 0..l {
                    w_img[[i, j, k]] = (img[[i, j, k]] * weights[[i, j]]
                        + 2.0 * mu * (u[[i, j, k]] + v[[i, j, k]] + z[[i, j, k]])
                        + 2.0 * (lam2[[i, j, k]] + lam4[[i, j, k]] - lam6[[i, j, k]]))
                        / weights_prime[[i, j]];
                }
            }
        }
        apply_diag(&mut w_img, &weights_prime, gamma_prime_d);

        // v step (vertical)
        for i in 0..m {
            for j in 0..n {
                for k in 0..l {
                    v[[i, j, k]] = (img[[i, j, k]] * weights[[i, j]]
                        + 2.0 * mu * (u[[i, j, k]] + w_img[[i, j, k]] + z[[i, j, k]])
                        + 2.0 * (lam1[[i, j, k]] - lam4[[i, j, k]] - lam5[[i, j, k]]))
                        / weights_prime[[i, j]];
                }
            }
        }
        apply_vertically(&mut v, &weights_prime, gamma_prime_c);

        // z step (anti-diagonal)
        for i in 0..m {
            for j in 0..n {
                for k in 0..l {
                    z[[i, j, k]] = (img[[i, j, k]] * weights[[i, j]]
                        + 2.0 * mu * (u[[i, j, k]] + w_img[[i, j, k]] + v[[i, j, k]])
                        + 2.0 * (lam3[[i, j, k]] + lam5[[i, j, k]] + lam6[[i, j, k]]))
                        / weights_prime[[i, j]];
                }
            }
        }
        apply_anti_diag(&mut z, &weights_prime, gamma_prime_d);

        // Update Lagrange multipliers and compute error
        error = 0.0;
        for i in 0..m {
            for j in 0..n {
                for k in 0..l {
                    lam1[[i, j, k]] += mu * (u[[i, j, k]] - v[[i, j, k]]);
                    lam2[[i, j, k]] += mu * (u[[i, j, k]] - w_img[[i, j, k]]);
                    lam3[[i, j, k]] += mu * (u[[i, j, k]] - z[[i, j, k]]);
                    lam4[[i, j, k]] += mu * (v[[i, j, k]] - w_img[[i, j, k]]);
                    lam5[[i, j, k]] += mu * (v[[i, j, k]] - z[[i, j, k]]);
                    lam6[[i, j, k]] += mu * (w_img[[i, j, k]] - z[[i, j, k]]);
                    let diff = u[[i, j, k]] - v[[i, j, k]];
                    error += diff * diff;
                }
            }
        }

        mu *= mu_step;
        n_iter += 1;

        if verbose {
            eprint!("*");
            if n_iter % 50 == 0 {
                eprintln!();
            }
        }
    }

    if verbose {
        eprintln!("\nTotal iterations: {}", n_iter);
    }

    u
}
