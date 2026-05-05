// Ported from Java/src/pottslab/L2Potts.java (Storath & Weinmann)
// by Claude Sonnet coding agent, Anthropic, 2026.
//
// Solves the 1D L2-Potts functional:
//   min_u  gamma * ||Du||_0  +  ||u - f||_2^2  (weighted)
// via dynamic programming in O(n^2) time.
//
// Works for scalar (vec_len=1) and vector-valued (vec_len>1) signals.
// Data layout: row-major flat slice of length n*vec_len, i.e.,
//   data[i*vec_len + k] is the k-th channel of the i-th sample.

/// Solve the 1D L2-Potts problem in-place.
///
/// `data`     — flattened (n × vec_len), modified in-place with the solution
/// `vec_len`  — number of channels (1 for scalar signals)
/// `weights`  — per-sample weights; None means all weights = 1
/// `gamma`    — jump penalty
pub fn solve_l2_potts(
    data: &mut [f64],
    vec_len: usize,
    weights: Option<&[f64]>,
    gamma: f64,
) {
    let n = data.len() / vec_len;
    if n == 0 {
        return;
    }

    let get_w = |i: usize| -> f64 {
        weights.map_or(1.0, |w| w[i])
    };

    // Cumulative moments (index 0 is the "zero" sentinel before the first element).
    // m[j+1][k] = sum_{i=0}^{j} w[i] * data[i*vec_len + k]
    // s[j+1]    = sum_{i=0}^{j} w[i] * ||data[i,:]||^2
    // wc[j+1]   = sum_{i=0}^{j} w[i]
    let mut m = vec![0.0f64; (n + 1) * vec_len]; // cumulative weighted first moments
    let mut s = vec![0.0f64; n + 1];              // cumulative weighted squared norms
    let mut wc = vec![0.0f64; n + 1];             // cumulative weights

    for j in 0..n {
        let wj = get_w(j);
        wc[j + 1] = wc[j] + wj;
        s[j + 1] = s[j];
        for k in 0..vec_len {
            let d = data[j * vec_len + k];
            m[(j + 1) * vec_len + k] = m[j * vec_len + k] + wj * d;
            s[j + 1] += wj * d * d;
        }
    }

    // DP arrays
    // arr_p[r-1] = optimal Potts value for the sub-signal [0..r]
    // arr_j[r-1] = best jump location (last index of previous segment) for [0..r]
    let mut arr_p = vec![0.0f64; n];
    let mut arr_j = vec![0usize; n]; // 0 means "no jump" (entire [0..r] is constant)

    for r in 1..=n {
        // Cost of the constant solution on [0..r]:
        //   s[r] - ||m[r]||^2 / wc[r]
        let w_r = wc[r];
        let cost_const = if w_r == 0.0 {
            0.0
        } else {
            let mut norm_m_sq = 0.0f64;
            for k in 0..vec_len {
                let mk = m[r * vec_len + k];
                norm_m_sq += mk * mk;
            }
            s[r] - norm_m_sq / w_r
        };
        arr_p[r - 1] = cost_const;
        arr_j[r - 1] = 0;

        // Try every possible last jump at l-1 (so the last segment is [l..r]).
        // l runs from r down to 2 (segments of length >= 1).
        for l in (2..=r).rev() {
            let w_diff = wc[r] - wc[l - 1];
            let d = if w_diff == 0.0 {
                0.0
            } else {
                let mut norm_sq = 0.0f64;
                for k in 0..vec_len {
                    let diff = m[r * vec_len + k] - m[(l - 1) * vec_len + k];
                    norm_sq += diff * diff;
                }
                s[r] - s[l - 1] - norm_sq / w_diff
            };
            let dpg = d + gamma;

            // Acceleration: if d + gamma already exceeds the best value for [0..r],
            // no further extension can improve it (deviations are non-decreasing as
            // the segment grows), so break.
            if dpg > arr_p[r - 1] {
                break;
            }

            let p = arr_p[l - 2] + dpg;
            if p < arr_p[r - 1] {
                arr_p[r - 1] = p;
                arr_j[r - 1] = l - 1;
            }
        }
    }

    // Backtrack through arr_j to reconstruct piecewise-constant solution.
    let mut r = n;
    loop {
        let l = arr_j[r - 1]; // last index of the previous segment (0 = beginning)
        // Compute mean of data[l..r] with weights
        let w_total = wc[r] - wc[l];
        for k in 0..vec_len {
            let mean = if w_total == 0.0 {
                0.0
            } else {
                (m[r * vec_len + k] - m[l * vec_len + k]) / w_total
            };
            for j in l..r {
                data[j * vec_len + k] = mean;
            }
        }
        if l == 0 {
            break;
        }
        r = l;
    }
}
