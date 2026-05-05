// Ported from Java/src/pottslab/IndexedLinkedHistogram.java (Storath & Weinmann)
// by Claude Sonnet coding agent, Anthropic, 2026.
//
// Computes L1-Potts deviations via an indexed linked histogram —
// a sorted doubly-linked list that supports efficient temporary removal
// of elements and median tracking.
//
// Used in the L1-Potts DP:
//   min_u  gamma * ||Du||_0  +  ||u - f||_1  (scalar only)

/// A node in the sorted histogram list.
#[derive(Clone)]
struct HistNode {
    value: f64,
    weight: f64,
    count: usize,
    // Temporary fields mirror the original Java "Temp" variants.
    weight_temp: f64,
    count_temp: usize,
    // Linked list pointers (indices into the arena, usize::MAX = null).
    next: usize,
    prev: usize,
    next_temp: usize,
    prev_temp: usize,
}

impl HistNode {
    fn new(value: f64, weight: f64) -> Self {
        HistNode {
            value,
            weight,
            count: 1,
            weight_temp: weight,
            count_temp: 1,
            next: usize::MAX,
            prev: usize::MAX,
            next_temp: usize::MAX,
            prev_temp: usize::MAX,
        }
    }

    fn reset_temp(&mut self) {
        self.next_temp = self.next;
        self.prev_temp = self.prev;
        self.weight_temp = self.weight;
        self.count_temp = self.count;
    }
}

/// Arena-based indexed linked histogram.
/// Mirrors IndexedLinkedHistogram.java.
pub struct IndexedLinkedHistogram {
    arena: Vec<HistNode>,
    original_order: Vec<usize>, // indices into arena, in insertion order
    weights: Vec<f64>,
    first: usize,
    last: usize,
    median: usize,
    total_weight: f64,
    weight_above_median: f64,
    weight_below_median: f64,
    total_deviation: f64,
}

const NULL: usize = usize::MAX;

impl IndexedLinkedHistogram {
    pub fn new(weights: Vec<f64>) -> Self {
        IndexedLinkedHistogram {
            arena: Vec::with_capacity(weights.len()),
            original_order: Vec::with_capacity(weights.len()),
            weights,
            first: NULL,
            last: NULL,
            median: NULL,
            total_weight: 0.0,
            weight_above_median: 0.0,
            weight_below_median: 0.0,
            total_deviation: 0.0,
        }
    }

    pub fn size(&self) -> usize {
        self.original_order.len()
    }

    /// Insert an element into the sorted list (ascending order).
    /// Mirrors IndexedLinkedHistogram.insertSorted().
    pub fn insert_sorted(&mut self, elem: f64) {
        let weight = self.weights[self.size()];

        if self.first == NULL {
            // Empty list
            let idx = self.alloc_node(elem, weight);
            self.first = idx;
            self.last = idx;
            self.median = idx;
            self.total_deviation = 0.0;
            self.weight_above_median = 0.0;
            self.weight_below_median = 0.0;
            self.total_weight = weight;
            self.original_order.push(idx);
        } else {
            // Find insertion point
            let mut iter = self.first;
            // Reset temp pointers on nodes we pass
            while iter != NULL {
                let nv = self.arena[iter].value;
                if nv >= elem {
                    break;
                }
                self.arena[iter].reset_temp();
                iter = self.arena[iter].next;
            }

            let pivot = if iter != NULL {
                if (self.arena[iter].value - elem).abs() < f64::EPSILON * elem.abs().max(1.0)
                    && self.arena[iter].value == elem
                {
                    // Same value — add weight to existing node
                    self.arena[iter].weight += weight;
                    self.arena[iter].count += 1;
                    iter
                } else {
                    let new_idx = self.alloc_node(elem, weight);
                    self.insert_before(iter, new_idx);
                    new_idx
                }
            } else {
                let new_idx = self.alloc_node(elem, weight);
                self.insert_after(self.last, new_idx);
                new_idx
            };

            // Continue resetting temp pointers
            let mut it2 = if iter != NULL { self.arena[iter].next } else { NULL };
            while it2 != NULL {
                self.arena[it2].reset_temp();
                it2 = self.arena[it2].next;
            }
            // Also reset pivot's temp if we re-used a node
            self.arena[pivot].reset_temp();

            self.total_weight += weight;
            self.original_order.push(pivot);
        }

        // Recompute median from scratch (matches the Java implementation after the
        // incremental approach was marked OBSOLETE due to loss of significance).
        self.recompute_median();
    }

    fn recompute_median(&mut self) {
        let half = self.total_weight / 2.0;
        let mut wbm = 0.0f64;
        let mut iter = self.first;
        // Walk sorted list; the weighted median is the first element where
        // cumulative weight >= half (lower-median convention).
        self.median = self.first;
        while iter != NULL {
            wbm += self.arena[iter].weight;
            if wbm >= half {
                self.median = iter;
                break;
            }
            iter = self.arena[iter].next;
        }
        // Fallback: empty list or all-zero-weight elements
        if self.median == NULL {
            self.median = self.first;
        }

        // Recompute weights and deviation
        let med_val = self.arena[self.median].value;
        self.weight_above_median = 0.0;
        self.weight_below_median = 0.0;
        self.total_deviation = 0.0;
        iter = self.first;
        while iter != NULL {
            let v = self.arena[iter].value;
            let w = self.arena[iter].weight;
            if v < med_val {
                self.weight_below_median += w;
            } else if v > med_val {
                self.weight_above_median += w;
            }
            self.total_deviation += (med_val - v).abs() * w;
            iter = self.arena[iter].next;
        }
    }

    /// Compute the L1-deviations d*[l,r] for l=1..size (outer loop removes l=1,2,...).
    /// Writes results into `out[0..size]`; `out[size-1]` = 0 (single-element deviation).
    /// Mirrors IndexedLinkedHistogram.computeDeviations().
    ///
    /// Uses &mut self to avoid cloning the arena: insert_sorted already resets all
    /// temp fields on every call, so the temp state is always clean at entry.
    /// Accepts a pre-allocated slice to avoid Vec allocation on every DP step.
    pub fn compute_deviations_into(&mut self, out: &mut [f64]) {
        let n = self.size();

        let mut median_temp = self.median;
        let mut deviation_temp = self.total_deviation;
        let mut weight_above_temp = self.weight_above_median;
        let mut weight_below_temp = self.weight_below_median;
        let mut total_weight_temp = self.total_weight;
        let mut first_t = self.first;
        let mut last_t = self.last;

        // No arena clone needed: temp fields on each node were reset by the
        // preceding insert_sorted call, so we operate on self.arena directly.

        if n > 0 { out[n - 1] = 0.0; }  // single-element deviation = 0
        for l in 0..(n - 1) {
            out[l] = deviation_temp;

            let node_to_remove = self.original_order[l];
            let weight_to_remove = self.weights[l];

            // Remove weight from node
            self.arena[node_to_remove].weight_temp -= weight_to_remove;
            if self.arena[node_to_remove].count_temp > 0 {
                self.arena[node_to_remove].count_temp -= 1;
            }
            if self.arena[node_to_remove].weight_temp < 0.0 {
                self.arena[node_to_remove].weight_temp = 0.0;
            }

            // Update deviation
            let nrv = self.arena[node_to_remove].value;
            let med_val = self.arena[median_temp].value;
            deviation_temp -= weight_to_remove * (nrv - med_val).abs();
            total_weight_temp -= weight_to_remove;

            if nrv > med_val {
                weight_above_temp -= weight_to_remove;
            } else if nrv < med_val {
                weight_below_temp -= weight_to_remove;
            }

            let twth = total_weight_temp / 2.0;

            // Shift median right if weight above is too large
            while weight_above_temp > twth {
                let next_med = self.arena[median_temp].next_temp;
                if next_med == NULL { break; }
                let old_med_val = self.arena[median_temp].value;
                let med_wt = self.arena[median_temp].weight_temp;
                weight_below_temp += med_wt;
                median_temp = next_med;
                let med_diff = (old_med_val - self.arena[median_temp].value).abs();
                deviation_temp -= med_diff * (weight_below_temp - weight_above_temp).abs();
                weight_above_temp -= self.arena[median_temp].weight_temp;
            }

            // Shift median left if weight below is too large
            while weight_below_temp > twth {
                let prev_med = self.arena[median_temp].prev_temp;
                if prev_med == NULL { break; }
                let old_med_val = self.arena[median_temp].value;
                let med_wt = self.arena[median_temp].weight_temp;
                weight_above_temp += med_wt;
                median_temp = prev_med;
                let med_diff = (old_med_val - self.arena[median_temp].value).abs();
                deviation_temp -= med_diff * (weight_above_temp - weight_below_temp).abs();
                weight_below_temp -= self.arena[median_temp].weight_temp;
            }

            // Remove node temporarily from list if empty
            if self.arena[node_to_remove].count_temp == 0 {
                let prev = self.arena[node_to_remove].prev_temp;
                let next = self.arena[node_to_remove].next_temp;
                if node_to_remove == last_t {
                    last_t = prev;
                    if prev != NULL { self.arena[prev].next_temp = NULL; }
                } else if node_to_remove == first_t {
                    first_t = next;
                    if next != NULL { self.arena[next].prev_temp = NULL; }
                } else {
                    if next != NULL { self.arena[next].prev_temp = prev; }
                    if prev != NULL { self.arena[prev].next_temp = next; }
                }
            }
        }

    }

    fn alloc_node(&mut self, value: f64, weight: f64) -> usize {
        let idx = self.arena.len();
        self.arena.push(HistNode::new(value, weight));
        idx
    }

    fn insert_before(&mut self, pivot: usize, new_idx: usize) {
        let prev_of_pivot = self.arena[pivot].prev;
        self.arena[new_idx].next = pivot;
        self.arena[new_idx].prev = prev_of_pivot;
        self.arena[pivot].prev = new_idx;
        if pivot == self.first {
            self.first = new_idx;
        } else {
            self.arena[prev_of_pivot].next = new_idx;
            self.arena[prev_of_pivot].reset_temp();
        }
        self.arena[new_idx].reset_temp();
        self.arena[pivot].reset_temp();
    }

    fn insert_after(&mut self, pivot: usize, new_idx: usize) {
        let next_of_pivot = self.arena[pivot].next;
        self.arena[new_idx].prev = pivot;
        self.arena[new_idx].next = next_of_pivot;
        self.arena[pivot].next = new_idx;
        if pivot == self.last {
            self.last = new_idx;
        } else {
            self.arena[next_of_pivot].prev = new_idx;
            self.arena[next_of_pivot].reset_temp();
        }
        self.arena[pivot].reset_temp();
        self.arena[new_idx].reset_temp();
    }
}


/// Solve the 1D L1-Potts problem in-place (scalar signals only).
///
/// Uses the IndexedLinkedHistogram to compute the O(n²) DP that minimises
///   gamma * ||Du||_0  +  ||u - f||_1  (weighted)
pub fn solve_l1_potts(data: &mut [f64], weights: Option<&[f64]>, gamma: f64) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let default_weights: Vec<f64> = vec![1.0; n];
    let w: &[f64] = weights.unwrap_or(&default_weights);

    // DP arrays
    // B[r] = optimal Potts value for [0..r];  B[0] = -gamma (boundary condition from Matlab)
    let mut b = vec![0.0f64; n + 1];
    b[0] = -gamma;
    let mut partition = vec![0usize; n]; // partition[r] = best left bound index

    let mut hist = IndexedLinkedHistogram::new(w.to_vec());
    // Preallocate deviations buffer once; reused every DP step (eliminates O(n²/2) allocations).
    let mut dev_buf = vec![0.0f64; n + 1];

    for rb in 0..n {
        hist.insert_sorted(data[rb]);
        hist.compute_deviations_into(&mut dev_buf[..=rb]);

        // dev_buf[lb] = deviation of segment [lb..=rb]; dev_buf[rb] = 0 (single element).
        let mut best_val = f64::INFINITY;
        let mut best_lb = 0usize;
        for lb in 0..=rb {
            let candidate = b[lb] + gamma + dev_buf[lb];
            if candidate < best_val {
                best_val = candidate;
                best_lb = lb;
            }
        }
        partition[rb] = best_lb;
        b[rb + 1] = best_val;
    }

    // Reconstruct piecewise-constant solution using weighted median per segment
    // Backtrack partition to find segment boundaries
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut r = n;
    loop {
        let l = partition[r - 1];
        segments.push((l, r));
        if l == 0 {
            break;
        }
        r = l;
    }

    for (l, r) in segments {
        // Weighted median of data[l..r]
        let seg_data = &data[l..r];
        let seg_w = &w[l..r];
        let med = weighted_median(seg_data, seg_w);
        for j in l..r {
            data[j] = med;
        }
    }
}

/// Compute the weighted median of a slice.
pub fn weighted_median(data: &[f64], weights: &[f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return data[0];
    }
    // Sort by value
    let mut pairs: Vec<(f64, f64)> = data.iter().zip(weights.iter()).map(|(&d, &w)| (d, w)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let total_w: f64 = pairs.iter().map(|(_, w)| w).sum();
    let half = total_w / 2.0;
    let mut cumsum = 0.0f64;
    for (v, w) in &pairs {
        cumsum += w;
        if cumsum >= half {
            return *v;
        }
    }
    pairs.last().unwrap().0
}
