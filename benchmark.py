#!/usr/bin/env python3
"""
Runtime comparison: original Java pottslab  vs.  Python/Rust port.

The Java column shows the timing of the *same algorithm* as it ran when
called from MATLAB (the Java classes are the performance-critical layer the
original pottslab MATLAB code delegated to).  The Rust column is our port.
A pure-Python reference column is included where the algorithm is simple
enough to implement without the Java/Rust extension.

Run:
    python benchmark.py
"""

import subprocess, sys, time, re
import numpy as np

sys.path.insert(0, ".")
import pottslab as pl

JAVA_CLASSPATH = "Java/bin"

# ─────────────────────────────────────────────────────────────────────────────
# Run the Java benchmark and parse its output
# ─────────────────────────────────────────────────────────────────────────────

def run_java_benchmark():
    """Compile (if needed) and run the Java benchmark; return raw stdout."""
    import pathlib, os
    root = pathlib.Path(__file__).parent
    src_files = list((root / "Java/src/pottslab").glob("*.java"))
    bin_dir = root / JAVA_CLASSPATH
    bin_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["javac", "-d", str(bin_dir)] + [str(p) for p in src_files],
        capture_output=True, check=True, cwd=str(root),
    )
    result = subprocess.run(
        ["java", "-cp", str(bin_dir), "pottslab.Benchmark"],
        capture_output=True, text=True, check=True, cwd=str(root),
    )
    return result.stdout


def parse_java_output(raw: str) -> dict:
    """Parse Java benchmark stdout into {section_title: {n_label: ms}}."""
    sections = {}
    current = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("-") or \
                line.startswith("Java") or line.startswith("Median"):
            continue
        if line.startswith("n") and "Java" in line:
            continue
        # New section header (no digits at start)
        if not re.match(r'^[\d×]', line):
            current = line
            sections[current] = {}
            continue
        # Data row: "100    0.24"
        m = re.match(r'^(\S+)\s+([\d.]+)', line)
        if m and current is not None:
            sections[current][m.group(1)] = float(m.group(2))
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Python reference implementations (baselines)
# ─────────────────────────────────────────────────────────────────────────────

def py_l2_potts(f, gamma):
    n = len(f)
    cs  = np.zeros(n + 1);  cs[1:]  = np.cumsum(f)
    css = np.zeros(n + 1);  css[1:] = np.cumsum(f * f)
    arr_p = np.full(n + 1, np.inf);  arr_p[0] = -gamma
    arr_j = np.zeros(n + 1, dtype=np.intp)
    for r in range(1, n + 1):
        lens  = np.arange(1, r + 1)
        sums  = cs[r]  - cs[:r]
        sums2 = css[r] - css[:r]
        costs = sums2 - sums * sums / lens
        best  = np.argmin(arr_p[:r] + gamma + costs)
        arr_p[r] = arr_p[best] + gamma + costs[best]
        arr_j[r] = best
    u = np.empty(n);  r = n
    while r > 0:
        l = arr_j[r];  u[l:r] = (cs[r] - cs[l]) / (r - l);  r = l
    return u


def py_l2_potts_vv(f, gamma):
    n, c = f.shape
    cs  = np.zeros((n + 1, c));  cs[1:]  = np.cumsum(f, axis=0)
    css = np.zeros(n + 1);       css[1:] = np.cumsum(np.sum(f * f, axis=1))
    arr_p = np.full(n + 1, np.inf);  arr_p[0] = -gamma
    arr_j = np.zeros(n + 1, dtype=np.intp)
    for r in range(1, n + 1):
        lens  = np.arange(1, r + 1, dtype=float)
        sums  = cs[r] - cs[:r]
        costs = css[r] - css[:r] - np.sum(sums * sums, axis=1) / lens
        best  = np.argmin(arr_p[:r] + gamma + costs)
        arr_p[r] = arr_p[best] + gamma + costs[best]
        arr_j[r] = best
    u = np.empty_like(f);  r = n
    while r > 0:
        l = arr_j[r];  u[l:r] = (cs[r] - cs[l]) / (r - l);  r = l
    return u


def py_l1_potts(f, gamma):
    """O(n³) naive — np.median per segment."""
    n = len(f);  b = np.full(n + 1, np.inf);  b[0] = -gamma
    part = np.zeros(n + 1, dtype=np.intp)
    for r in range(1, n + 1):
        for l in range(r):
            med = np.median(f[l:r])
            cand = b[l] + gamma + float(np.sum(np.abs(f[l:r] - med)))
            if cand < b[r]:
                b[r] = cand;  part[r] = l
    u = np.empty(n);  r = n
    while r > 0:
        l = part[r];  u[l:r] = np.median(f[l:r]);  r = l
    return u


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _timeit(fn, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3          # ms


def _time_or_skip(fn, skip_ms=15_000, repeats=5):
    t0 = time.perf_counter()
    fn()
    probe = (time.perf_counter() - t0) * 1e3
    if probe > skip_ms:
        return None
    return _timeit(fn, repeats)


def _fmt(ms):
    if ms is None:
        return ">15s"
    return f"{ms:.2f}"


def _speedup(java_ms, rust_ms):
    if java_ms is None or rust_ms is None or rust_ms == 0:
        return "—"
    return f"{java_ms / rust_ms:.1f}×"


# ─────────────────────────────────────────────────────────────────────────────
# Table helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(title, headers, rows):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print()
    print(title)
    print("=" * len(title))
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def bench_l2_1d(java):
    sizes   = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    rows = []
    rng = np.random.default_rng(0)
    for n in sizes:
        f = rng.standard_normal(n);  gamma = 1.0
        py_ms   = _time_or_skip(lambda: py_l2_potts(f, gamma))
        rust_ms = _timeit(lambda: pl.min_l2_potts(f, gamma))
        java_ms = java.get(str(n))
        rows.append((n, _fmt(py_ms), _fmt(java_ms), f"{rust_ms:.2f}",
                     _speedup(java_ms, rust_ms)))
    return rows


def bench_l2_vv(java):
    sizes = [100, 500, 1_000, 5_000, 10_000, 50_000]
    rows = []
    rng = np.random.default_rng(1)
    for n in sizes:
        f = rng.standard_normal((n, 3));  gamma = 1.0
        py_ms   = _time_or_skip(lambda: py_l2_potts_vv(f, gamma))
        rust_ms = _timeit(lambda: pl.min_l2_potts(f, gamma))
        java_ms = java.get(str(n))
        rows.append((n, _fmt(py_ms), _fmt(java_ms), f"{rust_ms:.2f}",
                     _speedup(java_ms, rust_ms)))
    return rows


def bench_l1_1d(java):
    sizes_py = [50, 100, 200, 500]
    sizes_rs = [50, 100, 200, 500, 1_000, 5_000, 10_000]
    rng = np.random.default_rng(2)
    py_t = {}
    for n in sizes_py:
        f = rng.standard_normal(n)
        py_t[n] = _time_or_skip(lambda: py_l1_potts(f, 1.0), skip_ms=30_000)
    rs_t = {}
    for n in sizes_rs:
        f = rng.standard_normal(n)
        rs_t[n] = _timeit(lambda: pl.min_l1_potts(f, 1.0))
    rows = []
    for n in sizes_rs:
        py_ms   = py_t.get(n)
        rust_ms = rs_t[n]
        java_ms = java.get(str(n))
        rows.append((n, _fmt(py_ms), _fmt(java_ms), f"{rust_ms:.2f}",
                     _speedup(java_ms, rust_ms)))
    return rows


def bench_2d(java):
    sizes = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
    rows = []
    rng = np.random.default_rng(3)
    for h, w in sizes:
        f = rng.standard_normal((h, w)).clip(0, 1);  gamma = 0.05
        label = f"{h}×{w}"
        rust_ms = _timeit(
            lambda: pl.min_l2_potts_2d(f, gamma, isotropic=False,
                                        tol=1e-4, quantize=False),
            repeats=3)
        java_ms = java.get(label)
        rows.append((label, _fmt(java_ms), f"{rust_ms:.2f}",
                     _speedup(java_ms, rust_ms)))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Compiling and running Java benchmark …", flush=True)
    java_raw = run_java_benchmark()
    java_sections = parse_java_output(java_raw)

    # Map section names to the dicts we need
    j_l2_1d = java_sections.get("minL2Potts  (1-D scalar)", {})
    j_l2_vv = java_sections.get("minL2Potts  (vector-valued, 3 channels)", {})
    j_l1_1d = java_sections.get(
        "minL1Potts  (1-D scalar, DP with IndexedLinkedHistogram)", {})
    j_2d    = java_sections.get("minL2PottsADMM4  (4-connected, tol=1e-4)", {})

    print()
    print("pottslab: Java (original)  vs.  Python/Rust port")
    print("All times in milliseconds (median of 5 runs; 2-D: 3 runs).")
    print("Python = pure-NumPy reference implementation (same algorithm).")
    print("Java   = original pottslab Java classes, as called by MATLAB.")
    print("Rust   = this port's compiled Rust extension.")
    print("Speedup = Java ÷ Rust.")

    r = bench_l2_1d(j_l2_1d)
    _print_table(
        "min_l2_potts  (1-D scalar)",
        ["n", "Python (ms)", "Java (ms)", "Rust (ms)", "Java/Rust"],
        r)

    r = bench_l2_vv(j_l2_vv)
    _print_table(
        "min_l2_potts  (vector-valued, 3 channels)",
        ["n", "Python (ms)", "Java (ms)", "Rust (ms)", "Java/Rust"],
        r)

    r = bench_l1_1d(j_l1_1d)
    _print_table(
        "min_l1_potts  (1-D scalar)  [Python O(n³), Java & Rust O(n²)]",
        ["n", "Python (ms)", "Java (ms)", "Rust (ms)", "Java/Rust"],
        r)

    r = bench_2d(j_2d)
    _print_table(
        "min_l2_potts_2d  (4-connected ADMM, tol=1e-4)",
        ["size", "Java (ms)", "Rust (ms)", "Java/Rust"],
        r)

    print()
    print("Notes")
    print("─────")
    print("• The Java column reproduces what MATLAB delegated to for performance.")
    print("  MATLAB's own script overhead (loop dispatch, MEX bridge) is NOT")
    print("  included — these are raw Java timings from a standalone harness.")
    print("• L2-Potts scalar: both Java and Rust use the same O(n²) DP with")
    print("  early-termination; Rust removes JVM/GC overhead and uses SIMD.")
    print("• L1-Potts: Java uses IndexedLinkedHistogram O(n²); so does Rust.")
    print("  Speedup comes from compiled-Rust vs. JVM overhead + GC pauses.")
    print("• 2-D ADMM: Java uses multi-threaded PLProcessor (ExecutorService);")
    print("  Rust uses Rayon. Speedup primarily from lower per-task overhead.")
    print("• Add the MATLAB script layer (loop dispatch, MEX bridge, array")
    print("  copies) and the effective speedup from the user's perspective")
    print("  would be even larger.")


if __name__ == "__main__":
    main()
