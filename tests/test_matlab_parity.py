# Rust-vs-MATLAB parity tests for Pottslab.
#
# Each test:
#   1. Calls the Rust port (the loaded `_core` extension).
#   2. Calls the MATLAB implementation via the host `matlab` shim.
#   3. Asserts agreement at the per-test tolerance.
#
# Skipped (not failed) when the matlab shim or HOST_MATLAB env are unavailable.
# Hard-fails if the Rust extension is missing — a pure-Python-vs-MATLAB
# comparison would not be a Rust port parity test.
#
# Java note: host MATLAB R2025a ships Java 11 (Amazon Corretto), but the
# committed .class files in Java/bin/ were compiled for Java 17. The session
# fixture below recompiles Java/src/ with --release 11 into a workspace-visible
# temp dir and points MATLAB's javaaddpath there.
#
# L1 vs L2 parity semantics:
#   - L2-Potts has a unique optimum on each segment (the mean), so element-wise
#     parity on `u` is well-defined and tested at rtol=1e-10.
#   - L1-Potts has a non-unique optimum on even-length segments: any μ in
#     [median_low, median_high] minimises Σ|y - μ|. MATLAB averages the two
#     middle values, the Rust port returns the lower one. Both are L1-optimal
#     -> identical Potts energy -> identical optimal partition. L1 parity is
#     therefore tested on (segment list, energy), not on `u`.

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import pottslab as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
JAVA_SRC = REPO_ROOT / "Java" / "src"
WORKSPACE_ROOT = REPO_ROOT.parents[1]


def _shim_available() -> bool:
    return shutil.which("matlab") is not None and bool(os.environ.get("HOST_MATLAB"))


def _rust_extension_loaded() -> bool:
    try:
        import pottslab._core  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not _shim_available(),
        reason="matlab shim not configured (HOST_MATLAB unset or matlab not on PATH)",
    ),
    pytest.mark.skipif(
        not _rust_extension_loaded(),
        reason="pottslab._core not built — would compare MATLAB to pure-Python fallback",
    ),
]


@pytest.fixture(scope="session")
def matlab_parity_env():
    """Build Pottslab Java sources for Java 11 in a workspace-visible temp dir.

    Yields a dict with the absolute path of the resulting class directory so
    each test's generated .m script can `javaaddpath` it. The temp dir lives
    under <devcontainer>/.parity-cache/ so host MATLAB sees the same path.
    """
    cache_root = WORKSPACE_ROOT / ".parity-cache"
    cache_root.mkdir(exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="pottslab-java11-", dir=cache_root))

    sources = [str(p.relative_to(JAVA_SRC)) for p in JAVA_SRC.rglob("*.java")]
    if not sources:
        pytest.skip("no Java sources found under Java/src/")

    cmd = [
        "ssh",
        "host",
        f"cd {JAVA_SRC} && /usr/bin/javac --release 11 -d {out_dir} " + " ".join(sources),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        shutil.rmtree(out_dir, ignore_errors=True)
        pytest.skip(f"javac failed on host:\n{result.stderr}")

    try:
        yield {"java_classpath": str(out_dir)}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _run_matlab_potts(env, f: np.ndarray, gamma: float, kind: str):
    """Invoke `min{L1,L2}Potts(f, gamma)` in host MATLAB and parse outputs.

    For L2 returns the resulting `u` (np.ndarray, shape (n,)).
    For L1 returns a dict with keys {u, partition, n_jumps, data_error, energy}.
    """
    if kind not in ("l1", "l2"):
        raise ValueError(kind)

    work_dir = Path(tempfile.mkdtemp(prefix="pottslab-parity-", dir=WORKSPACE_ROOT))
    try:
        script_path = work_dir / "run_parity.m"
        f_literal = ", ".join(f"{x:.17e}" for x in f)
        if kind == "l2":
            out_path = work_dir / "u.csv"
            body = (
                "u = minL2Potts(f_in, gamma);\n"
                f"writematrix(u, '{out_path}');\n"
            )
        else:
            partition_path = work_dir / "partition.csv"
            energy_path = work_dir / "energy.csv"
            u_path = work_dir / "u.csv"
            body = (
                "partition = findBestPartition(f_in, gamma, 'L1');\n"
                "[u, nJumps] = reconstructionFromPartition(f_in, partition, 'L1');\n"
                "dErr = sum(abs(u - f_in));\n"
                "energy = gamma * nJumps + dErr;\n"
                f"writematrix(partition, '{partition_path}');\n"
                f"writematrix(u, '{u_path}');\n"
                f"writematrix([nJumps; dErr; energy], '{energy_path}');\n"
            )
        script = (
            f"addpath(genpath('{REPO_ROOT}'));\n"
            f"javaaddpath('{env['java_classpath']}');\n"
            f"f_in = [{f_literal}]';\n"
            f"gamma = {gamma:.17e};\n"
            f"{body}"
        )
        script_path.write_text(script)

        result = subprocess.run(
            ["matlab", "-batch", f"addpath('{work_dir}'); run_parity"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MATLAB failed (rc={result.returncode}):\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        if kind == "l2":
            return np.loadtxt(out_path).ravel()

        meta = np.loadtxt(energy_path).ravel()
        return {
            "u": np.loadtxt(u_path).ravel(),
            "partition": np.loadtxt(partition_path).astype(np.int64).ravel(),
            "n_jumps": int(meta[0]),
            "data_error": float(meta[1]),
            "energy": float(meta[2]),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _segments_from_matlab_partition(partition: np.ndarray) -> list:
    """Walk MATLAB findBestPartition output backward to recover active segments.

    MATLAB convention: partition is 1-indexed; partition(rb) = lb means the
    rightmost segment of the optimal [1..rb] sub-problem is (lb, rb]. Sub-problem
    optima at non-active right-bounds are not part of the final partition.
    Returns segments as 0-indexed half-open intervals [start, end).
    """
    n = len(partition)
    segs = []
    rb = n
    while rb > 0:
        lb = int(partition[rb - 1])
        segs.append((lb, rb))
        rb = lb
    segs.reverse()
    return segs


def _segments_from_u(u: np.ndarray) -> list:
    n = len(u)
    segs = []
    start = 0
    for i in range(1, n):
        if u[i] != u[i - 1]:
            segs.append((start, i))
            start = i
    segs.append((start, n))
    return segs


# ---------- L2-Potts (unique optimum -> element-wise parity) ----------

L2_CASES = [
    pytest.param(0, 8, 0.1, id="seed0_n8_g0.1"),
    pytest.param(1, 16, 0.5, id="seed1_n16_g0.5"),
    pytest.param(2, 24, 1.0, id="seed2_n24_g1.0"),
]


@pytest.mark.parametrize("seed,n,gamma", L2_CASES)
def test_l2_potts_random_signal(matlab_parity_env, seed, n, gamma):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    u_rust = np.asarray(pl.min_l2_potts(f, gamma)).ravel()
    u_matlab = _run_matlab_potts(matlab_parity_env, f, gamma, "l2")
    npt.assert_allclose(u_rust, u_matlab, rtol=1e-10, atol=1e-12)


def test_l2_potts_step(matlab_parity_env):
    f = np.concatenate([np.zeros(8), np.full(8, 3.7)])
    u_rust = np.asarray(pl.min_l2_potts(f, 0.5)).ravel()
    u_matlab = _run_matlab_potts(matlab_parity_env, f, 0.5, "l2")
    npt.assert_allclose(u_rust, u_matlab, rtol=1e-10, atol=1e-12)


# ---------- L1-Potts (segment list + energy parity; element values on
# even-length segments are not uniquely defined and intentionally not compared) ----------

L1_CASES = [
    pytest.param(0, 8, 0.3, id="seed0_n8_g0.3"),
    pytest.param(1, 16, 0.5, id="seed1_n16_g0.5"),
    pytest.param(2, 24, 0.7, id="seed2_n24_g0.7"),
]


@pytest.mark.parametrize("seed,n,gamma", L1_CASES)
def test_l1_potts_random_signal(matlab_parity_env, seed, n, gamma):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)

    u_rust, dErr_rust, n_jumps_rust, energy_rust = pl.min_l1_potts(f, gamma)
    u_rust = np.asarray(u_rust).ravel()

    out = _run_matlab_potts(matlab_parity_env, f, gamma, "l1")

    segs_rust = _segments_from_u(u_rust)
    segs_matlab = _segments_from_matlab_partition(out["partition"])
    assert segs_rust == segs_matlab

    assert n_jumps_rust == out["n_jumps"]
    npt.assert_allclose(energy_rust, out["energy"], rtol=1e-10, atol=1e-12)
    npt.assert_allclose(dErr_rust, out["data_error"], rtol=1e-10, atol=1e-12)


def test_l1_potts_step_odd_segments(matlab_parity_env):
    """Odd-length segments make the median unambiguous, so element-wise parity holds."""
    f = np.concatenate([np.zeros(7), np.full(9, 3.7)])
    u_rust = np.asarray(pl.min_l1_potts(f, 0.5)[0]).ravel()
    out = _run_matlab_potts(matlab_parity_env, f, 0.5, "l1")
    npt.assert_allclose(u_rust, out["u"], rtol=1e-10, atol=1e-12)
