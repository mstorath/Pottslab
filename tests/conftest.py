# Test fixtures for pottslab test suite.
# Ported from MATLAB/Java pottslab by Claude Sonnet coding agent, Anthropic, 2026.

import numpy as np
import pytest


@pytest.fixture
def step_signal_1d():
    """Clean step signal: 50 zeros followed by 50 ones."""
    return np.array([0.0] * 50 + [1.0] * 50)


@pytest.fixture
def noisy_step_signal_1d(rng):
    """Step signal with small Gaussian noise (sigma=0.05)."""
    s = np.array([0.0] * 50 + [1.0] * 50)
    return s + rng.normal(0, 0.05, size=s.shape)


@pytest.fixture
def piecewise_const_1d():
    """3-segment piecewise constant: [0]*30 + [2]*40 + [-1]*30."""
    return np.array([0.0] * 30 + [2.0] * 40 + [-1.0] * 30)


@pytest.fixture
def step_image_2d():
    """64x64 grayscale image: left half = 0, right half = 1."""
    img = np.zeros((64, 64), dtype=np.float64)
    img[:, 32:] = 1.0
    return img


@pytest.fixture
def rgb_step_image():
    """64x64x3 RGB image: left half = (0,0,0), right half = (1,0.5,0.2)."""
    img = np.zeros((64, 64, 3), dtype=np.float64)
    img[:, 32:, 0] = 1.0
    img[:, 32:, 1] = 0.5
    img[:, 32:, 2] = 0.2
    return img


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)
