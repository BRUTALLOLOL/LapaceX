"""
conftest.py — shared pytest configuration for LaplaceX.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    """Seeded NumPy random generator shared across the test session."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_exp_data():
    """Return (t, f) for f(t) = exp(-t), 300 points over [0.01, 10]."""
    t = np.linspace(0.01, 10, 300)
    return t, np.exp(-t)


@pytest.fixture
def multi_exp_data():
    """Return (t, f) for f(t) = 2*exp(-0.5t) + 0.5*exp(-5t)."""
    t = np.linspace(0.01, 15, 500)
    f = 2.0 * np.exp(-0.5 * t) + 0.5 * np.exp(-5 * t)
    return t, f
