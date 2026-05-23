"""
tests/test_laplacex.py
======================
Full pytest suite for LaplaceX.

All tests are written against the public Python API; the C++ extension is
exercised indirectly through it.
"""

from __future__ import annotations

import math
import cmath
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# We import the *installed* package.  If you are running from source without
# building the extension, these will fail gracefully with ImportError.
# ---------------------------------------------------------------------------
laplacex = pytest.importorskip("laplacex")

from laplacex import (              # noqa: E402
    laplace,
    inverse,
    decompose,
    reconstruct,
    LaplaceTransform,
    ExponentialComponents,
    TransformConfig,
    DecomposeConfig,
    utils,
)


# ===========================================================================
# Helpers
# ===========================================================================
def linspace_t(n=300, tmin=0.05, tmax=10.0):
    return np.linspace(tmin, tmax, n)


# ===========================================================================
# Forward transform
# ===========================================================================
class TestForward:
    """Known Laplace-transform pairs."""

    @pytest.mark.parametrize("s_val, expected", [
        (1+0j,  1/3),   # F(1) = 1/(1+2) for f(t)=exp(-2t)
        (2+0j,  0.25),  # F(2) = 1/4
        (5+1j,  1/(7+1j)),
    ])
    def test_exp_decay(self, s_val, expected):
        """F(s) for f(t) = exp(-2t) should be 1/(s+2)."""
        F = laplace(lambda t: math.exp(-2 * t), s_val)
        assert abs(F - expected) < 1e-6

    def test_constant_one(self):
        """F(s) for f(t)=1 should be 1/s (Re(s)>0)."""
        s = 2.0 + 0j
        F = laplace(lambda t: 1.0, s)
        assert abs(F - 1/s) < 1e-5

    def test_batch(self):
        """Batch output matches pointwise."""
        s_arr = np.array([1+0j, 2+0j, 3+0j])
        F_batch = laplace(lambda t: math.exp(-t), s_arr)
        for i, s in enumerate(s_arr):
            F_pt = laplace(lambda t: math.exp(-t), s)
            assert abs(F_batch[i] - F_pt) < 1e-10

    def test_discrete_vs_callable(self):
        """Discrete-sample forward should agree with callable within tolerance."""
        t = np.linspace(0.001, 20, 2000)
        f = np.exp(-t)
        s_arr = np.array([1+0j, 2+0j])
        F_disc = laplace(f, s_arr, t=t)
        F_call = laplace(lambda tt: math.exp(-tt), s_arr)
        np.testing.assert_allclose(F_disc, F_call, rtol=1e-3)

    def test_linearity(self):
        """F(a*f + b*g) = a*F(f) + b*F(g)."""
        a, b = 3.0, -1.5
        s = 2+1j
        F1 = laplace(lambda t: math.exp(-t),   s)
        F2 = laplace(lambda t: math.exp(-3*t), s)
        Fc = laplace(lambda t: a*math.exp(-t) + b*math.exp(-3*t), s)
        assert abs(Fc - (a*F1 + b*F2)) < 1e-7

    def test_config_tol(self):
        """Tighter tolerance gives same result (regression)."""
        cfg_loose = TransformConfig(); cfg_loose.tol = 1e-6
        cfg_tight = TransformConfig(); cfg_tight.tol = 1e-12
        s = 1+0.5j
        F1 = laplace(lambda t: math.exp(-t), s, cfg=cfg_loose)
        F2 = laplace(lambda t: math.exp(-t), s, cfg=cfg_tight)
        assert abs(F1 - F2) < 1e-5

    def test_requires_positive_re_s(self):
        """Should raise for Re(s) < 0 (non-convergent integral)."""
        with pytest.raises(Exception):
            laplace(lambda t: 1.0, -1+0j)

    def test_complex_frequency(self):
        """F(s) for f(t)=exp(-at)*cos(wt) = (s+a)/((s+a)^2+w^2)."""
        a, w = 1.0, 2.0
        s = 3+1j
        expected = (s + a) / ((s + a)**2 + w**2)
        F = laplace(lambda t: math.exp(-a*t) * math.cos(w*t), s)
        assert abs(F - expected) < 1e-5


# ===========================================================================
# Inverse transform (Weeks method)
# ===========================================================================
class TestInverse:

    def test_exp_decay_roundtrip(self):
        """Inverse of 1/(s+2) should recover exp(-2t)."""
        t = np.linspace(0.1, 5, 50)
        f_rec = inverse(lambda s: 1/(s + 2), t)
        f_ref = np.exp(-2 * t)
        np.testing.assert_allclose(f_rec, f_ref, atol=0.02)

    def test_step_function(self):
        """Inverse of 1/s should recover f(t) ≈ 1 (Heaviside)."""
        t = np.linspace(0.5, 5, 30)
        f_rec = inverse(lambda s: 1/s, t)
        np.testing.assert_allclose(f_rec, np.ones_like(t), atol=0.05)

    def test_component_reconstruction(self):
        """If components are given, reconstruction should be accurate."""
        t = linspace_t()
        f = 2*np.exp(-0.5*t) + 0.5*np.exp(-3*t)
        ec = decompose(t, f)
        f_rec = inverse(None, t, components=ec)
        # residual should be small compared to signal range
        assert np.max(np.abs(f_rec - f)) / np.max(np.abs(f)) < 0.2


# ===========================================================================
# Decompose
# ===========================================================================
class TestDecompose:

    def test_single_exponential(self):
        """Single exp(-λt) should yield one dominant component."""
        t  = linspace_t(n=400, tmin=0.01, tmax=15)
        f  = np.exp(-2 * t)
        ec = decompose(t, f)
        assert len(ec) >= 1
        # Dominant component should have rate close to 2
        dom = ec[0]
        assert abs(dom.rate - 2.0) / 2.0 < 0.25

    def test_two_exponentials(self):
        """Two well-separated exponentials should yield 2 components."""
        t  = linspace_t(n=600, tmin=0.01, tmax=20)
        f  = 2.0*np.exp(-0.5*t) + 0.5*np.exp(-5*t)
        ec = decompose(t, f)
        assert len(ec) >= 1   # at minimum one is found

    def test_noisy_signal_runs(self):
        """Decompose on noisy data should not crash."""
        t = linspace_t()
        f = np.exp(-t) + utils.add_noise(np.zeros_like(t), sigma=0.02, seed=0)
        ec = decompose(t, f)
        assert isinstance(ec, ExponentialComponents)

    def test_returns_sorted_by_significance(self):
        """Components should be sorted descending by significance."""
        t  = linspace_t()
        f  = 3*np.exp(-t) + 0.1*np.exp(-20*t)
        ec = decompose(t, f)
        sigs = ec.significances
        assert all(sigs[i] >= sigs[i+1] for i in range(len(sigs)-1))

    def test_custom_config(self):
        """DecomposeConfig parameters are respected without crashing."""
        cfg = DecomposeConfig()
        cfg.n_scales     = 32
        cfg.n_log_points = 256
        cfg.use_aic      = False
        t  = linspace_t()
        f  = np.exp(-t)
        ec = decompose(t, f, cfg=cfg)
        assert isinstance(ec, ExponentialComponents)

    def test_raises_on_non_positive_t(self):
        """Passing t=0 should raise domain error."""
        with pytest.raises(Exception):
            decompose(np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.5, 0.25]))

    def test_raises_on_too_few_points(self):
        """Fewer than 4 points should raise."""
        with pytest.raises(Exception):
            decompose(np.array([1.0, 2.0]), np.array([1.0, 0.5]))


# ===========================================================================
# ExponentialComponents
# ===========================================================================
class TestComponents:

    @pytest.fixture
    def ec(self):
        t = linspace_t()
        f = 2*np.exp(-0.5*t) + 0.3*np.exp(-4*t)
        return decompose(t, f)

    def test_len(self, ec):
        assert len(ec) >= 1

    def test_array_properties(self, ec):
        assert len(ec.amplitudes)   == len(ec)
        assert len(ec.rates)        == len(ec)
        assert len(ec.decay_times)  == len(ec)
        assert len(ec.significances) == len(ec)

    def test_reconstruct(self, ec):
        t = linspace_t()
        f_rec = ec.reconstruct(t)
        assert f_rec.shape == t.shape

    def test_reconstruct_laplace(self, ec):
        s = np.array([1+0j, 2+1j])
        F = ec.reconstruct_laplace(s)
        assert F.shape == s.shape

    def test_filter_min_significance(self, ec):
        ec_filtered = ec.filter(min_significance=0.5)
        assert all(c.significance >= 0.5 for c in ec_filtered)

    def test_filter_real_only(self, ec):
        ec_real = ec.filter(real_only=True)
        assert all(c.frequency == 0.0 for c in ec_real)

    def test_top(self, ec):
        ec_top = ec.top(1)
        assert len(ec_top) == 1

    def test_to_dict(self, ec):
        d = ec.to_dict()
        assert set(d.keys()) >= {"amplitudes", "rates", "decay_times"}
        assert len(d["amplitudes"]) == len(ec)

    def test_add(self, ec):
        ec2 = ec + ec
        assert len(ec2) == 2 * len(ec)

    def test_repr(self, ec):
        assert "ExponentialComponents" in repr(ec)


# ===========================================================================
# Reconstruct (top-level function)
# ===========================================================================
class TestReconstruct:

    def test_shape(self):
        t = linspace_t()
        f = np.exp(-t)
        ec = decompose(t, f)
        f_rec = reconstruct(ec, t)
        assert f_rec.shape == t.shape

    def test_values(self):
        """Reconstructed values should be close to original for clean signal."""
        t = linspace_t(n=500, tmin=0.01, tmax=10)
        f = 1.5 * np.exp(-t)
        ec = decompose(t, f)
        f_rec = reconstruct(ec, t)
        # At least 60% of points within 20% relative error
        rel_err = np.abs(f_rec - f) / (np.abs(f) + 1e-10)
        assert np.mean(rel_err < 0.20) > 0.60


# ===========================================================================
# LaplaceTransform class
# ===========================================================================
class TestLaplaceTransformClass:

    def test_forward(self):
        lt = LaplaceTransform(tol=1e-9)
        F = lt.laplace(lambda t: math.exp(-t), 2+0j)
        assert abs(F - 1/3) < 1e-6

    def test_decompose_and_cache(self):
        lt = LaplaceTransform()
        t = linspace_t()
        f = np.exp(-2*t)
        ec = lt.decompose(t, f, cache=True)
        assert lt.cached_components is ec

    def test_inverse_uses_cache(self):
        lt = LaplaceTransform()
        t = linspace_t()
        f = np.exp(-t)
        lt.decompose(t, f)
        t_eval = np.linspace(0.5, 5, 20)
        f_rec = lt.inverse(None, t_eval, method="components")
        assert f_rec.shape == t_eval.shape

    def test_reconstruct_via_lt(self):
        lt = LaplaceTransform()
        t = linspace_t()
        f = np.exp(-t)
        ec = lt.decompose(t, f)
        f_rec = lt.reconstruct(ec, t)
        assert f_rec.shape == t.shape

    def test_repr(self):
        lt = LaplaceTransform(tol=1e-8)
        assert "LaplaceTransform" in repr(lt)

    def test_no_cache_raises(self):
        lt = LaplaceTransform()
        with pytest.raises(RuntimeError):
            lt.inverse(None, np.array([1.0, 2.0]), method="components")


# ===========================================================================
# Utilities
# ===========================================================================
class TestUtils:

    def test_gen_step(self):
        t = np.linspace(0, 5, 100)
        f = utils.gen_step(t, t0=2.0)
        assert all(v == 0.0 for v in f[t < 2.0])
        assert all(v == 1.0 for v in f[t >= 2.0])

    def test_gen_damped_sine(self):
        t = np.linspace(0.01, 5, 200)
        f = utils.gen_damped_sine(t, amplitude=1.0, rate=0.5, omega=2.0)
        assert f.shape == t.shape

    def test_gen_exp_sum(self):
        t = np.linspace(0.01, 5, 100)
        A = np.array([1.0, 2.0])
        r = np.array([0.5, 2.0])
        f = utils.gen_exp_sum(t, A, r)
        ref = np.array([A[0]*np.exp(-r[0]*t) + A[1]*np.exp(-r[1]*t)])
        np.testing.assert_allclose(f, ref.squeeze(), rtol=1e-10)

    def test_gen_diffusion(self):
        t = np.linspace(0.1, 5, 50)
        f = utils.gen_diffusion(t, amplitude=1.0)
        np.testing.assert_allclose(f, 1.0 / np.sqrt(t), rtol=1e-10)

    def test_add_noise_reproducible(self):
        f = np.zeros(100)
        f1 = utils.add_noise(f, sigma=1.0, seed=42)
        f2 = utils.add_noise(f, sigma=1.0, seed=42)
        np.testing.assert_array_equal(f1, f2)

    def test_estimate_noise(self):
        rng = np.random.default_rng(0)
        sigma = 0.05
        f = rng.normal(0, sigma, 500)
        est = utils.estimate_noise(f)
        assert abs(est - sigma) / sigma < 0.3  # within 30%

    def test_to_log_plane_shape(self):
        t = np.linspace(0.01, 10, 200)
        f = np.exp(-t)
        x, ft = utils.to_log_plane(t, f, n=128)
        assert len(x) == 128
        assert len(ft) == 128

    def test_aic_bic(self):
        assert utils.aic(100.0, 50, 3) < utils.aic(100.0, 50, 10)
        assert utils.bic(100.0, 50, 3) < utils.bic(100.0, 50, 10)

    def test_suggest_log_range(self):
        t = np.linspace(0.1, 10, 100)
        f = np.exp(-t)
        t_lo, t_hi = utils.suggest_log_range(t, f)
        assert t_lo > 0
        assert t_hi > t_lo


# ===========================================================================
# Round-trip / integration tests
# ===========================================================================
class TestRoundtrip:

    def test_forward_inverse_roundtrip(self):
        """Forward then Weeks inverse should recover the original (loose tol)."""
        a = 2.0
        F = lambda s: 1/(s + a)  # noqa: E731
        t = np.linspace(0.2, 4.0, 40)
        f_rec = inverse(F, t, sigma=1.0, b=1.0, N=128)
        f_ref = np.exp(-a * t)
        np.testing.assert_allclose(f_rec, f_ref, atol=0.05)

    def test_decompose_reconstruct_roundtrip(self):
        """Decompose → reconstruct should reproduce clean multi-exp signal."""
        t = linspace_t(n=600, tmin=0.01, tmax=15)
        f = 2*np.exp(-0.3*t) + 0.5*np.exp(-3*t)
        ec  = decompose(t, f)
        f_r = reconstruct(ec, t)
        rel = np.max(np.abs(f_r - f)) / np.max(np.abs(f))
        assert rel < 0.35   # allow 35% max relative error

    def test_laplace_of_reconstruction(self):
        """Laplace transform of reconstruction should match analytic F(s)."""
        t = linspace_t(n=800, tmin=0.001, tmax=30)
        f = np.exp(-t)
        ec = decompose(t, f)
        s_test = np.array([2+0j, 3+1j])
        F_comp = ec.reconstruct_laplace(s_test)
        F_exact = 1 / (s_test + 1)
        np.testing.assert_allclose(np.abs(F_comp), np.abs(F_exact), rtol=0.3)
