#pragma once

#include <complex>
#include <functional>
#include <vector>
#include <cstdint>

namespace laplacex {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct TransformConfig {
    double tol        = 1e-10;   // absolute tolerance for adaptive quadrature
    double rel_tol    = 1e-8;    // relative tolerance
    int    max_levels = 20;      // maximum adaptive bisection depth
    double t_max      = 1e6;     // upper integration limit (exponentially compressed)
    bool   use_long   = false;   // reserved for long-double specialisation
};

// ---------------------------------------------------------------------------
// Forward transform
// ---------------------------------------------------------------------------

/**
 * Compute F(s) = integral_0^inf f(t) * exp(-s*t) dt
 * using adaptive Gauss-Kronrod quadrature with exponential axis compression.
 *
 * @param f      Callable double -> double representing f(t), t >= 0
 * @param s      Complex frequency
 * @param cfg    Quadrature configuration
 * @return       Complex value F(s)
 */
std::complex<double> forward(
    const std::function<double(double)>& f,
    std::complex<double> s,
    const TransformConfig& cfg = {}
);

/**
 * Compute F(s) for a batch of complex frequencies.
 * Thread-safe: each evaluation is independent.
 */
std::vector<std::complex<double>> forward_batch(
    const std::function<double(double)>& f,
    const std::vector<std::complex<double>>& s_vals,
    const TransformConfig& cfg = {}
);

/**
 * Forward transform from discrete samples (t_i, f_i).
 * Internally constructs a cubic-spline interpolant.
 */
std::vector<std::complex<double>> forward_discrete(
    const std::vector<double>& t,
    const std::vector<double>& f,
    const std::vector<std::complex<double>>& s_vals,
    const TransformConfig& cfg = {}
);

// ---------------------------------------------------------------------------
// Inverse transform – Weeks method (Laguerre expansion)
// ---------------------------------------------------------------------------

/**
 * Approximate the inverse Laplace transform via the Weeks method:
 * expands F(s) in a series of Laguerre polynomials and reconstructs f(t).
 *
 * @param F       Callable complex<double> -> complex<double>
 * @param t_eval  Points at which f(t) is desired
 * @param sigma   Laguerre shift parameter (auto-selected if 0)
 * @param b       Laguerre scale parameter  (auto-selected if 0)
 * @param N       Number of Laguerre terms
 * @return        Real part of the reconstruction at each t
 */
std::vector<double> inverse_weeks(
    const std::function<std::complex<double>(std::complex<double>)>& F,
    const std::vector<double>& t_eval,
    double sigma = 0.0,
    double b     = 0.0,
    int    N     = 64,
    const TransformConfig& cfg = {}
);

} // namespace laplacex
