#pragma once

#include <vector>
#include <complex>
#include <string>

namespace laplacex {

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/**
 * A single exponential component recovered from a signal.
 *
 *   contribution(t) = amplitude * exp(-rate * t)
 *
 * For oscillatory modes, frequency != 0 and the contribution is
 *   amplitude * exp(-rate * t) * cos(frequency * t + phase)
 */
struct ExponentialComponent {
    double amplitude   = 0.0;  ///< A_k
    double rate        = 0.0;  ///< lambda_k  (decay rate, >= 0)
    double decay_time  = 0.0;  ///< 1 / lambda_k  (relaxation time)
    double frequency   = 0.0;  ///< omega_k (0 for purely real modes)
    double phase       = 0.0;  ///< phi_k
    double significance = 0.0; ///< relative energy of this component (0–1)
    double log_time    = 0.0;  ///< x_k = ln(t_k) – position in log plane
};

// ---------------------------------------------------------------------------
// Critical-points decomposition
// ---------------------------------------------------------------------------

struct DecomposeConfig {
    // Wavelet
    std::string wavelet    = "mexh";  ///< "mexh" | "cgauss" (complex Gaussian)
    int    n_scales        = 64;      ///< number of dyadic scales
    double scale_min       = 0.5;     ///< smallest wavelet scale (log-space units)
    double scale_max       = 4.0;     ///< largest  wavelet scale (log-space units)

    // Log-space resampling
    int    n_log_points    = 512;     ///< uniform grid size in x = ln(t)

    // Skeleton / ridge tracking
    double ridge_threshold = 0.05;   ///< min |W| / max|W| to keep a ridge line
    int    min_ridge_len   = 4;      ///< minimum ridge length (in scale steps)

    // Noise & filtering
    double noise_sigma     = -1.0;   ///< <0 → auto-estimated from finest scales
    double snr_threshold   = 3.0;    ///< min amplitude / noise to keep component

    // Model selection
    bool   use_aic         = true;   ///< use AIC to prune over-fitted models
};

/**
 * Decompose a time-series into a sum of exponential (+ oscillatory) components
 * by operating in the logarithmic time plane with a continuous wavelet transform.
 *
 * @param t       Strictly increasing time values (t > 0)
 * @param f       Signal values at t
 * @param cfg     Algorithm configuration
 * @return        Sorted (by significance, descending) list of components
 */
std::vector<ExponentialComponent> decompose(
    const std::vector<double>& t,
    const std::vector<double>& f,
    const DecomposeConfig& cfg = {}
);

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/**
 * Reconstruct f(t) from a list of exponential components.
 */
std::vector<double> reconstruct(
    const std::vector<ExponentialComponent>& components,
    const std::vector<double>& t_eval
);

/**
 * Compute the Laplace image F(s) analytically from components:
 *   F(s) = sum_k  A_k / (s + lambda_k)          (continuous exponentials)
 */
std::vector<std::complex<double>> reconstruct_laplace(
    const std::vector<ExponentialComponent>& components,
    const std::vector<std::complex<double>>& s_eval
);

} // namespace laplacex
