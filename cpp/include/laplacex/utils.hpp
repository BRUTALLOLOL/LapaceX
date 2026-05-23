#pragma once

#include <vector>
#include <string>
#include "critical.hpp"

namespace laplacex {
namespace utils {

// ---------------------------------------------------------------------------
// Log-scale helpers
// ---------------------------------------------------------------------------

/**
 * Resample (t, f) onto a uniform grid of n points in x = ln(t).
 * Returns (x_grid, f_tilde) where f_tilde(x) = f(exp(x)) * exp(x).
 */
std::pair<std::vector<double>, std::vector<double>> to_log_plane(
    const std::vector<double>& t,
    const std::vector<double>& f,
    int n = 512
);

/**
 * Suggest a good t_min / t_max window for logarithmic analysis
 * based on the data's dynamic range.
 */
std::pair<double, double> suggest_log_range(
    const std::vector<double>& t,
    const std::vector<double>& f
);

// ---------------------------------------------------------------------------
// Model-selection criteria
// ---------------------------------------------------------------------------

/**
 * Compute AIC for a model with k parameters fit to n data points
 * with residual sum of squares rss.
 */
double aic(double rss, int n, int k);

/**
 * Compute BIC for a model with k parameters fit to n data points.
 */
double bic(double rss, int n, int k);

/**
 * Select the optimal number of components from a candidate list
 * using AIC or BIC (criterion = "aic" | "bic").
 */
std::vector<ExponentialComponent> select_components(
    const std::vector<std::vector<ExponentialComponent>>& candidates,
    const std::vector<double>& t,
    const std::vector<double>& f,
    const std::string& criterion = "aic"
);

// ---------------------------------------------------------------------------
// Test-signal generators
// ---------------------------------------------------------------------------

/** f(t) = H(t - t0)  (Heaviside step at t0) */
std::vector<double> gen_step(
    const std::vector<double>& t,
    double t0 = 1.0,
    double amplitude = 1.0
);

/** f(t) = A * exp(-rate * t) * cos(omega * t + phi) */
std::vector<double> gen_damped_sine(
    const std::vector<double>& t,
    double amplitude = 1.0,
    double rate      = 0.5,
    double omega     = 2.0,
    double phi       = 0.0
);

/**
 * f(t) = sum_k  A_k * exp(-rate_k * t)
 * @param amplitudes  vector of A_k
 * @param rates       vector of lambda_k
 */
std::vector<double> gen_exp_sum(
    const std::vector<double>& t,
    const std::vector<double>& amplitudes,
    const std::vector<double>& rates
);

/** f(t) = A / sqrt(t)  (diffusion-like tail) */
std::vector<double> gen_diffusion(
    const std::vector<double>& t,
    double amplitude = 1.0
);

// ---------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------

/**
 * Add Gaussian white noise with given standard deviation.
 * @param seed  RNG seed (-1 → random)
 */
std::vector<double> add_noise(
    const std::vector<double>& f,
    double sigma,
    int seed = -1
);

/**
 * Estimate noise standard deviation from finest wavelet scales.
 */
double estimate_noise(const std::vector<double>& f);

} // namespace utils
} // namespace laplacex
