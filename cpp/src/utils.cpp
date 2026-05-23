#include "laplacex/utils.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <random>

namespace laplacex {
namespace utils {

// ---------------------------------------------------------------------------
// Log-plane helpers
// ---------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double>> to_log_plane(
    const std::vector<double>& t,
    const std::vector<double>& f,
    int n)
{
    if (t.empty() || t.size() != f.size())
        throw std::invalid_argument("to_log_plane: t and f must be non-empty and same size");

    double x_min = std::log(t.front());
    double x_max = std::log(t.back());
    double dx    = (x_max - x_min) / (n - 1);

    std::vector<double> x_grid(n), f_tilde(n);
    for (int i = 0; i < n; ++i) {
        x_grid[i] = x_min + i * dx;
        double ti  = std::exp(x_grid[i]);
        // linear interp in original space
        auto it = std::lower_bound(t.begin(), t.end(), ti);
        std::size_t idx;
        double fi;
        if (it == t.end()) {
            fi = f.back();
        } else if (it == t.begin()) {
            fi = f.front();
        } else {
            idx = it - t.begin();
            double alpha = (ti - t[idx-1]) / (t[idx] - t[idx-1]);
            fi = f[idx-1] + alpha * (f[idx] - f[idx-1]);
        }
        f_tilde[i] = fi * ti;
    }
    return {x_grid, f_tilde};
}

std::pair<double, double> suggest_log_range(
    const std::vector<double>& t,
    const std::vector<double>& f)
{
    if (t.empty()) throw std::invalid_argument("suggest_log_range: empty input");

    // Find where signal is above 1% of its maximum
    double f_max = *std::max_element(f.begin(), f.end());
    double thr   = 0.01 * f_max;

    double t_lo = t.back(), t_hi = t.front();
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (std::abs(f[i]) > thr) {
            t_lo = std::min(t_lo, t[i]);
            t_hi = std::max(t_hi, t[i]);
        }
    }
    // Add one decade margin
    return {t_lo * 0.1, t_hi * 10.0};
}

// ---------------------------------------------------------------------------
// Information criteria
// ---------------------------------------------------------------------------
double aic(double rss, int n, int k) {
    if (rss <= 0 || n <= 0) return 1e30;
    return n * std::log(rss / n) + 2.0 * k;
}

double bic(double rss, int n, int k) {
    if (rss <= 0 || n <= 0) return 1e30;
    return n * std::log(rss / n) + k * std::log(n);
}

std::vector<ExponentialComponent> select_components(
    const std::vector<std::vector<ExponentialComponent>>& candidates,
    const std::vector<double>& t,
    const std::vector<double>& f,
    const std::string& criterion)
{
    if (candidates.empty()) return {};
    int n = static_cast<int>(t.size());

    double best_score = 1e30;
    std::size_t best_i = 0;

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& comps = candidates[i];
        auto pred = reconstruct(comps, t);
        double rss = 0;
        for (int j = 0; j < n; ++j) {
            double d = pred[j] - f[j]; rss += d*d;
        }
        int k = static_cast<int>(comps.size()) * 3; // amplitude, rate, decay
        double score = (criterion == "bic") ? bic(rss, n, k) : aic(rss, n, k);
        if (score < best_score) { best_score = score; best_i = i; }
    }
    return candidates[best_i];
}

// ---------------------------------------------------------------------------
// Test-signal generators
// ---------------------------------------------------------------------------
std::vector<double> gen_step(
    const std::vector<double>& t, double t0, double amplitude)
{
    std::vector<double> out(t.size());
    for (std::size_t i = 0; i < t.size(); ++i)
        out[i] = (t[i] >= t0) ? amplitude : 0.0;
    return out;
}

std::vector<double> gen_damped_sine(
    const std::vector<double>& t,
    double amplitude, double rate, double omega, double phi)
{
    std::vector<double> out(t.size());
    for (std::size_t i = 0; i < t.size(); ++i)
        out[i] = amplitude * std::exp(-rate * t[i]) * std::cos(omega * t[i] + phi);
    return out;
}

std::vector<double> gen_exp_sum(
    const std::vector<double>& t,
    const std::vector<double>& amplitudes,
    const std::vector<double>& rates)
{
    if (amplitudes.size() != rates.size())
        throw std::invalid_argument("gen_exp_sum: amplitudes and rates must have equal length");

    std::vector<double> out(t.size(), 0.0);
    for (std::size_t k = 0; k < amplitudes.size(); ++k)
        for (std::size_t i = 0; i < t.size(); ++i)
            out[i] += amplitudes[k] * std::exp(-rates[k] * t[i]);
    return out;
}

std::vector<double> gen_diffusion(
    const std::vector<double>& t, double amplitude)
{
    std::vector<double> out(t.size());
    for (std::size_t i = 0; i < t.size(); ++i)
        out[i] = (t[i] > 0) ? amplitude / std::sqrt(t[i]) : 0.0;
    return out;
}

// ---------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------
std::vector<double> add_noise(
    const std::vector<double>& f, double sigma, int seed)
{
    std::mt19937 rng((seed < 0) ? std::random_device{}() : static_cast<unsigned>(seed));
    std::normal_distribution<double> dist(0.0, sigma);

    std::vector<double> out(f.size());
    for (std::size_t i = 0; i < f.size(); ++i)
        out[i] = f[i] + dist(rng);
    return out;
}

double estimate_noise(const std::vector<double>& f) {
    if (f.size() < 2) return 0.0;
    // Estimate from first differences: sigma ≈ MAD(diff) / sqrt(2)
    std::vector<double> diffs(f.size()-1);
    for (std::size_t i = 0; i < diffs.size(); ++i)
        diffs[i] = std::abs(f[i+1] - f[i]);
    std::sort(diffs.begin(), diffs.end());
    double mad = diffs[diffs.size()/2];
    return mad / (std::sqrt(2.0) * 0.6745);
}

} // namespace utils
} // namespace laplacex
