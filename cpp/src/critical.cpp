#include "laplacex/critical.hpp"
#include "laplacex/utils.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <map>
#include <vector>
#include <iostream>   // optional – remove after debugging

namespace laplacex {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Mexican Hat wavelet psi(x) = (1 - x^2) * exp(-x^2/2)
static double mexh(double x) {
    return (1.0 - x * x) * std::exp(-0.5 * x * x);
}

// Real Continuous Wavelet Transform at scale a, translation b
static double cwt_point(
    const std::vector<double>& x_grid,
    const std::vector<double>& f_tilde,
    double a, double b,
    const std::string& wavelet)
{
    int n = static_cast<int>(x_grid.size());
    double dx = x_grid[1] - x_grid[0]; // uniform grid
    double inv_sqrt_a = 1.0 / std::sqrt(a);
    double sum = 0.0;

    for (int i = 0; i < n; ++i) {
        double xi = (x_grid[i] - b) / a;
        double psi_val = (wavelet == "mexh") ? mexh(xi) : mexh(xi);
        sum += f_tilde[i] * psi_val;
    }
    return inv_sqrt_a * sum * dx;
}

// ---------------------------------------------------------------------------
// Resample to uniform log grid (simple linear interp)
// ---------------------------------------------------------------------------
static void to_log_grid(
    const std::vector<double>& t,
    const std::vector<double>& f,
    int n,
    std::vector<double>& x_grid,
    std::vector<double>& f_tilde)
{
    double x_min = std::log(t.front());
    double x_max = std::log(t.back());
    double dx    = (x_max - x_min) / (n - 1);

    x_grid.resize(n);
    f_tilde.resize(n);

    for (int i = 0; i < n; ++i) {
        x_grid[i] = x_min + i * dx;
        double ti = std::exp(x_grid[i]);

        auto it = std::lower_bound(t.begin(), t.end(), ti);
        if (it == t.end()) {
            f_tilde[i] = f.back() * ti;
            continue;
        }
        if (it == t.begin()) {
            f_tilde[i] = f.front() * ti;
            continue;
        }
        std::size_t idx = it - t.begin();
        double t0 = t[idx-1], t1 = t[idx];
        double f0 = f[idx-1], f1 = f[idx];
        double alpha = (ti - t0) / (t1 - t0);
        double fi = f0 + alpha * (f1 - f0);
        f_tilde[i] = fi * ti;
    }
}

// ---------------------------------------------------------------------------
// Linear least squares solver for small systems (normal equations + Gauss elim)
// ---------------------------------------------------------------------------
static std::vector<double> solve_normal_eq(
    const std::vector<std::vector<double>>& A,   // A^T A
    const std::vector<double>& b)                // A^T f
{
    int n = static_cast<int>(A.size());
    if (n == 0) return {};
    // Augmented matrix
    std::vector<std::vector<double>> aug(n, std::vector<double>(n+1, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug[i][j] = A[i][j];
        aug[i][n] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double max_val = std::abs(aug[col][col]);
        for (int row = col+1; row < n; ++row) {
            if (std::abs(aug[row][col]) > max_val) {
                max_val = std::abs(aug[row][col]);
                pivot = row;
            }
        }
        if (max_val < 1e-15) continue; // singular – leave as zero
        if (pivot != col) std::swap(aug[col], aug[pivot]);

        double div = aug[col][col];
        for (int j = col; j <= n; ++j) aug[col][j] /= div;
        for (int i = 0; i < n; ++i) {
            if (i == col) continue;
            double factor = aug[i][col];
            for (int j = col; j <= n; ++j) aug[i][j] -= factor * aug[col][j];
        }
    }

    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = aug[i][n];
    return x;
}

// ---------------------------------------------------------------------------
// Refine component rates and amplitudes via alternating least squares
// ---------------------------------------------------------------------------
static void refine_components_als(
    std::vector<ExponentialComponent>& components,
    const std::vector<double>& t,
    const std::vector<double>& f)
{
    if (components.empty()) return;

    const int n_times = static_cast<int>(t.size());
    const int n_comp  = static_cast<int>(components.size());

    const int outer_iter = 3;
    const int ternary_iter = 40;

    for (int iter = 0; iter < outer_iter; ++iter) {
        for (int k = 0; k < n_comp; ++k) {
            // Skip oscillatory components in rate search — they need different
            // (damped-cosine) basis functions. Their amplitudes are still
            // updated in the global amplitude solve at the end of each outer pass.
            if (components[k].frequency != 0.0) continue;

            double rate_init = components[k].rate;
            if (rate_init <= 0.0) rate_init = 0.1;

            double lo = rate_init * 0.25;
            double hi = rate_init * 4.0;
            if (lo < 1e-12) lo = 1e-12;   // avoid zero rate

            // Ternary search for best rate of component k
            auto rss_for_rate = [&](double r) -> double {
                // Build design matrix for current set of rates (with candidate r for k)
                std::vector<double> rates = std::vector<double>(n_comp);
                for (int i = 0; i < n_comp; ++i) rates[i] = components[i].rate;
                rates[k] = r;

                // Form A (n_times x n_comp)
                std::vector<std::vector<double>> A(n_times, std::vector<double>(n_comp, 0.0));
                for (int i = 0; i < n_times; ++i) {
                    for (int j = 0; j < n_comp; ++j) {
                        A[i][j] = std::exp(-rates[j] * t[i]);
                    }
                }

                // Compute A^T A and A^T f
                std::vector<std::vector<double>> ATA(n_comp, std::vector<double>(n_comp, 0.0));
                std::vector<double> ATf(n_comp, 0.0);
                for (int i = 0; i < n_times; ++i) {
                    for (int p = 0; p < n_comp; ++p) {
                        double aip = A[i][p];
                        for (int q = 0; q < n_comp; ++q) {
                            ATA[p][q] += aip * A[i][q];
                        }
                        ATf[p] += aip * f[i];
                    }
                }

                // Solve for amplitudes
                std::vector<double> amps = solve_normal_eq(ATA, ATf);

                // Compute RSS
                double rss = 0.0;
                for (int i = 0; i < n_times; ++i) {
                    double pred = 0.0;
                    for (int j = 0; j < n_comp; ++j)
                        pred += amps[j] * A[i][j];
                    double diff = pred - f[i];
                    rss += diff * diff;
                }
                return rss;
            };

            // Ternary search
            for (int step = 0; step < ternary_iter; ++step) {
                double m1 = lo + (hi - lo) / 3.0;
                double m2 = hi - (hi - lo) / 3.0;
                double f1 = rss_for_rate(m1);
                double f2 = rss_for_rate(m2);
                if (f1 < f2) {
                    hi = m2;
                } else {
                    lo = m1;
                }
            }
            double best_rate = (lo + hi) * 0.5;
            components[k].rate = best_rate;
        }

        // After refining all components, recompute amplitudes for all simultaneously
        // (using updated rates)
        std::vector<double> rates(n_comp);
        for (int j = 0; j < n_comp; ++j) rates[j] = components[j].rate;

        std::vector<std::vector<double>> A(n_times, std::vector<double>(n_comp));
        for (int i = 0; i < n_times; ++i)
            for (int j = 0; j < n_comp; ++j)
                A[i][j] = std::exp(-rates[j] * t[i]);

        std::vector<std::vector<double>> ATA(n_comp, std::vector<double>(n_comp, 0.0));
        std::vector<double> ATf(n_comp, 0.0);
        for (int i = 0; i < n_times; ++i) {
            for (int p = 0; p < n_comp; ++p) {
                double aip = A[i][p];
                for (int q = 0; q < n_comp; ++q) ATA[p][q] += aip * A[i][q];
                ATf[p] += aip * f[i];
            }
        }
        std::vector<double> new_amps = solve_normal_eq(ATA, ATf);
        for (int j = 0; j < n_comp; ++j) {
            components[j].amplitude = new_amps[j];
            if (components[j].rate > 0.0) {
                components[j].decay_time = 1.0 / components[j].rate;
                components[j].log_time   = std::log(components[j].decay_time);
            } else {
                components[j].decay_time = std::numeric_limits<double>::infinity();
                components[j].log_time   = -std::numeric_limits<double>::infinity();
            }
        }
    }

    // Final recompute amplitudes one last time (already done in the last inner loop)
    // Re‑normalise significance by absolute amplitudes
    double sum_abs = 0.0;
    for (const auto& c : components) sum_abs += std::abs(c.amplitude);
    if (sum_abs > 0.0) {
        for (auto& c : components)
            c.significance = std::abs(c.amplitude) / sum_abs;
    }
}

// ---------------------------------------------------------------------------
// decompose
// ---------------------------------------------------------------------------
std::vector<ExponentialComponent> decompose(
    const std::vector<double>& t,
    const std::vector<double>& f,
    const DecomposeConfig& cfg)
{
    if (t.size() != f.size() || t.size() < 4)
        throw std::invalid_argument("laplacex::decompose: need at least 4 (t,f) pairs");
    for (double ti : t)
        if (ti <= 0.0)
            throw std::domain_error("laplacex::decompose: all t values must be > 0");

    // 1. Log-plane resampling
    std::vector<double> x_grid, f_tilde;
    to_log_grid(t, f, cfg.n_log_points, x_grid, f_tilde);

    int n = cfg.n_log_points;
    double x_min = x_grid.front();
    double x_max = x_grid.back();

    // 2. Build scale array (log-spaced)
    int ns = cfg.n_scales;
    std::vector<double> scales(ns);
    double log_amin = std::log(cfg.scale_min);
    double log_amax = std::log(cfg.scale_max);
    for (int j = 0; j < ns; ++j)
        scales[j] = std::exp(log_amin + j * (log_amax - log_amin) / (ns - 1));

    // 3. Compute CWT matrix W[j][i] = W(scale_j, x_i)
    std::vector<std::vector<double>> W(ns, std::vector<double>(n, 0.0));
    double W_max = 0.0;

    for (int j = 0; j < ns; ++j) {
        for (int i = 0; i < n; ++i) {
            W[j][i] = cwt_point(x_grid, f_tilde, scales[j], x_grid[i], cfg.wavelet);
            W_max = std::max(W_max, std::abs(W[j][i]));
        }
    }

    if (W_max == 0.0) return {}; // flat signal

    // 4. Noise estimation from finest scales
    double noise_sigma = cfg.noise_sigma;
    if (noise_sigma < 0.0) {
        std::vector<double> fine(W[0].begin(), W[0].end());
        for (auto& v : fine) v = std::abs(v);
        std::sort(fine.begin(), fine.end());
        noise_sigma = fine[fine.size()/2] / 0.6745;
    }

    // 5. Find local maxima in x for each scale → ridge points
    std::vector<std::vector<std::pair<int,double>>> ridges(ns);
    for (int j = 0; j < ns; ++j) {
        for (int i = 1; i + 1 < n; ++i) {
            double va = std::abs(W[j][i-1]);
            double vb = std::abs(W[j][i]);
            double vc = std::abs(W[j][i+1]);
            if (vb > va && vb >= vc && vb > cfg.ridge_threshold * W_max) {
                ridges[j].push_back({i, vb});
            }
        }
    }

    // 6. Connect ridge points across scales → ridge lines
    struct RidgeLine {
        std::vector<std::pair<int,int>> points; // (scale_idx, x_idx)
        double peak_energy = 0.0;
        int    peak_scale  = 0;
        int    peak_x      = 0;
    };

    std::vector<RidgeLine> lines;
    // Start from coarsest scale
    for (auto& [xi, energy] : ridges[ns-1]) {
        RidgeLine line;
        line.points.push_back({ns-1, xi});
        line.peak_energy = energy;
        line.peak_scale  = ns-1;
        line.peak_x      = xi;

        int cur_x = xi;
        for (int j = ns-2; j >= 0; --j) {
            int best_i = -1;
            double best_d = 1e9;
            for (auto& [ri, re] : ridges[j]) {
                double d = std::abs(ri - cur_x);
                if (d < best_d && d <= 3) { best_d = d; best_i = ri; }
            }
            if (best_i < 0) break;
            line.points.push_back({j, best_i});
            double E = std::abs(W[j][best_i]);
            if (E > line.peak_energy) {
                line.peak_energy = E;
                line.peak_scale  = j;
                line.peak_x      = best_i;
            }
            cur_x = best_i;
        }

        if (static_cast<int>(line.points.size()) >= cfg.min_ridge_len)
            lines.push_back(std::move(line));
    }

    // 7. Extract component parameters from each ridge line
    std::vector<ExponentialComponent> components;
    double total_energy = 0.0;

    for (auto& line : lines) {
        int j = line.peak_scale;
        int i = line.peak_x;

        double x_k = x_grid[i];
        double t_k = std::exp(x_k);
        double A_k = line.peak_energy * scales[j]; // un-normalised

        // SNR check
        if (A_k < cfg.snr_threshold * noise_sigma) continue;

        ExponentialComponent comp;
        comp.log_time   = x_k;
        comp.decay_time = t_k;
        comp.rate       = (t_k > 0) ? 1.0 / t_k : 0.0;
        comp.amplitude  = A_k;
        comp.significance = A_k; // will normalise below

        // Oscillation detection – count sign changes of W within a LOCAL window.
        // Capped at 20 grid points each side to avoid false positives: at coarse
        // scales the uncapped window spans the whole grid and the Mexican Hat
        // CWT always shows sign changes that are NOT true signal oscillations.
        int win = std::min(static_cast<int>(scales[j] / (x_grid[1]-x_grid[0]) * 2), 20);
        int cnt = 0;
        for (int k = std::max(0,i-win); k+1 < std::min(n, i+win); ++k) {
            if (W[j][k] * W[j][k+1] < 0) ++cnt;
        }
        if (cnt >= 2) {
            comp.frequency = cnt * M_PI / (2.0 * scales[j]);
        }

        total_energy += A_k;
        components.push_back(comp);
    }

    // 8. Normalise significance
    if (total_energy > 0) {
        for (auto& c : components)
            c.significance = c.significance / total_energy;
    }

    // 9. Sort by significance descending
    std::sort(components.begin(), components.end(),
        [](const ExponentialComponent& a, const ExponentialComponent& b) {
            return a.significance > b.significance;
        });

    // ---- NEW: Refine rates and amplitudes (ALS) ----
    refine_components_als(components, t, f);
    // Re‑sort after refinement (significance was updated inside refine)
    std::sort(components.begin(), components.end(),
        [](const ExponentialComponent& a, const ExponentialComponent& b) {
            return a.significance > b.significance;
        });

    // 10. Optional AIC pruning (using refined components)
    if (cfg.use_aic && components.size() > 1) {
        auto rss = [&](std::size_t k) {
            auto sub = std::vector<ExponentialComponent>(components.begin(), components.begin()+k);
            auto r   = reconstruct(sub, t);
            double s = 0;
            for (std::size_t idx = 0; idx < t.size(); ++idx) {
                double d = r[idx] - f[idx];
                s += d * d;
            }
            return s;
        };

        int best_k = 1;
        double best_aic = std::log(rss(1) / t.size()) + 2.0 * 3;
        for (std::size_t k = 2; k <= components.size(); ++k) {
            double a = std::log(rss(k) / t.size()) + 2.0 * 3 * k;
            if (a < best_aic) { best_aic = a; best_k = static_cast<int>(k); }
        }
        components.resize(best_k);
    }

    return components;
}

// ---------------------------------------------------------------------------
// reconstruct
// ---------------------------------------------------------------------------
std::vector<double> reconstruct(
    const std::vector<ExponentialComponent>& components,
    const std::vector<double>& t_eval)
{
    std::vector<double> out(t_eval.size(), 0.0);
    for (std::size_t i = 0; i < t_eval.size(); ++i) {
        double tt = t_eval[i];
        for (const auto& c : components) {
            double val = c.amplitude * std::exp(-c.rate * tt);
            if (c.frequency != 0.0)
                val *= std::cos(c.frequency * tt + c.phase);
            out[i] += val;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// reconstruct_laplace
// ---------------------------------------------------------------------------
std::vector<std::complex<double>> reconstruct_laplace(
    const std::vector<ExponentialComponent>& components,
    const std::vector<std::complex<double>>& s_eval)
{
    std::vector<std::complex<double>> out(s_eval.size(), 0.0);
    for (std::size_t i = 0; i < s_eval.size(); ++i) {
        std::complex<double> s = s_eval[i];
        for (const auto& c : components) {
            std::complex<double> denom = s + std::complex<double>(c.rate, -c.frequency);
            if (std::abs(denom) > 1e-30)
                out[i] += c.amplitude / denom;
        }
    }
    return out;
}

} // namespace laplacex
