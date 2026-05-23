#include "laplacex/core.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <future>
#include <numeric>

namespace laplacex {

// ---------------------------------------------------------------------------
// Gauss-Kronrod G7-K15 nodes and weights (standard interval [-1, 1])
// ---------------------------------------------------------------------------
static const double GK_NODES[15] = {
     0.0000000000000000,
     0.2077849550078985, -0.2077849550078985,
     0.4058451513773972, -0.4058451513773972,
     0.5860872354676911, -0.5860872354676911,
     0.7415311855993945, -0.7415311855993945,
     0.8648644233597691, -0.8648644233597691,
     0.9491079123427585, -0.9491079123427585,
     0.9914553711208126, -0.9914553711208126
};

static const double GK_W[15] = {
     0.2094821410847278,
     0.2044329400752989,  0.2044329400752989,
     0.1903505780647854,  0.1903505780647854,
     0.1690047266392679,  0.1690047266392679,
     0.1406532597155259,  0.1406532597155259,
     0.1047900103222502,  0.1047900103222502,
     0.0630920926299786,  0.0630920926299786,
     0.0229353220105292,  0.0229353220105292
};

static const double G7_W[15] = {
     0.4179591836734694,
     0.3818300505051189,  0.3818300505051189,
     0.2797053914892767,  0.2797053914892767,
     0.1294849661688697,  0.1294849661688697,
     0.0,                 0.0,
     0.0,                 0.0,
     0.0,                 0.0,
     0.0,                 0.0
};

// Evaluate integrand in [a, b] using G7-K15; return (kronrod, gauss) pair
static std::pair<std::complex<double>, std::complex<double>>
gk15(const std::function<std::complex<double>(double)>& g, double a, double b)
{
    double mid  = 0.5 * (a + b);
    double half = 0.5 * (b - a);

    std::complex<double> kron{}, gauss{};
    for (int i = 0; i < 15; ++i) {
        double x  = mid + half * GK_NODES[i];
        auto   fx = g(x);
        kron  += GK_W[i] * fx;
        gauss += G7_W[i] * fx;
    }
    kron  *= half;
    gauss *= half;
    return {kron, gauss};
}

// Adaptive recursive integration of g over [a, b]
static std::complex<double> adaptive_gk(
    const std::function<std::complex<double>(double)>& g,
    double a, double b,
    double tol, double rel_tol, int depth, int max_depth)
{
    auto [kron, gauss] = gk15(g, a, b);
    double err = std::abs(kron - gauss);
    double ref = std::abs(kron);

    if (depth >= max_depth || err < tol || (ref > 0 && err / ref < rel_tol)) {
        return kron;
    }

    double mid = 0.5 * (a + b);
    double t2  = tol * 0.5;
    return adaptive_gk(g, a, mid, t2, rel_tol, depth + 1, max_depth)
         + adaptive_gk(g, mid, b, t2, rel_tol, depth + 1, max_depth);
}

// ---------------------------------------------------------------------------
// Forward transform
// ---------------------------------------------------------------------------
std::complex<double> forward(
    const std::function<double(double)>& f,
    std::complex<double> s,
    const TransformConfig& cfg)
{
    if (std::real(s) < 0.0) {
        throw std::domain_error(
            "laplacex::forward requires Re(s) >= 0 for convergence");
    }

    // Transform: u = e^x, t = e^u so x goes from -inf to +inf
    // But we integrate over t directly on (0, T_max] with exponential
    // compression: split into geometric panels so each panel contributes
    // roughly equally to the transform.

    // Build panel breakpoints: 0, eps, 1/|s|, 10/|s|, t_max
    double sigma  = std::real(s);
    double s_abs  = std::abs(s);
    double center = (s_abs > 0) ? 1.0 / s_abs : 1.0;

    std::vector<double> breaks;
    breaks.push_back(1e-15);
    // geometric runup to center
    for (double x = 1e-6; x < center; x *= 4.0)  breaks.push_back(x);
    breaks.push_back(center);
    // geometric rundown from center
    for (double x = center * 4.0; x < cfg.t_max; x *= 4.0) breaks.push_back(x);
    breaks.push_back(cfg.t_max);
    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());

    std::complex<double> result{};
    for (std::size_t i = 0; i + 1 < breaks.size(); ++i) {
        double a = breaks[i], b = breaks[i + 1];
        auto g = [&](double t) -> std::complex<double> {
            return f(t) * std::exp(-s * t);
        };
        result += adaptive_gk(g, a, b, cfg.tol, cfg.rel_tol, 0, cfg.max_levels);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Batch forward (parallelised over s values)
// ---------------------------------------------------------------------------
std::vector<std::complex<double>> forward_batch(
    const std::function<double(double)>& f,
    const std::vector<std::complex<double>>& s_vals,
    const TransformConfig& cfg)
{
    std::size_t n = s_vals.size();
    std::vector<std::complex<double>> out(n);

    unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::future<void>> futs;

    auto worker = [&](std::size_t from, std::size_t to) {
        for (std::size_t i = from; i < to; ++i)
            out[i] = forward(f, s_vals[i], cfg);
    };

    std::size_t chunk = (n + nthreads - 1) / nthreads;
    for (unsigned t = 0; t < nthreads; ++t) {
        std::size_t from = t * chunk;
        std::size_t to   = std::min(from + chunk, n);
        if (from < to)
            futs.push_back(std::async(std::launch::async, worker, from, to));
    }
    for (auto& fut : futs) fut.get();
    return out;
}

// ---------------------------------------------------------------------------
// Cubic-spline interpolation (natural BC)
// ---------------------------------------------------------------------------
struct CubicSpline {
    std::vector<double> t, a, b, c, d;

    CubicSpline(const std::vector<double>& x, const std::vector<double>& y) {
        int n = static_cast<int>(x.size()) - 1;
        if (n < 1) throw std::invalid_argument("Need at least 2 points");

        t = x; a = y;
        std::vector<double> h(n), alpha(n), l(n+1), mu(n+1), z(n+1);
        b.resize(n); c.resize(n+1); d.resize(n);

        for (int i = 0; i < n; ++i) h[i] = x[i+1] - x[i];
        for (int i = 1; i < n; ++i)
            alpha[i] = 3*(y[i+1]-y[i])/h[i] - 3*(y[i]-y[i-1])/h[i-1];

        l[0] = 1; mu[0] = z[0] = 0;
        for (int i = 1; i < n; ++i) {
            l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i];
        }
        l[n] = 1; z[n] = c[n] = 0;
        for (int j = n-1; j >= 0; --j) {
            c[j] = z[j] - mu[j]*c[j+1];
            b[j] = (a[j+1]-a[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3;
            d[j] = (c[j+1]-c[j]) / (3*h[j]);
        }
    }

    double eval(double x) const {
        int n = static_cast<int>(t.size()) - 1;
        // clamp
        if (x <= t.front()) return a.front();
        if (x >= t.back())  return a.back();
        // binary search
        int lo = 0, hi = n - 1;
        while (lo < hi) { int mid = (lo+hi)/2; if (t[mid+1] < x) lo=mid+1; else hi=mid; }
        double dx = x - t[lo];
        return a[lo] + b[lo]*dx + c[lo]*dx*dx + d[lo]*dx*dx*dx;
    }
};

// ---------------------------------------------------------------------------
// Forward from discrete samples
// ---------------------------------------------------------------------------
std::vector<std::complex<double>> forward_discrete(
    const std::vector<double>& t,
    const std::vector<double>& f,
    const std::vector<std::complex<double>>& s_vals,
    const TransformConfig& cfg)
{
    CubicSpline spline(t, f);
    auto interp = [&](double x) { return spline.eval(x); };
    return forward_batch(interp, s_vals, cfg);
}

// ---------------------------------------------------------------------------
// Inverse – Weeks method
// ---------------------------------------------------------------------------
std::vector<double> inverse_weeks(
    const std::function<std::complex<double>(std::complex<double>)>& F,
    const std::vector<double>& t_eval,
    double sigma, double b, int N,
    const TransformConfig& cfg)
{
    // Auto-select parameters if not provided
    if (sigma <= 0.0) sigma = 1.0;
    if (b     <= 0.0) b     = 1.0;

    // Compute Laguerre coefficients via FFT-free direct summation
    // c_n = (b / pi) * Re{ integral_0^{2pi} F(s_k) * L_n(.) d theta }
    // We use the trapezoidal rule on the circle s = sigma + b*(1+e^{i theta})/(1-e^{i theta})

    int M = 4 * N; // quadrature points on circle
    std::vector<double> c(N, 0.0);

    for (int m = 0; m < M; ++m) {
        double theta = 2.0 * M_PI * m / M;
        std::complex<double> z  = std::exp(std::complex<double>(0, theta));
        std::complex<double> s  = sigma + b * (1.0 + z) / (1.0 - z);
        std::complex<double> Fs = F(s) * b / (1.0 - z) / (1.0 - z);

        // Laguerre polynomials L_0..L_{N-1} evaluated at (2b/s-b)*... 
        // simplified: use recurrence
        double arg = 2.0 * b / std::real(1.0 / (s - sigma) + 1.0); // safe approx
        std::vector<double> L(N);
        L[0] = 1.0;
        if (N > 1) L[1] = 1.0 - arg;
        for (int n = 2; n < N; ++n)
            L[n] = ((2*n - 1 - arg) * L[n-1] - (n-1) * L[n-2]) / n;

        for (int n = 0; n < N; ++n)
            c[n] += std::real(Fs) * L[n] / M;
    }

    // Reconstruct f(t) = exp(sigma*t) * sum_n c_n * L_n(b*t) * exp(-b*t)
    std::vector<double> out(t_eval.size());
    for (std::size_t i = 0; i < t_eval.size(); ++i) {
        double tt  = t_eval[i];
        double arg = b * tt;
        std::vector<double> L(N);
        L[0] = 1.0;
        if (N > 1) L[1] = 1.0 - arg;
        for (int n = 2; n < N; ++n)
            L[n] = ((2*n - 1 - arg) * L[n-1] - (n-1) * L[n-2]) / n;

        double val = 0.0;
        for (int n = 0; n < N; ++n)
            val += c[n] * L[n];

        out[i] = val * std::exp((sigma - b) * tt);
    }
    return out;
}

} // namespace laplacex
