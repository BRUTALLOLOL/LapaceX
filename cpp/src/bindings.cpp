#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "laplacex/core.hpp"
#include "laplacex/critical.hpp"
#include "laplacex/utils.hpp"

namespace py = pybind11;
using namespace laplacex;

// ---------------------------------------------------------------------------
// Helper: convert numpy array to std::vector<double>
// ---------------------------------------------------------------------------
static std::vector<double> to_vec(py::array_t<double> arr) {
    auto buf = arr.request();
    auto* ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

static std::vector<std::complex<double>> to_cvec(py::array_t<std::complex<double>> arr) {
    auto buf = arr.request();
    auto* ptr = static_cast<std::complex<double>*>(buf.ptr);
    return std::vector<std::complex<double>>(ptr, ptr + buf.size);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_laplacex, m) {
    m.doc() = "LaplaceX C++ core — stable Laplace transforms via logarithmic-plane analysis";

    // -----------------------------------------------------------------------
    // TransformConfig
    // -----------------------------------------------------------------------
    py::class_<TransformConfig>(m, "TransformConfig",
        "Configuration for adaptive Gauss-Kronrod quadrature.")
        .def(py::init<>())
        .def_readwrite("tol",        &TransformConfig::tol)
        .def_readwrite("rel_tol",    &TransformConfig::rel_tol)
        .def_readwrite("max_levels", &TransformConfig::max_levels)
        .def_readwrite("t_max",      &TransformConfig::t_max)
        .def("__repr__", [](const TransformConfig& c) {
            return "<TransformConfig tol=" + std::to_string(c.tol)
                 + " rel_tol=" + std::to_string(c.rel_tol)
                 + " max_levels=" + std::to_string(c.max_levels) + ">";
        });

    // -----------------------------------------------------------------------
    // DecomposeConfig
    // -----------------------------------------------------------------------
    py::class_<DecomposeConfig>(m, "DecomposeConfig",
        "Configuration for the exponential-component decomposition.")
        .def(py::init<>())
        .def_readwrite("wavelet",         &DecomposeConfig::wavelet)
        .def_readwrite("n_scales",        &DecomposeConfig::n_scales)
        .def_readwrite("scale_min",       &DecomposeConfig::scale_min)
        .def_readwrite("scale_max",       &DecomposeConfig::scale_max)
        .def_readwrite("n_log_points",    &DecomposeConfig::n_log_points)
        .def_readwrite("ridge_threshold", &DecomposeConfig::ridge_threshold)
        .def_readwrite("min_ridge_len",   &DecomposeConfig::min_ridge_len)
        .def_readwrite("noise_sigma",     &DecomposeConfig::noise_sigma)
        .def_readwrite("snr_threshold",   &DecomposeConfig::snr_threshold)
        .def_readwrite("use_aic",         &DecomposeConfig::use_aic)
        .def("__repr__", [](const DecomposeConfig& c) {
            return "<DecomposeConfig wavelet=" + c.wavelet
                 + " n_scales=" + std::to_string(c.n_scales)
                 + " n_log_points=" + std::to_string(c.n_log_points) + ">";
        });

    // -----------------------------------------------------------------------
    // ExponentialComponent
    // -----------------------------------------------------------------------
    py::class_<ExponentialComponent>(m, "ExponentialComponent",
        "A single exponential (or oscillatory-exponential) mode detected in a signal.")
        .def(py::init<>())
        .def_readwrite("amplitude",    &ExponentialComponent::amplitude)
        .def_readwrite("rate",         &ExponentialComponent::rate)
        .def_readwrite("decay_time",   &ExponentialComponent::decay_time)
        .def_readwrite("frequency",    &ExponentialComponent::frequency)
        .def_readwrite("phase",        &ExponentialComponent::phase)
        .def_readwrite("significance", &ExponentialComponent::significance)
        .def_readwrite("log_time",     &ExponentialComponent::log_time)
        .def("__repr__", [](const ExponentialComponent& c) {
            return "<ExponentialComponent amplitude=" + std::to_string(c.amplitude)
                 + " rate=" + std::to_string(c.rate)
                 + " decay_time=" + std::to_string(c.decay_time)
                 + " significance=" + std::to_string(c.significance) + ">";
        });

    // -----------------------------------------------------------------------
    // Core: forward
    // -----------------------------------------------------------------------
    m.def("forward",
        [](py::object f_py, std::complex<double> s, const TransformConfig& cfg) {
            auto f = [&](double t) { return f_py(t).cast<double>(); };
            return forward(f, s, cfg);
        },
        py::arg("f"), py::arg("s"), py::arg("cfg") = TransformConfig{},
        "Compute F(s) = ∫₀^∞ f(t) exp(-st) dt via adaptive Gauss-Kronrod quadrature.");

    m.def("forward_batch",
        [](py::object f_py,
           py::array_t<std::complex<double>> s_arr,
           const TransformConfig& cfg) {
            auto f = [&](double t) { return f_py(t).cast<double>(); };
            auto s = to_cvec(s_arr);
            return forward_batch(f, s, cfg);
        },
        py::arg("f"), py::arg("s_vals"), py::arg("cfg") = TransformConfig{},
        "Compute F(s) for an array of complex frequencies (parallelised).");

    m.def("forward_discrete",
        [](py::array_t<double> t_arr,
           py::array_t<double> f_arr,
           py::array_t<std::complex<double>> s_arr,
           const TransformConfig& cfg) {
            return forward_discrete(to_vec(t_arr), to_vec(f_arr), to_cvec(s_arr), cfg);
        },
        py::arg("t"), py::arg("f"), py::arg("s_vals"), py::arg("cfg") = TransformConfig{},
        "Forward Laplace transform from discrete samples (cubic-spline interpolated).");

    // -----------------------------------------------------------------------
    // Core: inverse (Weeks)
    // -----------------------------------------------------------------------
    m.def("inverse_weeks",
        [](py::object F_py,
           py::array_t<double> t_arr,
           double sigma, double b, int N,
           const TransformConfig& cfg) {
            auto F = [&](std::complex<double> s) {
                return F_py(s).cast<std::complex<double>>();
            };
            return inverse_weeks(F, to_vec(t_arr), sigma, b, N, cfg);
        },
        py::arg("F"), py::arg("t_eval"),
        py::arg("sigma") = 0.0, py::arg("b") = 0.0, py::arg("N") = 64,
        py::arg("cfg")   = TransformConfig{},
        "Approximate inverse Laplace transform via Weeks–Laguerre method.");

    // -----------------------------------------------------------------------
    // Critical: decompose
    // -----------------------------------------------------------------------
    m.def("decompose",
        [](py::array_t<double> t_arr,
           py::array_t<double> f_arr,
           const DecomposeConfig& cfg) {
            return decompose(to_vec(t_arr), to_vec(f_arr), cfg);
        },
        py::arg("t"), py::arg("f"), py::arg("cfg") = DecomposeConfig{},
        "Decompose a time-series into exponential components using wavelet ridges in log-plane.");

    // -----------------------------------------------------------------------
    // Reconstruction
    // -----------------------------------------------------------------------
    m.def("reconstruct",
        [](const std::vector<ExponentialComponent>& comps,
           py::array_t<double> t_arr) {
            return reconstruct(comps, to_vec(t_arr));
        },
        py::arg("components"), py::arg("t_eval"),
        "Reconstruct f(t) = Σ A_k exp(-λ_k t) from ExponentialComponent list.");

    m.def("reconstruct_laplace",
        [](const std::vector<ExponentialComponent>& comps,
           py::array_t<std::complex<double>> s_arr) {
            return reconstruct_laplace(comps, to_cvec(s_arr));
        },
        py::arg("components"), py::arg("s_eval"),
        "Compute F(s) = Σ A_k / (s + λ_k) analytically from ExponentialComponent list.");

    // -----------------------------------------------------------------------
    // Utils
    // -----------------------------------------------------------------------
    auto u = m.def_submodule("utils", "Signal-processing utilities.");

    u.def("to_log_plane",
        [](py::array_t<double> t, py::array_t<double> f, int n) {
            return utils::to_log_plane(to_vec(t), to_vec(f), n);
        },
        py::arg("t"), py::arg("f"), py::arg("n") = 512,
        "Resample (t, f) onto a uniform log-space grid; returns (x_grid, f_tilde).");

    u.def("suggest_log_range",
        [](py::array_t<double> t, py::array_t<double> f) {
            return utils::suggest_log_range(to_vec(t), to_vec(f));
        },
        py::arg("t"), py::arg("f"),
        "Suggest (t_min, t_max) for logarithmic analysis based on signal dynamic range.");

    u.def("aic", &utils::aic,
        py::arg("rss"), py::arg("n"), py::arg("k"),
        "Akaike Information Criterion: n*log(rss/n) + 2k.");

    u.def("bic", &utils::bic,
        py::arg("rss"), py::arg("n"), py::arg("k"),
        "Bayesian Information Criterion: n*log(rss/n) + k*log(n).");

    u.def("gen_step",
        [](py::array_t<double> t, double t0, double amplitude) {
            return utils::gen_step(to_vec(t), t0, amplitude);
        },
        py::arg("t"), py::arg("t0") = 1.0, py::arg("amplitude") = 1.0,
        "Generate Heaviside step function.");

    u.def("gen_damped_sine",
        [](py::array_t<double> t, double amp, double rate, double omega, double phi) {
            return utils::gen_damped_sine(to_vec(t), amp, rate, omega, phi);
        },
        py::arg("t"), py::arg("amplitude") = 1.0, py::arg("rate") = 0.5,
        py::arg("omega") = 2.0, py::arg("phi") = 0.0,
        "Generate damped sinusoid A*exp(-rate*t)*cos(omega*t + phi).");

    u.def("gen_exp_sum",
        [](py::array_t<double> t,
           py::array_t<double> amps,
           py::array_t<double> rates) {
            return utils::gen_exp_sum(to_vec(t), to_vec(amps), to_vec(rates));
        },
        py::arg("t"), py::arg("amplitudes"), py::arg("rates"),
        "Generate sum of real exponentials Σ A_k * exp(-lambda_k * t).");

    u.def("gen_diffusion",
        [](py::array_t<double> t, double amp) {
            return utils::gen_diffusion(to_vec(t), amp);
        },
        py::arg("t"), py::arg("amplitude") = 1.0,
        "Generate diffusion-like tail A / sqrt(t).");

    u.def("add_noise",
        [](py::array_t<double> f, double sigma, int seed) {
            return utils::add_noise(to_vec(f), sigma, seed);
        },
        py::arg("f"), py::arg("sigma"), py::arg("seed") = -1,
        "Add Gaussian white noise with given standard deviation.");

    u.def("estimate_noise",
        [](py::array_t<double> f) { return utils::estimate_noise(to_vec(f)); },
        py::arg("f"),
        "Estimate noise standard deviation from first differences (MAD estimator).");

    // Version
    m.attr("__version__") = "0.1.0";
}
