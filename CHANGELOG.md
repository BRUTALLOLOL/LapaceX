# Changelog

All notable changes to LaplaceX are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] — 2026-05-23

### Added
- **C++ core** (`laplacex::core`) — adaptive Gauss-Kronrod G7/K15 quadrature
  for the forward transform with exponential axis compression; parallelised
  batch evaluation via `std::async`.
- **Cubic-spline forward transform** from discrete samples.
- **Weeks–Laguerre inverse** with automatic parameter selection.
- **Critical-points module** (`laplacex::critical`) — continuous wavelet
  transform (Mexican Hat) in the logarithmic time plane; ridge-line extraction;
  AIC-based component pruning.
- **Reconstruction module** — time-domain and Laplace-domain reconstruction
  from `ExponentialComponent` lists.
- **Utilities** — test-signal generators, noise estimation, log-plane
  resampling, AIC/BIC criteria.
- **pybind11 bindings** — full C++ surface exposed to Python with NumPy
  array support.
- **Python API** — `laplace()`, `inverse()`, `decompose()`, `reconstruct()`,
  `LaplaceTransform`, `ExponentialComponents`.
- **CI/CD** — GitHub Actions matrix (Linux, macOS Intel, macOS ARM, Windows),
  cibuildwheel for pre-built wheels, OIDC PyPI publishing.
- **Full pytest suite** — 40+ tests covering correctness, edge cases, and
  round-trip behaviour.
