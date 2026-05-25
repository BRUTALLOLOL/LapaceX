# LaplaceX

**Industrial-grade Python library for stable forward and inverse Laplace transforms.**

LaplaceX operates in the *logarithmic time plane* and decomposes signals into
exponential components via continuous wavelet analysis, achieving significantly
better stability than classical Mellin-integral inversion — especially in the
presence of noise and at the tails of the signal.

| Feature | LaplaceX | scipy.signal / classical |
|---|---|---|
| Ill-posedness handling | Wavelet ridge + physical regularisation | Tikhonov / none |
| Gibbs oscillations | None (localised error) | Common |
| Interpretable output | Relaxation times + amplitudes | No |
| C++ core | ✓ (pybind11) | NumPy / LAPACK |
| Pre-built wheels | ✓ (cibuildwheel, all major platforms) | — |

---


**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24.
Optional extras: `matplotlib` (plotting), `pandas` (DataFrame export).

---

## Quick Start

```python
import numpy as np
from laplacex import laplace, inverse, decompose, reconstruct, utils

# --- 1. Forward transform ---------------------------------------------------
# F(s) for f(t) = exp(-2t)  →  should equal 1/(s+2)
s_vals = np.array([1+0j, 2+0j, 3+1j])
F = laplace(lambda t: np.exp(-2*t), s_vals)

# --- 2. Inverse transform  ---------------------------
t = np.linspace(0.1, 5, 100)
f_rec = inverse(lambda s: 1/(s+2), t)          # recovers exp(-2t)

# --- 3. Decompose a noisy multi-exponential signal --------------------------
t   = np.linspace(0.01, 15, 600)
f   = utils.gen_exp_sum(t, [2.0, 0.5], [0.5, 5.0])
f_n = utils.add_noise(f, sigma=0.02, seed=0)

ec = decompose(t, f_n)       # → ExponentialComponents
print(ec)
# ExponentialComponents(2 components, significances=[0.832, 0.168])

for c in ec:
    print(f"  τ = {c.decay_time:.3f}   A = {c.amplitude:.3f}")

# --- 4. Reconstruct ---------------------------------------------------------
f_hat = reconstruct(ec, t)

# --- 5. Plot (requires matplotlib) -----------------------------------------
ec.plot(t_eval=t, t_original=t, f_original=f_n)
```

---

## API Reference

### Free functions

| Function | Description |
|---|---|
| `laplace(f, s, t=None)` | Forward Laplace transform F(s) |
| `inverse(F, t_eval, ...)` | Inverse Laplace transform f(t) |
| `decompose(t, f, cfg=None)` | Wavelet decomposition into `ExponentialComponents` |
| `reconstruct(comps, t_eval)` | Reconstruct f(t) from components |

### Classes

#### `LaplaceTransform`

Stateful engine — configure once, call many times.  Caches decomposition
results so that `inverse(..., method='auto')` uses the component
representation automatically.

```python
from laplacex import LaplaceTransform

lt = LaplaceTransform(tol=1e-12, weeks_N=128)
F  = lt.laplace(lambda t: np.exp(-t), s=2+1j)
ec = lt.decompose(t, f)             # cached internally
f_rec = lt.inverse(None, t_eval, method='auto')
```

#### `ExponentialComponents`

Container for decomposition results with NumPy arrays, filtering, export,
and visualisation.

```python
ec.amplitudes    # ndarray, shape (n,)
ec.rates         # ndarray, decay rates λ_k
ec.decay_times   # ndarray, τ_k = 1/λ_k
ec.significances # ndarray, relative energy

ec.filter(min_significance=0.1)
ec.top(3)
ec.reconstruct(t_eval)
ec.reconstruct_laplace(s_eval)
ec.to_dict()
ec.to_dataframe()   # requires pandas
ec.plot(t_eval, t_original, f_original)
```

### Configuration

```python
from laplacex import TransformConfig, DecomposeConfig

cfg = TransformConfig()
cfg.tol        = 1e-12     # absolute quadrature tolerance
cfg.rel_tol    = 1e-10
cfg.max_levels = 25
cfg.t_max      = 1e8

dcfg = DecomposeConfig()
dcfg.n_scales      = 128   # wavelet scales
dcfg.n_log_points  = 1024  # log-plane grid resolution
dcfg.wavelet       = "mexh"
dcfg.snr_threshold = 5.0
dcfg.use_aic       = True   # AIC-based pruning
```

### Utilities (`laplacex.utils`)

```python
from laplacex import utils

t = np.linspace(0.01, 10, 500)
f = utils.gen_exp_sum(t, [1.0, 0.5], [1.0, 3.0])
f = utils.add_noise(f, sigma=0.05, seed=42)
s = utils.estimate_noise(f)
x, ft = utils.to_log_plane(t, f, n=512)
```

---

## Theoretical Background

LaplaceX is based on the observation that the Laplace kernel
`K(s, x) = exp(-s·exp(x))` is *shift-invariant and localised* in logarithmic
coordinates `x = ln(t)`. An error in F(s) therefore affects only the
corresponding time scale, preventing error propagation across the whole signal
— in sharp contrast to classical Mellin inversion.

The decomposition algorithm:

1. Transforms the signal to the log plane: `f̃(x) = f(eˣ)·eˣ`
2. Computes the continuous wavelet transform (Mexican Hat)
3. Extracts ridge lines — chains of local wavelet maxima across scales
4. Reads off exponential parameters from each ridge's peak energy
5. Prunes noise-level components via SNR threshold + optional AIC

See the project document `docs/theory.md` for the full mathematical derivation.

---

## License

MIT © 2026 Vozmishchev Artem
