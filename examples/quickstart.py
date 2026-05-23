"""
examples/quickstart.py
======================
A minimal end-to-end demonstration of LaplaceX:

  1. Generate a multi-exponential test signal
  2. Forward Laplace transform at a set of complex frequencies
  3. Decompose into exponential components
  4. Reconstruct and compare with original
  5. (Optional) Plot if matplotlib is available

Run with:
    python examples/quickstart.py
"""

import numpy as np
import laplacex
from laplacex import laplace, inverse, decompose, reconstruct, utils

print(f"LaplaceX {laplacex.__version__}\n")

# ---------------------------------------------------------------------------
# 1. Test signal:  f(t) = 2*exp(-0.5t) + 0.5*exp(-5t)
# ---------------------------------------------------------------------------
t = np.linspace(0.01, 15, 600)
f = utils.gen_exp_sum(t,
    amplitudes=np.array([2.0, 0.5]),
    rates=np.array([0.5, 5.0]),
)
print(f"Signal shape: {f.shape},  range [{f.min():.4f}, {f.max():.4f}]")

# ---------------------------------------------------------------------------
# 2. Forward transform at a few complex frequencies
# ---------------------------------------------------------------------------
s_vals = np.array([1+0j, 2+0j, 1+2j])
F_vals = laplace(f, s_vals, t=t)
print("\nForward transform F(s):")
for s, F in zip(s_vals, F_vals):
    analytic = 2/(s+0.5) + 0.5/(s+5)
    print(f"  s={s}:  computed={F:.6f},  analytic={analytic:.6f},  "
          f"err={abs(F-analytic):.2e}")

# ---------------------------------------------------------------------------
# 3. Decompose into exponential components
# ---------------------------------------------------------------------------
ec = decompose(t, f)
print(f"\nDecomposition found {len(ec)} component(s):")
for i, c in enumerate(ec):
    print(f"  [{i}] amplitude={c.amplitude:.4f}  rate={c.rate:.4f}  "
          f"decay_time={c.decay_time:.4f}  significance={c.significance:.3f}")

# ---------------------------------------------------------------------------
# 4. Reconstruct & residual
# ---------------------------------------------------------------------------
f_rec = reconstruct(ec, t)
residual = np.abs(f_rec - f)
print(f"\nReconstruction residual:  max={residual.max():.4e}  "
      f"mean={residual.mean():.4e}")

# ---------------------------------------------------------------------------
# 5. Inverse transform via Weeks method (independent check)
# ---------------------------------------------------------------------------
def F_analytic(s):
    return 2/(s + 0.5) + 0.5/(s + 5)

t_inv = np.linspace(0.5, 8, 40)
f_inv = inverse(F_analytic, t_inv, sigma=1.0, b=0.5, N=128)
f_ref = utils.gen_exp_sum(t_inv,
    amplitudes=np.array([2.0, 0.5]),
    rates=np.array([0.5, 5.0]),
)
err_inv = np.max(np.abs(f_inv - f_ref))
print(f"\nWeeks inverse max error: {err_inv:.4e}")

# ---------------------------------------------------------------------------
# 6. Optional plot
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(t, f, "k-", lw=1, label="original")
    ax.plot(t, f_rec, "C1--", lw=1.5, label="reconstruction")
    ax.set_xlabel("t");  ax.set_ylabel("f(t)")
    ax.set_title("Signal & Reconstruction")
    ax.legend();  ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.stem(ec.decay_times, ec.amplitudes, markerfmt="C0o",
             linefmt="C0-", basefmt="k-")
    ax2.set_xscale("log")
    ax2.set_xlabel("τ = 1/λ (relaxation time)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Exponential Spectrum")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    plt.savefig("laplacex_quickstart.png", dpi=150)
    print("\nPlot saved to laplacex_quickstart.png")
    plt.show()
except ImportError:
    print("\n(matplotlib not installed — skipping plot)")

print("\nDone.")
