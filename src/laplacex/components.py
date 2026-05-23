"""
laplacex.components
===================
``ExponentialComponents`` — Python container around the C++ list of
``ExponentialComponent`` structs, with convenience methods for
visualisation, filtering, and export.
"""

from __future__ import annotations

from typing import List, Optional, Iterator
import numpy as np

from laplacex._laplacex import (
    ExponentialComponent,
    reconstruct      as _reconstruct,
    reconstruct_laplace as _reconstruct_laplace,
)


class ExponentialComponents:
    """
    Container for a list of :class:`ExponentialComponent` objects returned
    by :func:`laplacex.decompose`.

    The components are ordered by *significance* (descending).

    Parameters
    ----------
    comps : list of ExponentialComponent
        Raw C++ component objects.

    Examples
    --------
    >>> import numpy as np
    >>> from laplacex import decompose
    >>> t = np.linspace(0.01, 10, 500)
    >>> f = np.exp(-0.5 * t) + 0.3 * np.exp(-3 * t)
    >>> ec = decompose(t, f)
    >>> len(ec)            # number of detected components
    2
    >>> ec.amplitudes
    array([...])
    """

    def __init__(self, comps: List[ExponentialComponent]) -> None:
        self._comps: List[ExponentialComponent] = comps

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._comps)

    def __iter__(self) -> Iterator[ExponentialComponent]:
        return iter(self._comps)

    def __getitem__(self, idx: int) -> ExponentialComponent:
        return self._comps[idx]

    # ------------------------------------------------------------------
    # Numpy-friendly property arrays
    # ------------------------------------------------------------------
    @property
    def amplitudes(self) -> np.ndarray:
        """A_k for each component."""
        return np.array([c.amplitude for c in self._comps])

    @property
    def rates(self) -> np.ndarray:
        """Decay rates λ_k for each component."""
        return np.array([c.rate for c in self._comps])

    @property
    def decay_times(self) -> np.ndarray:
        """Relaxation times τ_k = 1/λ_k for each component."""
        return np.array([c.decay_time for c in self._comps])

    @property
    def frequencies(self) -> np.ndarray:
        """Oscillation frequencies ω_k (0 for purely real modes)."""
        return np.array([c.frequency for c in self._comps])

    @property
    def phases(self) -> np.ndarray:
        """Phase offsets φ_k."""
        return np.array([c.phase for c in self._comps])

    @property
    def significances(self) -> np.ndarray:
        """Relative energy of each component (sums to ≈ 1)."""
        return np.array([c.significance for c in self._comps])

    @property
    def log_times(self) -> np.ndarray:
        """Log-space positions x_k = ln(t_k)."""
        return np.array([c.log_time for c in self._comps])

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def filter(
        self,
        *,
        min_significance: float = 0.0,
        min_amplitude: float = 0.0,
        max_rate: Optional[float] = None,
        oscillatory_only: bool = False,
        real_only: bool = False,
    ) -> "ExponentialComponents":
        """
        Return a new :class:`ExponentialComponents` with components
        that satisfy all supplied criteria.

        Parameters
        ----------
        min_significance : float
            Discard components with significance < this value.
        min_amplitude : float
            Discard components with |amplitude| < this value.
        max_rate : float, optional
            Discard components with rate > max_rate.
        oscillatory_only : bool
            Keep only oscillatory components (frequency != 0).
        real_only : bool
            Keep only purely real exponential components (frequency == 0).
        """
        kept = []
        for c in self._comps:
            if c.significance < min_significance:
                continue
            if abs(c.amplitude) < min_amplitude:
                continue
            if max_rate is not None and c.rate > max_rate:
                continue
            if oscillatory_only and c.frequency == 0.0:
                continue
            if real_only and c.frequency != 0.0:
                continue
            kept.append(c)
        return ExponentialComponents(kept)

    def top(self, n: int) -> "ExponentialComponents":
        """Return the *n* most significant components."""
        return ExponentialComponents(self._comps[:n])

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------
    def reconstruct(self, t_eval: np.ndarray) -> np.ndarray:
        """
        Reconstruct f(t) = Σ A_k exp(-λ_k t) [cos(ω_k t + φ_k)] at *t_eval*.

        Parameters
        ----------
        t_eval : array-like

        Returns
        -------
        ndarray of float
        """
        t_arr = np.asarray(t_eval, dtype=float)
        return np.asarray(_reconstruct(self._comps, t_arr))

    def reconstruct_laplace(self, s_eval: np.ndarray) -> np.ndarray:
        """
        Compute F(s) = Σ A_k / (s + λ_k) analytically.

        Parameters
        ----------
        s_eval : array-like of complex

        Returns
        -------
        ndarray of complex
        """
        s_arr = np.asarray(s_eval, dtype=complex)
        return np.asarray(_reconstruct_laplace(self._comps, s_arr))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """
        Export all component arrays as a plain Python dict.

        Returns
        -------
        dict with keys: amplitudes, rates, decay_times, frequencies,
        phases, significances, log_times.
        """
        return {
            "amplitudes":   self.amplitudes.tolist(),
            "rates":        self.rates.tolist(),
            "decay_times":  self.decay_times.tolist(),
            "frequencies":  self.frequencies.tolist(),
            "phases":       self.phases.tolist(),
            "significances": self.significances.tolist(),
            "log_times":    self.log_times.tolist(),
        }

    def to_dataframe(self):
        """
        Export to a ``pandas.DataFrame`` (requires pandas).

        Returns
        -------
        pandas.DataFrame
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pandas is not installed. Install it with: pip install pandas"
            ) from exc
        return pd.DataFrame(self.to_dict())

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def plot(
        self,
        t_eval: Optional[np.ndarray] = None,
        t_original: Optional[np.ndarray] = None,
        f_original: Optional[np.ndarray] = None,
        *,
        figsize=(12, 5),
        show: bool = True,
    ):
        """
        Plot the component spectrum and optional reconstruction overlay.

        Requires *matplotlib*.

        Parameters
        ----------
        t_eval : array-like, optional
            Time grid for reconstruction curve.
        t_original, f_original : array-like, optional
            Original data to plot alongside reconstruction.
        figsize : tuple
            Matplotlib figure size.
        show : bool
            Call ``plt.show()`` at the end.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "matplotlib is not installed. Install it with: pip install matplotlib"
            ) from exc

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ---- Left: relaxation-time spectrum (stem plot) ----
        ax = axes[0]
        tau = self.decay_times
        amp = self.amplitudes
        ax.stem(tau, amp, markerfmt="C0o", linefmt="C0-", basefmt="k-")
        ax.set_xscale("log")
        ax.set_xlabel("Relaxation time τ = 1/λ")
        ax.set_ylabel("Amplitude")
        ax.set_title("Exponential Spectrum")
        ax.grid(True, which="both", alpha=0.3)

        # ---- Right: reconstruction vs original ----
        ax2 = axes[1]
        if t_original is not None and f_original is not None:
            ax2.plot(t_original, f_original, "k.", ms=2, alpha=0.5, label="data")
        if t_eval is not None:
            f_rec = self.reconstruct(np.asarray(t_eval, dtype=float))
            ax2.plot(t_eval, f_rec, "C1-", lw=1.5, label="LaplaceX reconstruction")
        ax2.set_xlabel("t")
        ax2.set_ylabel("f(t)")
        ax2.set_title("Reconstruction")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ExponentialComponents({len(self._comps)} components, "
            f"significances=[{', '.join(f'{s:.3f}' for s in self.significances[:5])}{'...' if len(self) > 5 else ''}])"
        )

    def __add__(self, other: "ExponentialComponents") -> "ExponentialComponents":
        """Concatenate two ExponentialComponents objects."""
        if not isinstance(other, ExponentialComponents):
            raise TypeError(f"Cannot add ExponentialComponents and {type(other)}")
        combined = sorted(
            self._comps + other._comps,
            key=lambda c: c.significance,
            reverse=True,
        )
        return ExponentialComponents(combined)
