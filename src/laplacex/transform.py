"""
laplacex.transform
==================
``LaplaceTransform`` — stateful object that wraps the C++ core and
exposes the full transform API with configurable precision and
automatic method selection.
"""

from __future__ import annotations

from typing import Callable, Optional, Union
import numpy as np

from laplacex._laplacex import (
    TransformConfig,
    DecomposeConfig,
    ExponentialComponent,
    forward          as _fwd,
    forward_batch    as _fwd_batch,
    forward_discrete as _fwd_disc,
    inverse_weeks    as _inv_weeks,
    decompose        as _decompose,
    reconstruct      as _reconstruct,
    reconstruct_laplace as _reconstruct_laplace,
)
from laplacex.components import ExponentialComponents


class LaplaceTransform:
    """
    High-level Laplace transform engine.

    Parameters
    ----------
    tol : float
        Absolute quadrature tolerance (default 1e-10).
    rel_tol : float
        Relative quadrature tolerance (default 1e-8).
    max_levels : int
        Maximum adaptive bisection depth (default 20).
    t_max : float
        Effective upper integration limit (default 1e6).
    weeks_N : int
        Number of Laguerre terms used by the Weeks inverse method (default 64).
    decompose_cfg : DecomposeConfig, optional
        Full configuration for the exponential-component decomposition.
        If *None*, a default ``DecomposeConfig`` is used.

    Examples
    --------
    >>> import numpy as np
    >>> from laplacex import LaplaceTransform
    >>> lt = LaplaceTransform(tol=1e-12)
    >>> F = lt.laplace(lambda t: np.exp(-2*t), s=3+0j)
    >>> float(F.real)  # should be ≈ 1/5 = 0.2
    0.2
    """

    def __init__(
        self,
        tol: float = 1e-10,
        rel_tol: float = 1e-8,
        max_levels: int = 20,
        t_max: float = 1e6,
        weeks_N: int = 64,
        decompose_cfg: Optional[DecomposeConfig] = None,
    ) -> None:
        self._cfg = TransformConfig()
        self._cfg.tol        = tol
        self._cfg.rel_tol    = rel_tol
        self._cfg.max_levels = max_levels
        self._cfg.t_max      = t_max

        self._weeks_N = weeks_N
        self._dcfg    = decompose_cfg or DecomposeConfig()

        # Optional cache: if the user already ran decompose(), store result here
        # so that inverse() can use the component representation automatically.
        self._components: Optional[ExponentialComponents] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> TransformConfig:
        """The underlying ``TransformConfig`` struct (mutable)."""
        return self._cfg

    @property
    def decompose_config(self) -> DecomposeConfig:
        """The underlying ``DecomposeConfig`` struct (mutable)."""
        return self._dcfg

    @property
    def cached_components(self) -> Optional[ExponentialComponents]:
        """Components cached from the last :meth:`decompose` call, or *None*."""
        return self._components

    # ------------------------------------------------------------------
    # Forward transform
    # ------------------------------------------------------------------
    def laplace(
        self,
        f: Union[Callable[[float], float], np.ndarray],
        s: Union[complex, np.ndarray],
        t: Optional[np.ndarray] = None,
    ) -> Union[complex, np.ndarray]:
        """
        Compute the forward Laplace transform F(s) = ∫₀^∞ f(t) exp(-st) dt.

        Parameters
        ----------
        f : callable or ndarray
            The function f(t).  If an ndarray, ``t`` must be supplied.
        s : complex or ndarray of complex
            One or more complex frequencies.
        t : ndarray, optional
            Time points corresponding to ``f`` when ``f`` is an array.

        Returns
        -------
        complex or ndarray of complex
            F(s) at the requested frequencies.
        """
        scalar_s = np.isscalar(s) or (hasattr(s, 'shape') and s.shape == ())

        if callable(f):
            s_arr = np.atleast_1d(np.asarray(s, dtype=complex))
            result = _fwd_batch(f, s_arr, self._cfg)
            return complex(result[0]) if scalar_s else np.array(result)
        else:
            # Discrete samples
            if t is None:
                raise ValueError(
                    "When `f` is an array, `t` (time points) must be provided."
                )
            t_arr = np.asarray(t, dtype=float)
            f_arr = np.asarray(f, dtype=float)
            s_arr = np.atleast_1d(np.asarray(s, dtype=complex))
            result = _fwd_disc(t_arr, f_arr, s_arr, self._cfg)
            return complex(result[0]) if scalar_s else np.array(result)

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------
    def inverse(
        self,
        F: Callable[[complex], complex],
        t_eval: np.ndarray,
        *,
        sigma: float = 0.0,
        b: float = 0.0,
        method: str = "auto",
    ) -> np.ndarray:
        """
        Compute the inverse Laplace transform f(t) from F(s).

        Parameters
        ----------
        F : callable
            The Laplace image F(s).
        t_eval : array-like
            Time points at which f(t) is desired.
        sigma, b : float
            Weeks-method parameters (auto-selected when zero).
        method : {"auto", "weeks", "components"}
            "auto"       → use cached components if available, else Weeks.
            "weeks"      → always use the Weeks–Laguerre method.
            "components" → always use the cached component representation
                           (raises if no components are cached).

        Returns
        -------
        ndarray of float
            f(t) at the requested time points.
        """
        t_arr = np.asarray(t_eval, dtype=float)

        if method == "components" or (method == "auto" and self._components is not None):
            if self._components is None:
                raise RuntimeError(
                    "No cached components — run decompose() first, or use method='weeks'."
                )
            return np.asarray(
                _reconstruct(self._components._comps, t_arr)
            )

        # Weeks fallback
        return np.asarray(
            _inv_weeks(F, t_arr, sigma, b, self._weeks_N, self._cfg)
        )

    # ------------------------------------------------------------------
    # Decompose
    # ------------------------------------------------------------------
    def decompose(
        self,
        t: np.ndarray,
        f: np.ndarray,
        *,
        cache: bool = True,
    ) -> ExponentialComponents:
        """
        Decompose a time-series into exponential components.

        Parameters
        ----------
        t : array-like
            Strictly increasing time values (t > 0).
        f : array-like
            Signal values at ``t``.
        cache : bool
            If *True* (default), store the result so that subsequent
            :meth:`inverse` calls can use the component representation.

        Returns
        -------
        ExponentialComponents
        """
        t_arr = np.asarray(t, dtype=float)
        f_arr = np.asarray(f, dtype=float)
        comps = _decompose(t_arr, f_arr, self._dcfg)
        ec = ExponentialComponents(comps)
        if cache:
            self._components = ec
        return ec

    # ------------------------------------------------------------------
    # Reconstruct
    # ------------------------------------------------------------------
    def reconstruct(
        self,
        components: ExponentialComponents,
        t_eval: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct f(t) from an :class:`ExponentialComponents` object.

        Parameters
        ----------
        components : ExponentialComponents
        t_eval : array-like

        Returns
        -------
        ndarray of float
        """
        t_arr = np.asarray(t_eval, dtype=float)
        return np.asarray(_reconstruct(components._comps, t_arr))

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"LaplaceTransform(tol={self._cfg.tol}, "
            f"rel_tol={self._cfg.rel_tol}, "
            f"max_levels={self._cfg.max_levels})"
        )
