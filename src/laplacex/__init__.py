"""
LaplaceX
========
Industrial-grade Laplace transform library operating in the logarithmic
time plane for superior numerical stability.

Author : Vozmishchev Artem
License: MIT
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__  = "Vozmishchev Artem"

# ---------------------------------------------------------------------------
# Re-export C++ core objects
# ---------------------------------------------------------------------------
from laplacex._laplacex import (           # noqa: F401
    TransformConfig,
    DecomposeConfig,
    ExponentialComponent,
    forward          as _forward_cpp,
    forward_batch    as _forward_batch_cpp,
    forward_discrete as _forward_discrete_cpp,
    inverse_weeks    as _inverse_weeks_cpp,
    decompose        as _decompose_cpp,
    reconstruct      as _reconstruct_cpp,
    reconstruct_laplace as _reconstruct_laplace_cpp,
    utils,
)

from laplacex.transform    import LaplaceTransform          # noqa: F401
from laplacex.components   import ExponentialComponents     # noqa: F401
from laplacex.api          import laplace, inverse, decompose, reconstruct  # noqa: F401

__all__ = [
    # High-level API
    "laplace",
    "inverse",
    "decompose",
    "reconstruct",
    # Classes
    "LaplaceTransform",
    "ExponentialComponents",
    # Config structs
    "TransformConfig",
    "DecomposeConfig",
    "ExponentialComponent",
    # Utilities sub-module
    "utils",
]
