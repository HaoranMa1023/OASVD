"""Online Adaptive SVD (OASVD) core package.

This package provides building blocks for implementing streaming low‑rank approximation
algorithms with adaptive rank control.  The main components include:

* :mod:`incremental_svd` – incremental updates of truncated SVDs for low‑rank matrix updates;
* :mod:`spectral_probe` – routines for estimating residual spectral energy via randomised sketches;
* :mod:`control_law` – the OASVD control loop combining incremental SVD updates, spectral
  probing, hysteresis‑based rank adaptation, error feedback and optional re‑orthogonalisation;
* :mod:`baselines` – baseline algorithms (fixed‑rank iSVD, full SVD with energy threshold);
* :mod:`metrics` – helper functions for computing error, coverage, jitter and orthogonality
  statistics;
* :mod:`plotting` – utilities for producing the figures used in our paper;
* :mod:`utils` – miscellaneous helpers (random seeding, timers, configuration parsing).

The top‑level API exports the :class:`OASVD` class for convenience.

"""

from .control_law import OASVD  # noqa: F401

__all__ = ["OASVD"]