"""Baseline streaming low‑rank approximation methods.

This module provides reference implementations of simple baseline algorithms
against which OASVD can be compared.  These include:

* Fixed‑rank incremental SVD (iSVD), which maintains a static truncation rank
  throughout the stream.
* Full SVD with energy threshold, which recomputes a (truncated) SVD of the
  entire matrix at every step.  This baseline serves as an oracle in terms
  of approximation quality but is generally too expensive for large problems.

Future extensions may include other sketch‑based streaming methods such as
Frequent Directions.  For now, we keep the baselines minimal to facilitate
comparison with OASVD.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd, norm

from .incremental_svd import incremental_update


def fixed_rank_isvd(A0: np.ndarray,
                    updates: list[tuple[np.ndarray, np.ndarray]],
                    r: int) -> dict[str, list]:
    """Run a fixed‑rank incremental SVD on a stream of rank‑1 updates.

    Parameters
    ----------
    A0 : ndarray of shape (m, n)
        Initial matrix used to initialise the truncated SVD.
    updates : list of tuples (u, v)
        Sequence of rank‑1 perturbations to apply.  Each update is an
        outer product ``Δ = u[:,None] * v[None,:]``.
    r : int
        Truncation rank to maintain throughout the stream.

    Returns
    -------
    results : dict
        Dictionary with keys ``'s'`` (singular values over time), ``'r'``
        (constant rank) and ``'err'`` (approximation error over time).
    """
    # Initial truncated SVD
    U, s, Vt = svd(A0, full_matrices=False)
    U = U[:, :r].copy()
    s = s[:r].copy()
    Vt = Vt[:r, :].copy()

    errs = []
    sigmas = []
    ranks = []

    A_current = A0.copy()
    # Compute initial error
    approx = (U * s) @ Vt
    err0 = norm(A_current - approx, 'fro') / max(1.0, norm(A_current, 'fro'))
    errs.append(err0)
    sigmas.append(s.copy())
    ranks.append(r)

    # Process each update
    for u_vec, v_vec in updates:
        A_current += np.outer(u_vec, v_vec)
        U, s, Vt = incremental_update(U, s, Vt, u_vec, v_vec)
        # Truncate back to rank r
        if len(s) > r:
            U = U[:, :r]
            s = s[:r].copy()
            Vt = Vt[:r, :]
        # Compute relative error
        approx = (U * s) @ Vt
        rel_err = norm(A_current - approx, 'fro') / max(1.0, norm(A_current, 'fro'))
        errs.append(rel_err)
        sigmas.append(s.copy())
        ranks.append(r)
    return {'s': sigmas, 'r': ranks, 'err': errs}


def full_svd_energy(A0: np.ndarray,
                    updates: list[tuple[np.ndarray, np.ndarray]],
                    energy_thresh: float) -> dict[str, list]:
    """Full SVD baseline with energy threshold.

    This baseline recomputes a full SVD of the accumulated matrix at each step
    and truncates the singular values so that the retained energy satisfies

        sum(s_i^2) / sum(all singular values^2) >= energy_thresh.

    Parameters
    ----------
    A0 : ndarray of shape (m, n)
        Initial matrix.
    updates : list of tuples (u, v)
        Rank‑1 updates.
    energy_thresh : float
        Fraction of energy to retain (e.g. 0.99 for 99 % energy).

    Returns
    -------
    results : dict
        Dictionary with singular values, ranks and relative errors over time.
    """
    A_current = A0.copy()
    errs = []
    sigmas = []
    ranks = []

    # Initial full SVD
    U, s, Vt = svd(A_current, full_matrices=False)
    # Determine truncation rank based on energy threshold
    total_energy = np.sum(s ** 2)
    cumulative = np.cumsum(s ** 2)
    # Find smallest k such that energy >= energy_thresh
    k = int(np.searchsorted(cumulative, energy_thresh * total_energy) + 1)
    U = U[:, :k].copy()
    s = s[:k].copy()
    Vt = Vt[:k, :].copy()
    approx = (U * s) @ Vt
    rel_err = norm(A_current - approx, 'fro') / max(1.0, norm(A_current, 'fro'))
    errs.append(rel_err)
    sigmas.append(s.copy())
    ranks.append(k)

    # Apply updates
    for u_vec, v_vec in updates:
        A_current += np.outer(u_vec, v_vec)
        U_f, s_f, Vt_f = svd(A_current, full_matrices=False)
        total_energy = np.sum(s_f ** 2)
        cumulative = np.cumsum(s_f ** 2)
        k = int(np.searchsorted(cumulative, energy_thresh * total_energy) + 1)
        U = U_f[:, :k].copy()
        s = s_f[:k].copy()
        Vt = Vt_f[:k, :].copy()
        approx = (U * s) @ Vt
        rel_err = norm(A_current - approx, 'fro') / max(1.0, norm(A_current, 'fro'))
        errs.append(rel_err)
        sigmas.append(s.copy())
        ranks.append(k)
    return {'s': sigmas, 'r': ranks, 'err': errs}