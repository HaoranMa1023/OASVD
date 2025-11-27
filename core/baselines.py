"""Baseline streaming low-rank approximation methods.

This module provides reference implementations of simple baseline algorithms
against which OASVD can be compared.  These include:

* Fixed-rank incremental SVD (iSVD), which maintains a static truncation rank
  throughout the stream.
* Full SVD with energy threshold (and an optional randomized SVD variant),
  which recomputes a (truncated) SVD of the entire matrix at a prescribed
  frequency.  This baseline serves as an oracle in terms of approximation
  quality but is generally too expensive for large problems.
* A sketch-based streaming baseline based on Frequent Directions (FD), which
  maintains a row sketch of the current matrix and uses it to approximate
  the dominant singular directions.  This is mainly used in moderate-scale
  experiments to illustrate the behaviour of sketching methods under
  piecewise-spectral / shock scenarios.

All baselines expose a common interface: given an initial matrix A0 and a
list of rank-1 updates, they return time series of singular values, ranks
and relative approximation errors.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd, norm

from .incremental_svd import incremental_update


# ---------------------------------------------------------------------------
# 1. Fixed-rank incremental SVD (Fixed-r iSVD)
# ---------------------------------------------------------------------------

def fixed_rank_isvd(A0: np.ndarray,
                    updates: list[tuple[np.ndarray, np.ndarray]],
                    r: int) -> dict[str, list]:
    """Run a fixed-rank incremental SVD on a stream of rank-1 updates.

    Parameters
    ----------
    A0 : ndarray of shape (m, n)
        Initial matrix used to initialise the truncated SVD.
    updates : list of tuples (u, v)
        Sequence of rank-1 perturbations to apply.  Each update is an
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

    errs: list[float] = []
    sigmas: list[np.ndarray] = []
    ranks: list[int] = []

    A_current = A0.copy()
    # Compute initial relative error
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


# ---------------------------------------------------------------------------
# 2. Full / randomized SVD with energy threshold (Full-SVD-energy oracle)
# ---------------------------------------------------------------------------

def _truncated_svd_energy(A: np.ndarray,
                          energy_thresh: float,
                          method: str = "full",
                          oversample: int = 10,
                          n_power_iter: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an energy-thresholded truncated SVD of A.

    Parameters
    ----------
    A : ndarray of shape (m, n)
        Input matrix.
    energy_thresh : float
        Fraction of Frobenius energy to retain (e.g. 0.99 for 99%).
    method : {'full', 'rsvd'}, optional
        If 'full', use a standard dense SVD.  If 'rsvd', use a simple
        randomized SVD (suitable for moderately large problems).
    oversample : int, optional
        Oversampling parameter for randomized SVD.
    n_power_iter : int, optional
        Number of power iterations for randomized SVD.

    Returns
    -------
    U_k, s_k, Vt_k : ndarray
        Truncated SVD factors such that the retained energy fraction is
        at least ``energy_thresh``.
    """
    m, n = A.shape

    if method == "rsvd":
        # Simple Gaussian randomized SVD; suitable for experiments
        r_guess = min(m, n)
        # Use a slightly conservative sample size; we will truncate by energy anyway
        k0 = min(r_guess, oversample + 50)
        rng = np.random.default_rng()
        Omega = rng.standard_normal(size=(n, k0))
        Y = A @ Omega
        for _ in range(n_power_iter):
            Y = A @ (A.T @ Y)
        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = Q.T @ A
        Ub, s, Vtb = svd(B, full_matrices=False)
        U_full = Q @ Ub
        Vt_full = Vtb
    else:
        # Dense full SVD
        U_full, s, Vt_full = svd(A, full_matrices=False)

    # Energy-based truncation
    total_energy = float(np.sum(s ** 2))
    if total_energy <= 0.0:
        # Degenerate case; return rank-0 factors
        return U_full[:, :0], s[:0], Vt_full[:0, :]

    cumulative = np.cumsum(s ** 2)
    # Smallest k such that energy >= energy_thresh * total_energy
    k = int(np.searchsorted(cumulative, energy_thresh * total_energy) + 1)
    k = max(1, min(k, len(s)))

    U_k = U_full[:, :k].copy()
    s_k = s[:k].copy()
    Vt_k = Vt_full[:k, :].copy()
    return U_k, s_k, Vt_k


def full_svd_energy(A0: np.ndarray,
                    updates: list[tuple[np.ndarray, np.ndarray]],
                    energy_thresh: float,
                    recompute_every: int = 1,
                    method: str = "full") -> dict[str, list]:
    """Full SVD baseline with energy threshold.

    This baseline recomputes a (possibly randomized) SVD of the accumulated
    matrix at a prescribed frequency and truncates the singular values so that
    the retained Frobenius energy satisfies

        sum(s_i^2) / sum(all singular values^2) >= energy_thresh.

    For ``recompute_every = 1`` and ``method = 'full'``, this corresponds to
    the strongest oracle baseline described in the paper, where a full SVD is
    recomputed at every step.

    Parameters
    ----------
    A0 : ndarray of shape (m, n)
        Initial matrix.
    updates : list of tuples (u, v)
        Rank-1 updates ``Δ_t = u_t v_t^T``.
    energy_thresh : float
        Fraction of energy to retain (e.g. 0.99 for 99 % energy).
    recompute_every : int, optional
        Period (in time steps) at which to recompute the SVD.  If set to 1,
        the SVD is recomputed at every step.  Larger values allow a cheaper
        (but less oracle-like) baseline.
    method : {'full', 'rsvd'}, optional
        Backend for the SVD computation.  'full' uses a dense SVD; 'rsvd'
        uses a simple randomized SVD.

    Returns
    -------
    results : dict
        Dictionary with singular values, ranks and relative errors over time.
    """
    A_current = A0.copy()
    m, n = A_current.shape

    errs: list[float] = []
    sigmas: list[np.ndarray] = []
    ranks: list[int] = []

    # Initial truncated SVD
    U, s, Vt = _truncated_svd_energy(A_current, energy_thresh, method=method)
    approx = (U * s) @ Vt
    rel_err = norm(A_current - approx, 'fro') / max(1.0, norm(A_current, 'fro'))
    errs.append(rel_err)
    sigmas.append(s.copy())
    ranks.append(len(s))

    # Apply updates
    for t, (u_vec, v_vec) in enumerate(updates, start=1):
        A_current += np.outer(u_vec, v_vec)

        if t % recompute_every == 0:
            U, s, Vt = _truncated_svd_energy(A_current, energy_thresh, method=method)

        # Use the latest (possibly slightly stale) factors to form an approximation
        approx = (U * s) @ Vt
        rel_err = norm(A_current - approx, 'fro') / max(1.0, norm(A_current, 'fro'))
        errs.append(rel_err)
        sigmas.append(s.copy())
        ranks.append(len(s))

    return {'s': sigmas, 'r': ranks, 'err': errs}


# ---------------------------------------------------------------------------
# 3. Frequent Directions sketch baseline (streaming sketch)
# ---------------------------------------------------------------------------

def _frequent_directions_sketch(A: np.ndarray,
                                ell: int) -> np.ndarray:
    """Compute a Frequent Directions sketch of the rows of A.

    This implements the classical FD algorithm for a row stream X_i in R^d,
    using the rows of A as the stream.  It returns a sketch matrix B such
    that B^T B approximates A^T A.

    Parameters
    ----------
    A : ndarray of shape (m, n)
        Input matrix whose rows define the stream.
    ell : int
        Sketch size (number of rows in the FD sketch).  Typical values are
        on the order of the target rank or a small multiple thereof.

    Returns
    -------
    B : ndarray of shape (ell, n)
        Frequent Directions sketch matrix.
    """
    m, n = A.shape
    B = np.zeros((ell, n), dtype=A.dtype)
    next_row = 0

    for i in range(m):
        x = A[i, :]
        B[next_row, :] = x
        next_row += 1

        if next_row == ell:
            # Sketch is full: perform shrinkage via SVD
            U, s, Vt = svd(B, full_matrices=False)
            # Classical FD shrinkage: subtract the squared (ell//2)-th singular value
            # (other variants use sigma_ell; here we follow a common choice for clarity)
            idx = max(0, min(ell - 1, ell // 2))
            delta = s[idx] ** 2
            s_shrink = np.sqrt(np.maximum(s ** 2 - delta, 0.0))
            B = (s_shrink[:, None] * Vt)
            # Count how many rows remain effectively nonzero
            next_row = int(np.count_nonzero(s_shrink > 0))

    return B


def frequent_directions_baseline(A0: np.ndarray,
                                 updates: list[tuple[np.ndarray, np.ndarray]],
                                 ell: int,
                                 r: int | None = None) -> dict[str, list]:
    """Frequent Directions (FD) sketch baseline.

    This baseline uses the FD algorithm to maintain a row sketch of the
    current matrix ``A_t`` at each step, and then derives an approximate
    low-rank reconstruction via the sketch.  It is primarily intended for
    moderate-scale synthetic experiments, as it requires access to the
    full matrix ``A_t`` to compute the sketch and the reconstruction.

    Conceptually, this corresponds to the ``Streaming sketch baseline
    (Frequent Directions)`` in the experiment design: it represents a
    non-SVD, sketch-based streaming method with theoretical covariance
    approximation guarantees, but without explicit spectral-phase control.

    Parameters
    ----------
    A0 : ndarray of shape (m, n)
        Initial matrix.
    updates : list of tuples (u, v)
        Rank-1 updates ``Δ_t = u_t v_t^T``.
    ell : int
        Sketch size used by Frequent Directions (number of rows of the
        sketch matrix).  Must satisfy ``ell >= 1``.
    r : int, optional
        Target truncation rank used to form the final approximation from
        the sketch.  If ``None``, the rank is chosen as ``min(ell, min(m,n))``.

    Returns
    -------
    results : dict
        Dictionary with keys ``'s'`` (approximate singular values over time),
        ``'r'`` (chosen rank over time) and ``'err'`` (relative errors).
    """
    if ell <= 0:
        raise ValueError("ell (sketch size) must be a positive integer.")

    A_current = A0.copy()
    m, n = A_current.shape

    errs: list[float] = []
    sigmas: list[np.ndarray] = []
    ranks: list[int] = []

    def _approx_from_sketch(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build an approximate rank-r SVD of A from its FD sketch."""
        B = _frequent_directions_sketch(A, ell=ell)  # shape (ell, n)
        # SVD of the sketch
        Ub, sb, Vtb = svd(B, full_matrices=False)
        # Decide rank to use
        if r is None:
            r_use = min(len(sb), ell, m, n)
        else:
            r_use = min(r, len(sb), ell, m, n)
        if r_use == 0:
            return Ub[:, :0], sb[:0], Vtb[:0, :]

        V_r = Vtb[:r_use, :].T          # shape (n, r_use)
        sigma_r = sb[:r_use]            # shape (r_use,)

        # Avoid division by zero
        mask = sigma_r > 1e-12
        if not np.any(mask):
            return Ub[:, :0], sb[:0], Vtb[:0, :]

        V_r_nz = V_r[:, mask]          # (n, r_eff)
        sigma_nz = sigma_r[mask]       # (r_eff,)

        # Approximate left factors via U ≈ A V Σ^{-1}
        U_approx = A @ (V_r_nz / sigma_nz[None, :])  # (m, r_eff)

        return U_approx, sigma_nz, V_r_nz.T          # U, s, Vt

    # Initial approximation
    U_approx, s_approx, Vt_approx = _approx_from_sketch(A_current)
    if s_approx.size > 0:
        approx = (U_approx * s_approx) @ Vt_approx
        rel_err = norm(A_current - approx, 'fro') / max(1.0, norm(A_current, 'fro'))
        errs.append(rel_err)
        sigmas.append(s_approx.copy())
        ranks.append(len(s_approx))
    else:
        errs.append(1.0)
        sigmas.append(s_approx.copy())
        ranks.append(0)

    # Process updates
    for u_vec, v_vec in updates:
        A_current += np.outer(u_vec, v_vec)
        U_approx, s_approx, Vt_approx = _approx_from_sketch(A_current)
        if s_approx.size > 0:
            approx = (U_approx * s_approx) @ Vt_approx
            rel_err = norm(A_current - approx, 'fro') / max(1.0, norm(A_current, 'fro'))
            errs.append(rel_err)
            sigmas.append(s_approx.copy())
            ranks.append(len(s_approx))
        else:
            errs.append(1.0)
            sigmas.append(s_approx.copy())
            ranks.append(0)

    return {'s': sigmas, 'r': ranks, 'err': errs}
