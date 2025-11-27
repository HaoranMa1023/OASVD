"""Residual spectral probing via randomised sketches.

This module provides a routine to estimate the dominant singular values
of the residual

    R = A - U diag(s) Vt,

where A is represented in factor form as

    A = U diag(s) Vt + sum_i u_i v_i^T.

The residual is probed without forming A explicitly, by applying A to a
Gaussian test matrix and then projecting out the contribution living in
the current subspace spanned by U.  An SVD of a small sketch of the
residual is then used to obtain approximate singular values.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd


def apply_A(U: np.ndarray,
            s: np.ndarray,
            Vt: np.ndarray,
            increments: list[tuple[np.ndarray, np.ndarray]],
            X: np.ndarray) -> np.ndarray:
    """Apply the matrix ``A = U diag(s) Vt + sum_i u_i v_i^T`` to a matrix ``X``.

    Parameters
    ----------
    U : ndarray of shape (m, r)
        Left singular vectors of the approximation.
    s : ndarray of shape (r,)
        Singular values.
    Vt : ndarray of shape (r, n)
        Right singular vectors (transposed).
    increments : list of tuples (u, v)
        List of recent rank-1 updates, where each ``u`` has shape (m,) and
        each ``v`` has shape (n,).
    X : ndarray of shape (n, p)
        Test matrix to which ``A`` is applied.

    Returns
    -------
    Y : ndarray of shape (m, p)
        The product ``A @ X``.
    """
    VtX = Vt @ X                  # (r, p)
    US  = U * s                   # (m, r)
    Y   = US @ VtX                # (m, p)

    for u_i, v_i in increments:
        Y += np.outer(u_i, v_i @ X)

    return Y


def spectral_probe(U: np.ndarray,
                   s: np.ndarray,
                   Vt: np.ndarray,
                   increments: list[tuple[np.ndarray, np.ndarray]],
                   p: int | None = None,
                   rng: np.random.Generator | None = None
                   ) -> np.ndarray:
    """Estimate singular values of the residual using randomised spectral probing.

    This routine performs a randomised SVD targeting the residual

        R = A - U diag(s) Vt,

    where ``A`` is represented by ``(U, s, Vt)`` and ``increments``.
    It returns approximate singular values of ``R`` based on a sketch
    of the form ``Q^T A``, where ``Q`` spans the range of

        R Ω = (I - U U^T) A Ω

    with Ω a Gaussian random matrix.

    Parameters
    ----------
    U, s, Vt : see :func:`apply_A`
    increments : list of rank-1 perturbations
    p : int, optional
        Target sketch dimension.  Defaults to ``len(s) + 2`` but is capped
        at ``n`` to avoid oversized sketches.
    rng : numpy.random.Generator, optional
        Random number generator to use.  A default generator is created if
        ``None``.

    Returns
    -------
    s_probe : ndarray
        Approximate singular values of the residual ``R`` (in descending order).
        Only the first ``p`` values are returned.
    """
    m, r = U.shape
    n = Vt.shape[1]

    if rng is None:
        rng = np.random.default_rng()
    if p is None:
        p = min(max(2, r + 2), n)

    # 1) Draw a random Gaussian test matrix Ω (n × p)
    Omega = rng.standard_normal(size=(n, p))

    # 2) Compute Y_full = A Ω in factor form
    Y_full = apply_A(U, s, Vt, increments, Omega)  # (m, p)

    # 3) Project out the contribution in span(U) to obtain a residual sketch:
    #    Y_res ≈ R Ω = (I - U U^T) A Ω
    UtY = U.T @ Y_full                 # (r, p)
    Y_res = Y_full - U @ UtY           # (m, p)

    # If the residual is (near) zero, early exit
    # (norms will be picked up in the singular values anyway)
    # 4) Orthonormalise the columns of Y_res
    Q, _ = np.linalg.qr(Y_res, mode="reduced")  # (m, p')

    # 5) Form B = Q^T A (p' × n) in factor form
    QtU = Q.T @ U                     # (p', r)
    B = (QtU * s) @ Vt                # contribution from U diag(s) Vt

    if increments:
        for u_i, v_i in increments:
            B += np.outer(Q.T @ u_i, v_i)  # (p', n)

    # 6) SVD of the small matrix B gives approximate singular values of R
    _, s_probe, _ = svd(B, full_matrices=False)

    return s_probe
