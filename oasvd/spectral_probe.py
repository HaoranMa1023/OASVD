"""Residual spectral probing via randomised sketches.

This module provides a simple routine to estimate the largest singular values
of the residual ``R = A - U S V.T`` without forming the full matrix ``A``.  It
uses a factor form of the matrix ``A`` described by an approximate SVD
``U @ diag(s) @ V.T`` plus a list of recent low‑rank increments.  A Gaussian
random test matrix is used to probe the action of the residual on a low
dimensional subspace, after which a small SVD is performed to obtain
approximate singular values of the residual.

The implementation here is intentionally straightforward and prioritises
simplicity over maximum efficiency.  It is adequate for use in the OASVD
control loop to detect when significant spectral energy is not captured by
the current truncated SVD.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd

def apply_A(U: np.ndarray, s: np.ndarray, Vt: np.ndarray,
            increments: list[tuple[np.ndarray, np.ndarray]],
            X: np.ndarray) -> np.ndarray:
    """Apply the matrix ``A = U diag(s) Vt + sum(u_i v_i^T)`` to a matrix ``X``.

    Parameters
    ----------
    U : ndarray of shape (m, r)
        Left singular vectors of the approximation.
    s : ndarray of shape (r,)
        Singular values.
    Vt : ndarray of shape (r, n)
        Right singular vectors (transposed).
    increments : list of tuples (u, v)
        List of recent rank‑1 updates, where each ``u`` has shape (m,) and
        each ``v`` has shape (n,).
    X : ndarray of shape (n, p)
        Test matrix to which ``A`` is applied.

    Returns
    -------
    Y : ndarray of shape (m, p)
        The product ``A @ X``.
    """
    # Contribution from the approximate SVD part
    # Compute Vt @ X (shape r×p), then multiply by diag(s) and U
    VtX = Vt @ X  # shape (r, p)
    # Multiply each row of VtX by s (broadcast along columns)
    US = U * s  # shape (m, r)
    Y = US @ VtX  # shape (m, p)
    # Contribution from the low‑rank increments
    for u_i, v_i in increments:
        # v_i @ X gives shape (p,), outer product with u_i gives (m,p)
        Y += np.outer(u_i, v_i @ X)
    return Y

def spectral_probe(U: np.ndarray,
                   s: np.ndarray,
                   Vt: np.ndarray,
                   increments: list[tuple[np.ndarray, np.ndarray]],
                   p: int | None = None,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Estimate singular values of the residual using randomised spectral probing.

    This routine performs a single randomised SVD on the matrix ``A`` represented
    by ``(U, s, Vt)`` and ``increments``.  It returns the singular values of the
    projection ``Q^T A``, where ``Q`` is an orthonormal basis for the range of
    ``A Ω`` with Ω a Gaussian random matrix.  The leading singular values of
    ``Q^T A`` approximate the leading singular values of ``A``.

    Parameters
    ----------
    U, s, Vt : see :func:`apply_A`
    increments : list of rank‑1 perturbations
    p : int, optional
        Target sketch dimension.  Defaults to ``len(s) + 2`` but is capped
        at ``n`` to avoid oversized sketches.
    rng : numpy.random.Generator, optional
        Random number generator to use.  A default generator is created if
        ``None``.

    Returns
    -------
    s_probe : ndarray
        Approximate singular values of the matrix ``A`` (in descending order).
        Only the first ``p`` values are returned.  These include the singular
        values already present in ``s`` and any residual spectral energy.
    """
    m, r = U.shape
    n = Vt.shape[1]
    if rng is None:
        rng = np.random.default_rng()
    if p is None:
        p = min(max(2, r + 2), n)

    # Draw a random Gaussian test matrix Ω of shape (n, p)
    Omega = rng.standard_normal(size=(n, p))
    # Compute Y = A Ω
    Y = apply_A(U, s, Vt, increments, Omega)  # shape (m, p)
    # Orthonormalise the columns of Y
    Q, _ = np.linalg.qr(Y, mode='reduced')  # shape (m, p)
    # Form B = Q^T A as a dense (p × n) matrix
    # Compute Q^T @ U diag(s) Vt efficiently
    QtU = Q.T @ U  # shape (p, r)
    B1 = (QtU * s) @ Vt  # shape (p, n)
    # Add the contributions from increments
    if increments:
        for u_i, v_i in increments:
            B1 += np.outer(Q.T @ u_i, v_i)
    # Compute SVD of the small matrix B1
    _, s_probe, _ = svd(B1, full_matrices=False)
    return s_probe