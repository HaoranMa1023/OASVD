"""Incremental SVD updates for low-rank perturbations.

This module implements a simple rank-1 update to a truncated singular value
decomposition (SVD).  Given a matrix approximation

    A_t ≈ U @ S @ V.T,

and a rank-1 perturbation ``Δ = u_vec[:, None] * v_vec[None, :]``, this function
computes an updated SVD of ``A_t + Δ`` in the subspace spanned by the current
singular vectors and (optionally) one new orthogonal direction.

The update is performed entirely on matrices of size ``r × r`` or
``(r+1) × (r+1)``, so the cost is ``O(m r^2)`` when ``m >> r``.

This implementation follows the classical approach described in [Brand, 2006],
with minor adaptations for clarity.  It supports real valued matrices only.

Returns additionally a scalar

    e_perp = ||p|| * ||q||,

where p, q are the orthogonal components of the update vectors with respect to
the current subspaces; this can be used as an “incremental orthogonal energy”
indicator in higher–level control logic.

Example
-------

```python
import numpy as np
from core.incremental_svd import incremental_update

# initial truncated SVD of some matrix A
U, s, Vt = np.linalg.svd(A, full_matrices=False)
r0 = 5
U = U[:, :r0]
s = s[:r0]
Vt = Vt[:r0, :]

# rank-1 update vector outer product
u_vec = np.random.randn(A.shape[0])
v_vec = np.random.randn(A.shape[1])

U_new, s_new, Vt_new, e_perp = incremental_update(U, s, Vt, u_vec, v_vec)
```
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm, svd

def incremental_update(U: np.ndarray,
                       s: np.ndarray,
                       Vt: np.ndarray,
                       u_vec: np.ndarray,
                       v_vec: np.ndarray,
                       tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Perform a rank‑1 update to a truncated SVD.

    Parameters
    ----------
    U : ndarray of shape (m, r)
        Left singular vectors of the current approximation.
    s : ndarray of shape (r,)
        Singular values of the current approximation.
    Vt : ndarray of shape (r, n)
        Right singular vectors (transposed) of the current approximation.
    u_vec : ndarray of shape (m,)
        Left vector of the rank-1 perturbation ``Δ = u_vec[:,None] * v_vec[None,:]``.
    v_vec : ndarray of shape (n,)
        Right vector of the rank-1 perturbation ``Δ = u_vec[:,None] * v_vec[None,:]``.
    tol : float, optional
        Tolerance below which an orthogonal component is treated as numerically zero.

    Returns
    -------
    U_new : ndarray of shape (m, r')   (r' is r or r+1)
        Updated left singular vectors.
    s_new : ndarray of shape (r',)
        Updated singular values (in descending order).
    Vt_new : ndarray of shape (r', n)
        Updated right singular vectors (transposed).
    e_perp : float
        Incremental orthogonal energy, defined as ``||p|| * ||q||``, where
        p and q are the components of u_vec and v_vec orthogonal to the
        current subspaces spanned by U and V.

    Notes
    -----
    This routine does *not* enforce a truncation to a prescribed target rank;
    it may return either r or r+1 singular directions depending on whether
    a new orthogonal direction is needed.  Truncation and rank control should
    be handled by the caller (e.g. via hysteresis rules in OASVD).
    """
    m, r = U.shape
    n = Vt.shape[1]

    # Project the update vectors onto the current subspaces
    alpha = U.T @ u_vec  # shape (r,)
    beta = Vt @ v_vec    # shape (r,)

    # Compute the orthogonal components
    p = u_vec - U @ alpha
    q = v_vec - Vt.T @ beta
    norm_p = norm(p)
    norm_q = norm(q)

    # Incremental orthogonal energy indicator
    e_perp = float(norm_p * norm_q)

    # Case 1: update lies (numerically) in the current subspace
    # --------------------------------------------------------
    # If either component is nearly zero, we treat the update as living
    # entirely in the span of U, V and only update the r×r core matrix.
    if norm_p < tol or norm_q < tol:
        # Core r×r matrix M representing A + Δ in the current basis:
        # A ≈ U diag(s) V^T,  Δ ≈ U alpha beta^T V^T
        # => A + Δ ≈ U (diag(s) + alpha beta^T) V^T
        M = np.diag(s) + np.outer(alpha, beta)  # shape (r, r)

        U_m, s_m, V_m = svd(M, full_matrices=False)

        # Rotate the existing bases
        U_new = U @ U_m          # (m, r)
        Vt_new = V_m @ Vt        # (r, n)

        return U_new, s_m, Vt_new, e_perp

    # Case 2: genuinely new orthogonal directions are present
    # -------------------------------------------------------
    # Form the augmented orthonormal directions
    P = p / norm_p  # shape (m,)
    Q = q / norm_q  # shape (n,)

    # Assemble the (r+1)×(r+1) small matrix K
    K = np.zeros((r + 1, r + 1), dtype=U.dtype)
    K[:r, :r] = np.diag(s)

    # w and z collect projection coefficients plus the orthogonal norms
    w = np.concatenate((alpha, [norm_p]))
    z = np.concatenate((beta, [norm_q]))
    K += np.outer(w, z)

    # SVD of K gives the updated singular values and rotation coefficients
    U_k, s_k, V_k = svd(K, full_matrices=False)

    # Build augmented U and Vt with the new orthonormal directions
    U_aug = np.hstack((U, P[:, None]))      # shape (m, r+1)
    Vt_aug = np.vstack((Vt, Q[None, :]))    # shape (r+1, n)

    # Form the updated U and Vt by applying the rotations
    U_new = U_aug @ U_k                     # shape (m, r+1)
    Vt_new = V_k @ Vt_aug                   # shape (r+1, n)

    return U_new, s_k, Vt_new, e_perp
