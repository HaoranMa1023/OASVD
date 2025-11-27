"""Incremental SVD updates for low‑rank perturbations.

This module implements a simple rank‑1 update to a truncated singular value
decomposition (SVD).  Given a matrix approximation

    A_t ≈ U @ S @ V.T,

and a rank‑1 perturbation ``Δ = u_vec[:, None] * v_vec[None, :]``, this function
computes an updated truncated SVD of ``A_t + Δ``.  The update is performed
entirely on matrices of size ``(r+1) × (r+1)``, so the cost is
``O(m r^2)`` when ``m >> r``.

This implementation follows the classical approach described in [Brand, 2006],
with minor adaptations for clarity.  It supports real valued matrices only.

Example
-------

```python
import numpy as np
from oasvd.incremental_svd import incremental_update

# initial truncated SVD of some matrix A
U, s, Vt = np.linalg.svd(A, full_matrices=False)
r0 = 5
U = U[:, :r0]
s = s[:r0]
Vt = Vt[:r0, :]

# rank‑1 update vector outer product
u_vec = np.random.randn(A.shape[0])
v_vec = np.random.randn(A.shape[1])

U_new, s_new, Vt_new = incremental_update(U, s, Vt, u_vec, v_vec)
```
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm, svd

def incremental_update(U: np.ndarray,
                       s: np.ndarray,
                       Vt: np.ndarray,
                       u_vec: np.ndarray,
                       v_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Left vector of the rank‑1 perturbation ``Δ = u_vec[:,None] * v_vec[None,:]``.
    v_vec : ndarray of shape (n,)
        Right vector of the rank‑1 perturbation ``Δ = u_vec[:,None] * v_vec[None,:]``.

    Returns
    -------
    U_new : ndarray of shape (m, r+1)
        Updated left singular vectors.
    s_new : ndarray of shape (r+1,)
        Updated singular values (in descending order).
    Vt_new : ndarray of shape (r+1, n)
        Updated right singular vectors (transposed).

    Notes
    -----
    The update does not enforce a truncation beyond ``r+1``.  Truncation
    should be performed by the caller if a fixed rank is desired.
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

    # If either component is nearly zero, the update lies entirely in the
    # current subspace; in that case we only need to update the small matrix
    if norm_p < 1e-12 or norm_q < 1e-12:
        # Expand p and q by adding a zero row/column to the small matrix
        K = np.zeros((r+1, r+1), dtype=U.dtype)
        K[:r, :r] = np.diag(s)
        # w and z collect the projection coefficients
        w = np.concatenate((alpha, [0.0]))
        z = np.concatenate((beta, [0.0]))
        K += np.outer(w, z)
        # SVD of the small K
        U_k, s_k, V_k = svd(K, full_matrices=False)
        # Update U and Vt
        # Append zero vectors to match dimensions
        U_aug = np.hstack((U, np.zeros((m, 1), dtype=U.dtype)))
        Vt_aug = np.vstack((Vt, np.zeros((1, n), dtype=Vt.dtype)))
        U_new = U_aug @ U_k
        Vt_new = V_k @ Vt_aug
        return U_new, s_k, Vt_new

    # Otherwise, form the augmented basis vectors
    P = p / norm_p  # shape (m,)
    Q = q / norm_q  # shape (n,)

    # Assemble the (r+1)x(r+1) small matrix K
    K = np.zeros((r+1, r+1), dtype=U.dtype)
    K[:r, :r] = np.diag(s)
    w = np.concatenate((alpha, [norm_p]))
    z = np.concatenate((beta, [norm_q]))
    K += np.outer(w, z)

    # SVD of K gives the updated singular values and rotation coefficients
    U_k, s_k, V_k = svd(K, full_matrices=False)

    # Build augmented U and Vt with the new orthonormal directions
    U_aug = np.hstack((U, P[:, None]))  # shape (m, r+1)
    Vt_aug = np.vstack((Vt, Q[None, :]))  # shape (r+1, n)

    # Form the updated U and Vt by applying the rotations
    U_new = U_aug @ U_k
    Vt_new = V_k @ Vt_aug

    return U_new, s_k, Vt_new