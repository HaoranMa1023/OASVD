"""Metric functions for evaluating streaming low‑rank approximations."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm


def relative_error(A: np.ndarray,
                   U: np.ndarray,
                   s: np.ndarray,
                   Vt: np.ndarray) -> float:
    """Compute the relative Frobenius error of a truncated SVD approximation.

    Parameters
    ----------
    A : ndarray of shape (m, n)
        Reference matrix.
    U : ndarray of shape (m, r)
        Left singular vectors.
    s : ndarray of shape (r,)
        Singular values.
    Vt : ndarray of shape (r, n)
        Right singular vectors (transposed).

    Returns
    -------
    rel_err : float
        Relative Frobenius norm of the error ``||A - U diag(s) Vt||_F / ||A||_F``.
    """
    approx = (U * s) @ Vt
    return norm(A - approx, 'fro') / max(1.0, norm(A, 'fro'))


def coverage(errors: list[float], threshold: float) -> float:
    """Compute the fraction of errors below a given threshold.

    Parameters
    ----------
    errors : list of float
        Sequence of relative errors over time.
    threshold : float
        Error threshold to test.

    Returns
    -------
    cov : float
        Fraction of time steps where ``errors[t] <= threshold``.
    """
    if not errors:
        return 0.0
    return sum(err <= threshold for err in errors) / len(errors)


def jitter(ranks: list[int]) -> int:
    """Compute the total rank jitter over a sequence of ranks.

    The jitter is defined as ``sum_t |r_t - r_{t-1}|``.  It measures how
    frequently and how violently the truncation rank oscillates.

    Parameters
    ----------
    ranks : list of int
        Sequence of truncation ranks over time.

    Returns
    -------
    total_jitter : int
        Sum of absolute rank changes between consecutive steps.
    """
    if len(ranks) < 2:
        return 0
    return int(sum(abs(ranks[i] - ranks[i - 1]) for i in range(1, len(ranks))))


def orth_error(U: np.ndarray) -> float:
    """Compute the orthogonality error ``||I - U^T U||_F``.

    Parameters
    ----------
    U : ndarray of shape (m, r)
        Left singular vectors.

    Returns
    -------
    gamma : float
        Frobenius norm of the deviation of ``U`` from orthonormality.
    """
    r = U.shape[1]
    return norm(np.eye(r) - U.T @ U, 'fro')