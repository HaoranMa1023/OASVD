"""Metric functions for evaluating streaming low-rank approximations."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from typing import Sequence, Mapping


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


def coverage(errors: Sequence[float], threshold: float) -> float:
    """Compute the fraction of errors below a given threshold.

    Parameters
    ----------
    errors : sequence of float
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


def multi_coverage(errors: Sequence[float],
                   thresholds: Sequence[float]) -> dict[float, float]:
    """Compute error coverage for a list of thresholds.

    This is a convenience wrapper used when reporting
    ``Cov_eps`` for several epsilon values, e.g.
    ``eps in {1e-2, 5e-3}``.

    Parameters
    ----------
    errors : sequence of float
        Relative errors over time.
    thresholds : sequence of float
        Threshold values.

    Returns
    -------
    cov_dict : dict
        Mapping ``eps -> Cov_eps``.
    """
    return {eps: coverage(errors, eps) for eps in thresholds}


def jitter(ranks: Sequence[int]) -> int:
    """Compute the total rank jitter over a sequence of ranks.

    The jitter is defined as ``sum_t |r_t - r_{t-1}|``.  It measures how
    frequently and how violently the truncation rank oscillates.

    Parameters
    ----------
    ranks : sequence of int
        Sequence of truncation ranks over time.

    Returns
    -------
    total_jitter : int
        Sum of absolute rank changes between consecutive steps.
    """
    if len(ranks) < 2:
        return 0
    return int(sum(abs(ranks[i] - ranks[i - 1]) for i in range(1, len(ranks))))


def mean_rank(ranks: Sequence[int]) -> float:
    """Compute the average truncation rank over time.

    This corresponds to ``\\bar r = (1/T) sum_t r_t`` used in the tables.

    Parameters
    ----------
    ranks : sequence of int
        Sequence of truncation ranks over time.

    Returns
    -------
    r_bar : float
        Average rank, or 0.0 if the sequence is empty.
    """
    if not ranks:
        return 0.0
    return float(sum(ranks) / len(ranks))


def max_error(errors: Sequence[float]) -> float:
    """Return the maximum relative error over time.

    Parameters
    ----------
    errors : sequence of float
        Relative errors over time.

    Returns
    -------
    max_err : float
        Maximum value of ``errors``, or 0.0 if empty.
    """
    if not errors:
        return 0.0
    return float(max(errors))


def mean_error(errors: Sequence[float]) -> float:
    """Return the average relative error over time.

    Parameters
    ----------
    errors : sequence of float
        Relative errors over time.

    Returns
    -------
    avg_err : float
        Average of ``errors``, or 0.0 if empty.
    """
    if not errors:
        return 0.0
    return float(sum(errors) / len(errors))


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


# --- Change-point / detection metrics for Experiment 5.2 ---------------------


def detection_delays(true_changes: Sequence[int],
                     detected_changes: Sequence[int],
                     T: int | None = None) -> list[int]:
    """Compute detection delays for a set of true change-points.

    For each true change time ``t*``, we look for the smallest detected time
    ``t_det >= t*``.  The delay is ``t_det - t*``; if no such detection
    exists, we optionally treat the delay as ``T - t*`` (if ``T`` is given),
    or ignore it if ``T`` is None.

    Parameters
    ----------
    true_changes : sequence of int
        Ground-truth change times (e.g. shock times t*).
    detected_changes : sequence of int
        Times at which the algorithm raises a change / probe alarm.
    T : int, optional
        Total length of the time series.  If provided, missing detections
        contribute a delay of ``T - t*``; if None, missing detections are
        skipped.

    Returns
    -------
    delays : list of int
        Detection delays for each true change (in the same order).
    """
    det = sorted(detected_changes)
    delays: list[int] = []
    for t_star in true_changes:
        # first detected time >= t_star
        t_det = next((t for t in det if t >= t_star), None)
        if t_det is None:
            if T is not None:
                delays.append(int(T - t_star))
            # if T is None, we simply skip this change
        else:
            delays.append(int(t_det - t_star))
    return delays


def mean_detection_delay(true_changes: Sequence[int],
                         detected_changes: Sequence[int],
                         T: int | None = None) -> float:
    """Compute the average detection delay over all true change-points."""
    delays = detection_delays(true_changes, detected_changes, T)
    if not delays:
        return 0.0
    return float(sum(delays) / len(delays))


def false_alarm_count(true_changes: Sequence[int],
                      detected_changes: Sequence[int],
                      tolerance: int = 0) -> int:
    """Count the number of false alarms among detected change times.

    A detected time is considered a true positive if it lies within
    ``[t* - tolerance, t* + tolerance]`` of any true change ``t*``.
    All other detected times are counted as false alarms.

    Parameters
    ----------
    true_changes : sequence of int
        Ground-truth change times.
    detected_changes : sequence of int
        Times at which the algorithm raises a change / probe alarm.
    tolerance : int, optional
        Symmetric window around each true change within which detections
        are considered correct.  Default is 0 (exact match).

    Returns
    -------
    n_false : int
        Number of detected times that are not matched to any true change.
    """
    if not detected_changes:
        return 0
    true_changes = list(true_changes)
    n_false = 0
    for t in detected_changes:
        ok = any(abs(t - t_star) <= tolerance for t_star in true_changes)
        if not ok:
            n_false += 1
    return int(n_false)
