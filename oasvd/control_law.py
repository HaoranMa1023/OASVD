"""OASVD control loop and rank adaptation.

This module defines the :class:`OASVD` class, which encapsulates the full online
adaptive SVD algorithm.  The algorithm maintains an approximate SVD of a
streaming matrix and adapts the truncation rank at each step based on
observations of spectral residuals, error estimates and orthogonality metrics.

The implementation here is an intentionally simplified version of the framework
described in our paper.  It provides all of the key components—incremental
updates, optional spectral probing, hysteresis for rank decisions, error
feedback and re‑orthogonalisation—while keeping the code accessible.  For
detailed experimental setups and parameter choices, refer to the notebooks.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm, qr

from .incremental_svd import incremental_update
from .spectral_probe import spectral_probe


class OASVD:
    """Online Adaptive SVD (OASVD) implementation.

    Parameters
    ----------
    A0 : ndarray of shape (m, n)
        Initial matrix (or a batch of snapshots) used to initialise the
        truncated SVD.  Only the leading ``r0`` singular values/vectors are
        retained.
    r0 : int
        Initial truncation rank.
    eps_max : float
        Target upper bound on the approximation error (Frobenius norm).  The
        algorithm adapts the rank so that the estimated error ``eps_hat``
        remains near or below this threshold.
    theta_up : float
        Upper spectral threshold; if the estimated ``sigma_(r+1)`` of the
        approximation exceeds this threshold for ``tau_hold`` consecutive
        steps, the rank is increased.
    theta_down : float
        Lower spectral threshold; if the estimated ``sigma_r`` falls below
        this threshold for ``tau_hold`` consecutive steps, the rank is
        decreased.
    r_min, r_max : int
        Minimum and maximum allowable truncation ranks.
    tau_hold : int
        Hysteresis hold time; the spectral condition must persist for this
        many consecutive steps before a rank change occurs.
    h : int
        Basic spectral probing period.  Spectral probing is performed every
        ``h`` steps to estimate residual singular values.
    gamma_max : float
        Tolerance on the orthogonality error ``gamma = ||I - U^T U||_F``.
        When exceeded, a QR re‑orthogonalisation is triggered.
    probe_dim : int, optional
        Target dimension for the randomised residual spectral probe.  If
        ``None``, a default of ``r+2`` is used.

    Notes
    -----
    This class stores only the factors ``(U, s, Vt)`` of the current truncated
    SVD.  It does not store the full matrix stream.  Low‑rank update vectors
    are stored in a short buffer for use in spectral probing.
    """

    def __init__(self,
                 A0: np.ndarray,
                 r0: int,
                 eps_max: float,
                 theta_up: float,
                 theta_down: float,
                 r_min: int,
                 r_max: int,
                 tau_hold: int,
                 h: int,
                 gamma_max: float,
                 probe_dim: int | None = None,
                 buffer_size: int = 10) -> None:
        self.eps_max = float(eps_max)
        self.theta_up = float(theta_up)
        self.theta_down = float(theta_down)
        self.r_min = int(r_min)
        self.r_max = int(r_max)
        self.tau_hold = int(tau_hold)
        self.h = int(h)
        self.gamma_max = float(gamma_max)
        self.probe_dim = probe_dim
        self.buffer_size = buffer_size

        # Initial SVD of A0
        U0, s0, Vt0 = np.linalg.svd(A0, full_matrices=False)
        # Retain the first r0 singular triplets
        U0 = U0[:, :r0].copy()
        s0 = s0[:r0].copy()
        Vt0 = Vt0[:r0, :].copy()
        self.U = U0
        self.s = s0
        self.Vt = Vt0
        self.r = r0

        # Estimate initial error bound as residual Frobenius norm
        residual = A0 - (self.U * self.s) @ self.Vt
        self.eps_hat = norm(residual, 'fro')

        # Orthogonality error
        self.gamma = norm(np.eye(self.r) - self.U.T @ self.U, 'fro')

        # Hysteresis counters
        self.t_up_streak = 0
        self.t_down_streak = 0

        # Low‑rank update buffer for residual probing
        self.increments: list[tuple[np.ndarray, np.ndarray]] = []

        # Step counter
        self.step = 0

    def _update_buffers(self, u_vec: np.ndarray, v_vec: np.ndarray) -> None:
        """Append a rank‑1 update to the buffer, keeping only the latest entries."""
        self.increments.append((u_vec, v_vec))
        # Maintain a fixed buffer size to limit memory and cost
        if len(self.increments) > self.buffer_size:
            self.increments.pop(0)

    def update(self, u_vec: np.ndarray, v_vec: np.ndarray) -> dict[str, np.ndarray | float]:
        """Process a rank‑1 update and adapt the truncation rank.

        Parameters
        ----------
        u_vec : ndarray of shape (m,)
            Left vector of the rank‑1 increment.
        v_vec : ndarray of shape (n,)
            Right vector of the rank‑1 increment.

        Returns
        -------
        state : dict
            Dictionary containing the updated factors and diagnostics:
            ``U``, ``s``, ``Vt``, ``r``, ``eps_hat`` and ``gamma``.
        """
        m = self.U.shape[0]
        # Update buffer for spectral probing
        self._update_buffers(u_vec, v_vec)

        # 1. Incremental SVD update to form a candidate SVD
        U_cand, s_cand, Vt_cand = incremental_update(self.U, self.s, self.Vt, u_vec, v_vec)

        r_cand = len(s_cand)

        # 2. Spectral probing (optional) to sense residual spectral energy
        perform_probe = False
        if self.h > 0 and (self.step % self.h == 0):
            perform_probe = True
        # Additional probing triggers could be added here, e.g. based on error
        # or the energy of the orthogonal component p,q from the update.  For
        # simplicity we probe periodically only.
        s_probe = np.array([])
        if perform_probe:
            s_probe = spectral_probe(
                self.U, self.s, self.Vt, self.increments,
                p=self.probe_dim
            )

        # Combine candidate singular values with probed residual singular values
        # to estimate sigma_(r) and sigma_(r+1).  We simply concatenate and sort.
        if s_probe.size > 0:
            sigma_combined = np.sort(np.concatenate((s_cand, s_probe)))[::-1]
        else:
            sigma_combined = s_cand.copy()

        # 3. Rank decision via hysteresis
        r = self.r
        # Next candidate singular value beyond current rank
        sigma_next = sigma_combined[r] if r < len(sigma_combined) else 0.0
        # Current smallest retained singular value
        sigma_r = sigma_combined[r - 1] if r - 1 < len(sigma_combined) else 0.0

        # Update streak counters
        if sigma_next >= self.theta_up and r < self.r_max:
            self.t_up_streak += 1
        else:
            self.t_up_streak = 0
        if sigma_r <= self.theta_down and r > self.r_min:
            self.t_down_streak += 1
        else:
            self.t_down_streak = 0

        # Decide new rank
        r_new = r
        if self.t_up_streak >= self.tau_hold and r < self.r_max:
            r_new = r + 1
            self.t_up_streak = 0
        elif self.t_down_streak >= self.tau_hold and r > self.r_min:
            r_new = r - 1
            self.t_down_streak = 0

        # 4. Truncate the candidate SVD to r_new
        # Ensure we do not exceed the available singular values
        r_trunc = min(r_new, len(s_cand))
        U_new = U_cand[:, :r_trunc]
        s_new = s_cand[:r_trunc].copy()
        Vt_new = Vt_cand[:r_trunc, :]

        # 5. Update error estimate: add the energy of discarded singular values
        tail = s_cand[r_trunc:]
        if tail.size > 0:
            self.eps_hat = np.sqrt(self.eps_hat ** 2 + np.sum(tail ** 2))

        # 6. If error exceeds threshold, conservatively increase rank
        if self.eps_hat > self.eps_max and r_trunc < len(s_cand) and r_trunc < self.r_max:
            # Add one more singular value if available
            r_trunc2 = min(r_trunc + 1, len(s_cand), self.r_max)
            U_new = U_cand[:, :r_trunc2]
            s_new = s_cand[:r_trunc2].copy()
            Vt_new = Vt_cand[:r_trunc2, :]
            # Update eps_hat with additional tail energy
            tail2 = s_cand[r_trunc2:]
            if tail2.size > 0:
                self.eps_hat = np.sqrt(self.eps_hat ** 2 + np.sum(tail2 ** 2))
            r_trunc = r_trunc2

        # 7. Orthogonality check and re‑orthogonalisation
        gamma = norm(np.eye(r_trunc) - U_new.T @ U_new, 'fro')
        if gamma > self.gamma_max:
            # QR factorisation to restore orthogonality; keep right factors intact
            Q, _ = qr(U_new, mode='reduced')
            U_new = Q
            gamma = norm(np.eye(r_trunc) - U_new.T @ U_new, 'fro')

        # Update internal state
        self.U = U_new
        self.s = s_new
        self.Vt = Vt_new
        self.r = r_trunc
        self.gamma = gamma

        # Advance step counter
        self.step += 1

        # Return diagnostics for logging
        return {
            'U': self.U,
            's': self.s,
            'Vt': self.Vt,
            'r': self.r,
            'eps_hat': self.eps_hat,
            'gamma': self.gamma,
        }