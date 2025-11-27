"""OASVD control loop and rank adaptation.

This module defines the :class:`OASVD` class, which encapsulates the full online
adaptive SVD algorithm.  The algorithm maintains an approximate SVD of a
streaming matrix and adapts the truncation rank at each step based on
observations of spectral residuals, error estimates and orthogonality metrics.

The implementation here is an intentionally simplified version of the framework
described in our paper.  It provides all the key components—incremental
updates, optional spectral probing, hysteresis for rank decisions, error
feedback and re-orthogonalisation—while keeping the code accessible.  For
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
        When exceeded, a QR re-orthogonalisation is triggered.
    probe_dim : int, optional
        Target dimension for the randomised residual spectral probe.  If
        ``None``, a default of ``r+2`` is used.
    buffer_size : int, optional
        Maximum number of recent rank-1 increments kept in the low-rank
        buffer used by the spectral probe.

    Ablation flags
    --------------
    enable_probing : bool, optional
        If ``False``, disable residual spectral probing entirely
        (No-probing variant).
    enable_hysteresis : bool, optional
        If ``False``, use instantaneous threshold decisions without
        streak counters (No-hysteresis variant).
    enable_feedback : bool, optional
        If ``False``, do not use the error estimate ``eps_hat`` for
        rank decisions or probe triggering (No-feedback variant).
    enable_reorth : bool, optional
        If ``False``, skip QR re-orthogonalisation (No-reorth variant).

    Notes
    -----
    This class stores only the factors ``(U, s, Vt)`` of the current truncated
    SVD.  It does not store the full matrix stream.  Low-rank update vectors
    are stored in a short buffer for use in residual spectral probing.
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
                 buffer_size: int = 10,
                 *,
                 enable_probing: bool = True,
                 enable_hysteresis: bool = True,
                 enable_feedback: bool = True,
                 enable_reorth: bool = True) -> None:
        # core thresholds / params
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

        # ablation flags
        self.enable_probing = bool(enable_probing)
        self.enable_hysteresis = bool(enable_hysteresis)
        self.enable_feedback = bool(enable_feedback)
        self.enable_reorth = bool(enable_reorth)

        # Initial SVD of A0
        U0, s0, Vt0 = np.linalg.svd(A0, full_matrices=False)
        U0 = U0[:, :r0].copy()
        s0 = s0[:r0].copy()
        Vt0 = Vt0[:r0, :].copy()
        self.U = U0
        self.s = s0
        self.Vt = Vt0
        self.r = r0

        # Initial error estimate (only meaningful if feedback is enabled)
        residual = A0 - (self.U * self.s) @ self.Vt
        self.eps_hat = norm(residual, 'fro') if self.enable_feedback else 0.0

        # Orthogonality error
        self.gamma = norm(np.eye(self.r) - self.U.T @ self.U, 'fro')

        # Hysteresis counters
        self.t_up_streak = 0
        self.t_down_streak = 0

        # Low-rank update buffer for residual probing
        self.increments: list[tuple[np.ndarray, np.ndarray]] = []

        # Step counter
        self.step = 0

    def _need_probe(self, e_perp: float) -> bool:
        """Decide whether to trigger an extra spectral probe.

        In the full OASVD configuration this uses both the orthogonal
        incremental energy and the current error estimate.  For the
        No-feedback variant, only the local orthogonal energy is used.
        If probing is disabled entirely, this is never called.
        """
        # If probing is globally disabled, we should not request extra probes
        if not self.enable_probing:
            return False

        if e_perp > self.theta_up:
            return True

        if self.enable_feedback and self.eps_hat > 0.8 * self.eps_max:
            return True

        return False

    def _update_buffers(self, u_vec: np.ndarray, v_vec: np.ndarray) -> None:
        """Append a rank-1 update to the buffer, keeping only the latest entries."""
        self.increments.append((u_vec, v_vec))
        if len(self.increments) > self.buffer_size:
            self.increments.pop(0)

    def update(self, u_vec: np.ndarray, v_vec: np.ndarray) -> dict[str, np.ndarray | float]:
        """Process a rank-1 update and adapt the truncation rank.

        Parameters
        ----------
        u_vec : ndarray of shape (m,)
            Left vector of the rank-1 increment.
        v_vec : ndarray of shape (n,)
            Right vector of the rank-1 increment.

        Returns
        -------
        state : dict
            Dictionary containing the updated factors and diagnostics:
            ``U``, ``s``, ``Vt``, ``r``, ``eps_hat`` and ``gamma``.
        """
        # 1. Incremental SVD update to form a candidate SVD
        U_cand, s_cand, Vt_cand, e_perp = incremental_update(
            self.U, self.s, self.Vt, u_vec, v_vec
        )
        r_cand = len(s_cand)

        # 2. Update low-rank buffer for residual spectral probing
        self._update_buffers(u_vec, v_vec)

        # 3. Spectral probing (optional) to sense residual spectral energy
        perform_probe = False
        if self.enable_probing:
            if self.h > 0 and (self.step % self.h == 0):
                perform_probe = True
            if self._need_probe(e_perp):
                perform_probe = True

        s_probe = np.array([])
        if perform_probe:
            s_probe = spectral_probe(
                U_cand, s_cand, Vt_cand, self.increments,
                p=self.probe_dim
            )

        # Combine candidate singular values with probed residual singular values
        if s_probe.size > 0:
            sigma_combined = np.sort(np.concatenate((s_cand, s_probe)))[::-1]
        else:
            sigma_combined = s_cand.copy()

        # 4. Rank decision
        r = self.r
        sigma_next = sigma_combined[r] if r < len(sigma_combined) else 0.0
        sigma_r = sigma_combined[r - 1] if r - 1 < len(sigma_combined) else 0.0

        if self.enable_hysteresis:
            # Hysteresis-based decision (default / full OASVD)
            if sigma_next >= self.theta_up and r < self.r_max:
                self.t_up_streak += 1
            else:
                self.t_up_streak = 0

            if sigma_r <= self.theta_down and r > self.r_min:
                self.t_down_streak += 1
            else:
                self.t_down_streak = 0

            r_new = r
            if self.t_up_streak >= self.tau_hold and r < self.r_max:
                r_new = r + 1
                self.t_up_streak = 0
            elif self.t_down_streak >= self.tau_hold and r > self.r_min:
                r_new = r - 1
                self.t_down_streak = 0
        else:
            # No-hysteresis: instantaneous decisions based only on current
            # sigma_next / sigma_r without streak counters.
            r_new = r
            if sigma_next >= self.theta_up and r < self.r_max:
                r_new = r + 1
            elif sigma_r <= self.theta_down and r > self.r_min:
                r_new = r - 1
            # keep counters for logging but they are not used
            self.t_up_streak = 0
            self.t_down_streak = 0

        # 5. Truncate the candidate SVD to r_new
        r_trunc = min(r_new, len(s_cand))
        U_new = U_cand[:, :r_trunc]
        s_new = s_cand[:r_trunc].copy()
        Vt_new = Vt_cand[:r_trunc, :]

        # 6. Error estimate update (only in feedback-enabled variants)
        if self.enable_feedback:
            tail = s_cand[r_trunc:]
            if tail.size > 0:
                # C_tail ≈ 1: discarded singular values give an upper bound
                self.eps_hat = np.sqrt(self.eps_hat ** 2 + np.sum(tail ** 2))

            # 7. If error exceeds threshold, conservatively increase rank
            if self.eps_hat > self.eps_max and r_trunc < len(s_cand) and r_trunc < self.r_max:
                r_trunc2 = min(r_trunc + 1, len(s_cand), self.r_max)
                U_new = U_cand[:, :r_trunc2]
                s_new = s_cand[:r_trunc2].copy()
                Vt_new = Vt_cand[:r_trunc2, :]
                tail2 = s_cand[r_trunc2:]
                if tail2.size > 0:
                    self.eps_hat = np.sqrt(self.eps_hat ** 2 + np.sum(tail2 ** 2))
                r_trunc = r_trunc2
        else:
            # No-feedback: keep eps_hat at a neutral value (for logging only).
            # We do not use it in rank decisions or probe triggering.
            self.eps_hat = 0.0

        # 8. Orthogonality check and re-orthogonalisation
        gamma = norm(np.eye(r_trunc) - U_new.T @ U_new, 'fro')
        if self.enable_reorth and gamma > self.gamma_max:
            Q, _ = qr(U_new, mode='reduced')
            U_new = Q
            gamma = norm(np.eye(r_trunc) - U_new.T @ U_new, 'fro')

        # Update internal state
        self.U = U_new
        self.s = s_new
        self.Vt = Vt_new
        self.r = r_trunc
        self.gamma = gamma

        self.step += 1

        return {
            "U": self.U,
            "s": self.s,
            "Vt": self.Vt,
            "r": self.r,
            "eps_hat": self.eps_hat,
            "gamma": self.gamma,
            "e_perp": float(e_perp),
        }
