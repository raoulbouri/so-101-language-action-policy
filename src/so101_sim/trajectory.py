"""Quintic (5th-order) interpolation in joint space.

A quintic polynomial is the lowest order that lets us pin position, velocity
*and* acceleration at both endpoints. Setting the end velocities and
accelerations to zero gives a segment that starts and stops perfectly smoothly,
which is what keeps the recorded action stream free of the step discontinuities
a linear or cubic blend would leave at every phase boundary.
"""

from __future__ import annotations

import numpy as np


def quintic_scaling(tau: np.ndarray) -> np.ndarray:
    """Minimum-jerk time scaling s(tau) on tau in [0, 1].

    s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0.
    """
    tau = np.clip(tau, 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def quintic_segment(q_start: np.ndarray, q_end: np.ndarray, n_steps: int) -> np.ndarray:
    """Interpolate joint-space waypoints with a minimum-jerk profile.

    Returns an ``(n_steps, n_joints)`` array. The start configuration is *not*
    repeated, so concatenating consecutive segments produces a stream with no
    duplicated frames.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    n_steps = max(int(n_steps), 1)
    tau = np.linspace(0.0, 1.0, n_steps + 1)[1:]
    s = quintic_scaling(tau)[:, None]
    return q_start[None, :] + s * (q_end - q_start)[None, :]


def cartesian_quintic(p_start: np.ndarray, p_end: np.ndarray, n_steps: int) -> np.ndarray:
    """Same minimum-jerk scaling applied to a straight Cartesian line.

    Used for the phases where the *path shape* matters (a straight descent, a
    straight transit) rather than just the endpoints.
    """
    return quintic_segment(p_start, p_end, n_steps)


def steps_for(duration_s: float, control_hz: float) -> int:
    """Number of control steps covering `duration_s`, at least one."""
    return max(round(duration_s * control_hz), 1)
