"""Two-body orbit propagation with universal variables.

The universal-variable formulation (Stumpff functions C(z), S(z)) propagates
elliptic, parabolic, and hyperbolic orbits with one algorithm and no special
cases — the standard formulation from Bate, Mueller & White.

Units are km, s, and km^3/s^2 throughout the package.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MU_SUN", "MU_EARTH", "stumpff_c", "stumpff_s", "propagate"]

MU_SUN: float = 1.32712440018e11  # km^3/s^2
MU_EARTH: float = 3.986004418e5  # km^3/s^2


def stumpff_c(z: float) -> float:
    """Stumpff function C(z) = (1 - cos sqrt(z)) / z, analytic at z = 0."""
    if z > 1e-8:
        return (1.0 - np.cos(np.sqrt(z))) / z
    if z < -1e-8:
        return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    return 1.0 / 2.0 - z / 24.0 + z * z / 720.0


def stumpff_s(z: float) -> float:
    """Stumpff function S(z) = (sqrt(z) - sin sqrt(z)) / z^{3/2}, analytic at 0."""
    if z > 1e-8:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / sz**3
    if z < -1e-8:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / sz**3
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def propagate(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    mu: float = MU_SUN,
    *,
    tol: float = 1e-10,
    max_iter: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate state (r0, v0) by time dt under two-body gravity.

    Returns (r, v) at t0 + dt. Newton iteration on the universal anomaly chi
    with the Lagrange f and g functions; works for any conic.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    r0n = float(np.linalg.norm(r0))
    v0n = float(np.linalg.norm(v0))
    vr0 = float(np.dot(r0, v0)) / r0n
    alpha = 2.0 / r0n - v0n**2 / mu  # 1/a; > 0 ellipse, < 0 hyperbola

    sqrt_mu = np.sqrt(mu)
    chi = sqrt_mu * abs(alpha) * dt if abs(alpha) > 1e-12 else sqrt_mu * dt / r0n
    for _ in range(max_iter):
        z = alpha * chi**2
        c, s = stumpff_c(z), stumpff_s(z)
        f_val = (
            r0n * vr0 / sqrt_mu * chi**2 * c
            + (1.0 - alpha * r0n) * chi**3 * s
            + r0n * chi
            - sqrt_mu * dt
        )
        f_der = (
            r0n * vr0 / sqrt_mu * chi * (1.0 - alpha * chi**2 * s)
            + (1.0 - alpha * r0n) * chi**2 * c
            + r0n
        )
        step = f_val / f_der
        chi -= step
        if abs(step) < tol:
            break
    else:
        raise RuntimeError(f"universal Kepler iteration did not converge (dt={dt})")

    z = alpha * chi**2
    c, s = stumpff_c(z), stumpff_s(z)
    f = 1.0 - chi**2 / r0n * c
    g = dt - chi**3 / sqrt_mu * s
    r = f * r0 + g * v0
    rn = float(np.linalg.norm(r))
    f_dot = sqrt_mu / (rn * r0n) * chi * (z * s - 1.0)
    g_dot = 1.0 - chi**2 / rn * c
    v = f_dot * r0 + g_dot * v0
    return r, v
