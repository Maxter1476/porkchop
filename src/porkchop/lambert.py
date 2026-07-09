"""Universal-variable Lambert solver (Bate-Mueller-White / Vallado form).

Given two position vectors and a time of flight, find the connecting conic's
terminal velocities. Zero-revolution transfers only — exactly what a porkchop
scan needs. The iteration is Newton on the universal parameter z with a
bisection safeguard, which converges for both short-way and long-way
transfers across the full elliptic/hyperbolic range.
"""

from __future__ import annotations

import numpy as np

from .kepler import MU_SUN, stumpff_c, stumpff_s

__all__ = ["lambert"]


def lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float = MU_SUN,
    *,
    prograde: bool = True,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Lambert's problem for the transfer from r1 to r2 in time tof.

    Parameters
    ----------
    r1, r2:
        Heliocentric (or geocentric) position vectors, km.
    tof:
        Time of flight in seconds; must be positive.
    mu:
        Gravitational parameter, km^3/s^2.
    prograde:
        Choose the transfer whose angular momentum has positive Z component
        (standard for ecliptic-plane planetary transfers).

    Returns
    -------
    (v1, v2):
        Velocity at departure and arrival, km/s.
    """
    if tof <= 0:
        raise ValueError("time of flight must be positive")
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    r1n = float(np.linalg.norm(r1))
    r2n = float(np.linalg.norm(r2))

    cos_dnu = float(np.dot(r1, r2)) / (r1n * r2n)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
    if 1.0 - cos_dnu < 1e-12:
        raise ValueError("transfer angle of 0 or pi: plane is undefined")
    cross = np.cross(r1, r2)
    # Transfer angle direction: prograde transfers sweep counterclockwise as
    # seen from +Z (cross_z > 0 => dnu < pi for the short way).
    if prograde:
        dnu = np.arccos(cos_dnu) if cross[2] >= 0.0 else 2.0 * np.pi - np.arccos(cos_dnu)
    else:
        dnu = 2.0 * np.pi - np.arccos(cos_dnu) if cross[2] >= 0.0 else np.arccos(cos_dnu)

    a_param = np.sin(dnu) * np.sqrt(r1n * r2n / (1.0 - cos_dnu))  # Vallado's A
    if abs(a_param) < 1e-12:
        raise ValueError("transfer angle of 0 or pi: plane is undefined")

    def tof_of_z(z: float) -> tuple[float, float, float]:
        c, s = stumpff_c(z), stumpff_s(z)
        y = r1n + r2n + a_param * (z * s - 1.0) / np.sqrt(c)
        if y < 0.0:
            return np.nan, y, c
        chi = np.sqrt(y / c)
        t = (chi**3 * s + a_param * np.sqrt(y)) / np.sqrt(mu)
        return t, y, c

    # Bracket the root in z. z < (2 pi)^2 for zero-rev transfers.
    z_low, z_high = -4.0 * np.pi**2, 4.0 * np.pi**2 - 1e-10
    # Ensure y(z_low) > 0 (long hyperbolic limit can drive y negative).
    for _ in range(200):
        t, y, _ = tof_of_z(z_low)
        if np.isfinite(t) and y > 0.0:
            break
        z_low = (z_low + z_high) / 2.0 if z_low < 0 else z_low
        z_low += 0.5
    z = 0.0
    for _ in range(max_iter):
        t, y, c = tof_of_z(z)
        if not np.isfinite(t) or y < 0.0:
            z = (z + z_high) / 2.0
            continue
        if abs(t - tof) < tol * max(1.0, tof):
            break
        if t < tof:
            z_low = z
        else:
            z_high = z
        # Newton step from finite-difference slope, safeguarded by bisection.
        dz = 1e-6 * max(1.0, abs(z))
        t2, y2, _ = tof_of_z(z + dz)
        z_new = z + (tof - t) * dz / (t2 - t) if np.isfinite(t2) and t2 != t else np.nan
        z = z_new if np.isfinite(z_new) and z_low < z_new < z_high else (z_low + z_high) / 2.0
    else:
        raise RuntimeError("Lambert iteration did not converge")

    f = 1.0 - y / r1n
    g = a_param * np.sqrt(y / mu)
    g_dot = 1.0 - y / r2n
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    return v1, v2
