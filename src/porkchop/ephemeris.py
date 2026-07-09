"""Low-precision planetary ephemerides from JPL mean Keplerian elements.

Uses the Standish (JPL) approximate mean elements valid 1800-2050 AD:
six elements plus linear rates per Julian century of TDB. Accuracy is a few
thousandths of an AU over the validity window — ample for transfer-window
studies (the test suite checks distances, periods, and the Earth-Mars synodic
period against known values).

Reference: E. M. Standish, "Keplerian Elements for Approximate Positions of
the Major Planets", JPL/Caltech memorandum.
"""

from __future__ import annotations

import numpy as np

from .kepler import MU_SUN

__all__ = ["AU_KM", "PLANETS", "julian_date", "planet_state"]

AU_KM: float = 1.495978707e8
_DEG = np.pi / 180.0

# (a [au], e, i [deg], L [deg], long. perihelion [deg], long. asc. node [deg])
# and per-Julian-century rates, from the 1800-2050 AD table.
PLANETS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "mercury": (
        (0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593),
        (0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081),
    ),
    "venus": (
        (0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
        (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418),
    ),
    "earth": (  # Earth-Moon barycenter
        (1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0),
        (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0),
    ),
    "mars": (
        (1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
        (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343),
    ),
    "jupiter": (
        (5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
        (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106),
    ),
    "saturn": (
        (9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
        (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794),
    ),
}


def julian_date(year: int, month: int, day: int, hour: float = 0.0) -> float:
    """Julian date from a Gregorian calendar date (Fliegel-Van Flandern)."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn + (hour - 12.0) / 24.0


def _kepler_e(mean_anomaly: float, ecc: float, tol: float = 1e-12) -> float:
    """Solve Kepler's equation M = E - e sin E by Newton iteration."""
    e_anom = mean_anomaly if ecc < 0.8 else np.pi
    for _ in range(60):
        delta = (e_anom - ecc * np.sin(e_anom) - mean_anomaly) / (1.0 - ecc * np.cos(e_anom))
        e_anom -= delta
        if abs(delta) < tol:
            return e_anom
    raise RuntimeError("Kepler's equation did not converge")


def planet_state(name: str, jd: float) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric ecliptic-J2000 position (km) and velocity (km/s) at jd."""
    key = name.lower()
    if key not in PLANETS:
        raise KeyError(f"unknown planet {name!r}; have {sorted(PLANETS)}")
    elements, rates = PLANETS[key]
    t = (jd - 2451545.0) / 36525.0  # Julian centuries since J2000
    a_au, e, i, ell, varpi, omega = (el + rate * t for el, rate in zip(elements, rates, strict=True))

    a = a_au * AU_KM
    i *= _DEG
    omega *= _DEG
    arg_peri = (varpi - (omega / _DEG)) * _DEG
    m_anom = np.remainder((ell - varpi) * _DEG + np.pi, 2.0 * np.pi) - np.pi
    e_anom = _kepler_e(m_anom, e)

    # Perifocal position and velocity
    cos_e, sin_e = np.cos(e_anom), np.sin(e_anom)
    r_pf = np.array([a * (cos_e - e), a * np.sqrt(1.0 - e**2) * sin_e, 0.0])
    rn = float(np.linalg.norm(r_pf))
    factor = np.sqrt(MU_SUN * a) / rn
    v_pf = factor * np.array([-sin_e, np.sqrt(1.0 - e**2) * cos_e, 0.0])

    # Rotate perifocal -> ecliptic: Rz(-Omega) Rx(-i) Rz(-argperi)
    co, so = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(arg_peri), np.sin(arg_peri)
    rot = np.array(
        [
            [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
            [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
            [sw * si, cw * si, ci],
        ]
    )
    return rot @ r_pf, rot @ v_pf
