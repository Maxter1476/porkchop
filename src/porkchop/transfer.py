"""Interplanetary transfer scans: C3, arrival v-infinity, porkchop grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ephemeris import planet_state
from .kepler import MU_SUN
from .lambert import lambert

__all__ = ["TransferPoint", "PorkchopGrid", "evaluate_transfer", "porkchop_grid"]

DAY_S = 86400.0


@dataclass(frozen=True)
class TransferPoint:
    """One Lambert transfer evaluated between two planets."""

    depart_jd: float
    arrive_jd: float
    c3_depart: float  # km^2/s^2
    vinf_arrive: float  # km/s
    total_dv_proxy: float  # sqrt(C3) + vinf, km/s


@dataclass(frozen=True)
class PorkchopGrid:
    """Porkchop scan results on a departure x arrival date grid."""

    depart_jds: np.ndarray
    arrive_jds: np.ndarray
    c3: np.ndarray  # shape (n_arrive, n_depart)
    vinf: np.ndarray

    def best(self) -> TransferPoint:
        """Grid point minimizing the dv proxy sqrt(C3) + vinf."""
        proxy = np.sqrt(np.where(np.isfinite(self.c3), self.c3, np.inf)) + np.where(
            np.isfinite(self.vinf), self.vinf, np.inf
        )
        k = int(np.nanargmin(proxy))
        ia, id_ = np.unravel_index(k, proxy.shape)
        return TransferPoint(
            depart_jd=float(self.depart_jds[id_]),
            arrive_jd=float(self.arrive_jds[ia]),
            c3_depart=float(self.c3[ia, id_]),
            vinf_arrive=float(self.vinf[ia, id_]),
            total_dv_proxy=float(proxy[ia, id_]),
        )


def evaluate_transfer(
    origin: str, target: str, depart_jd: float, arrive_jd: float
) -> TransferPoint:
    """Solve the Lambert arc between two planets on given dates."""
    tof = (arrive_jd - depart_jd) * DAY_S
    if tof <= 0:
        raise ValueError("arrival must be after departure")
    r1, v_planet1 = planet_state(origin, depart_jd)
    r2, v_planet2 = planet_state(target, arrive_jd)
    v1, v2 = lambert(r1, r2, tof, MU_SUN)
    vinf_dep = float(np.linalg.norm(v1 - v_planet1))
    vinf_arr = float(np.linalg.norm(v2 - v_planet2))
    return TransferPoint(
        depart_jd=depart_jd,
        arrive_jd=arrive_jd,
        c3_depart=vinf_dep**2,
        vinf_arrive=vinf_arr,
        total_dv_proxy=vinf_dep + vinf_arr,
    )


def porkchop_grid(
    origin: str,
    target: str,
    depart_jd0: float,
    depart_jd1: float,
    tof_min_days: float,
    tof_max_days: float,
    *,
    n_depart: int = 60,
    n_arrive: int = 60,
) -> PorkchopGrid:
    """Scan a departure-date x arrival-date grid of Lambert transfers.

    Unsolvable or degenerate geometries are left as NaN in the grids.
    """
    depart_jds = np.linspace(depart_jd0, depart_jd1, n_depart)
    arrive_jds = np.linspace(depart_jd0 + tof_min_days, depart_jd1 + tof_max_days, n_arrive)
    c3 = np.full((n_arrive, n_depart), np.nan)
    vinf = np.full((n_arrive, n_depart), np.nan)
    for jj, dep in enumerate(depart_jds):
        for ii, arr in enumerate(arrive_jds):
            tof_days = arr - dep
            if not (tof_min_days <= tof_days <= tof_max_days):
                continue
            try:
                point = evaluate_transfer(origin, target, dep, arr)
            except (RuntimeError, ValueError):
                continue
            c3[ii, jj] = point.c3_depart
            vinf[ii, jj] = point.vinf_arrive
    return PorkchopGrid(depart_jds=depart_jds, arrive_jds=arrive_jds, c3=c3, vinf=vinf)
