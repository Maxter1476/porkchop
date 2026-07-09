"""Porkchop contour plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .transfer import PorkchopGrid

__all__ = ["plot_porkchop"]


def _jd_to_label(jd: float) -> str:
    """Gregorian yyyy-mm-dd from a Julian date (inverse Fliegel-Van Flandern)."""
    j = int(jd + 0.5)
    f = j + 1401 + (((4 * j + 274277) // 146097) * 3) // 4 - 38
    e = 4 * f + 3
    g = (e % 1461) // 4
    h = 5 * g + 2
    day = (h % 153) // 5 + 1
    month = (h // 153 + 2) % 12 + 1
    year = e // 1461 - 4716 + (12 + 2 - month) // 12
    return f"{year:04d}-{month:02d}-{day:02d}"


def plot_porkchop(
    grid: PorkchopGrid,
    path: str | Path,
    *,
    title: str = "Porkchop plot",
    c3_levels: tuple[float, ...] = (10, 12, 15, 20, 25, 30, 40, 50),
    vinf_levels: tuple[float, ...] = (2.5, 3, 3.5, 4, 5, 6, 8),
) -> Path:
    """Contour departure C3 (solid) and arrival v-infinity (dashed)."""
    dep = grid.depart_jds - grid.depart_jds[0]
    arr = grid.arrive_jds - grid.arrive_jds[0]
    fig, ax = plt.subplots(figsize=(9, 7))

    cs1 = ax.contour(dep, arr, grid.c3, levels=c3_levels, cmap="viridis")
    ax.clabel(cs1, fmt=lambda v: f"C3={v:g}")
    cs2 = ax.contour(dep, arr, grid.vinf, levels=vinf_levels, colors="crimson", linestyles="--")
    ax.clabel(cs2, fmt=lambda v: f"v∞={v:g}")

    best = grid.best()
    ax.plot(
        best.depart_jd - grid.depart_jds[0],
        best.arrive_jd - grid.arrive_jds[0],
        "k*",
        ms=14,
        label=(
            f"best: depart {_jd_to_label(best.depart_jd)}, "
            f"C3={best.c3_depart:.1f} km²/s², v∞={best.vinf_arrive:.2f} km/s"
        ),
    )

    n_ticks = 6
    dep_ticks = np.linspace(dep[0], dep[-1], n_ticks)
    arr_ticks = np.linspace(arr[0], arr[-1], n_ticks)
    ax.set_xticks(dep_ticks)
    ax.set_xticklabels(
        [_jd_to_label(grid.depart_jds[0] + t) for t in dep_ticks], rotation=30, ha="right"
    )
    ax.set_yticks(arr_ticks)
    ax.set_yticklabels([_jd_to_label(grid.arrive_jds[0] + t) for t in arr_ticks])
    ax.set_xlabel("departure date")
    ax.set_ylabel("arrival date")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
