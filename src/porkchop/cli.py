"""Command-line interface: scan a transfer window and plot the porkchop."""

from __future__ import annotations

import argparse

from .ephemeris import PLANETS, julian_date
from .plot import plot_porkchop
from .transfer import porkchop_grid

__all__ = ["main"]


def _parse_date(text: str) -> float:
    try:
        year, month, day = (int(part) for part in text.split("-"))
    except ValueError as exc:
        raise SystemExit(f"bad date {text!r}, expected YYYY-MM-DD") from exc
    return julian_date(year, month, day)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="porkchop", description=__doc__)
    parser.add_argument("--origin", default="earth", choices=sorted(PLANETS))
    parser.add_argument("--target", default="mars", choices=sorted(PLANETS))
    parser.add_argument("--depart-start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--depart-end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--tof-min", type=float, default=120.0, help="days")
    parser.add_argument("--tof-max", type=float, default=320.0, help="days")
    parser.add_argument("--grid", type=int, default=80, help="points per axis")
    parser.add_argument("--out", default="porkchop.png")
    args = parser.parse_args(argv)

    grid = porkchop_grid(
        args.origin,
        args.target,
        _parse_date(args.depart_start),
        _parse_date(args.depart_end),
        args.tof_min,
        args.tof_max,
        n_depart=args.grid,
        n_arrive=args.grid,
    )
    best = grid.best()
    print(
        f"best transfer: C3 = {best.c3_depart:.2f} km^2/s^2, "
        f"arrival v-inf = {best.vinf_arrive:.2f} km/s, "
        f"TOF = {best.arrive_jd - best.depart_jd:.0f} days"
    )
    out = plot_porkchop(
        grid, args.out, title=f"{args.origin.title()} → {args.target.title()} porkchop"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
