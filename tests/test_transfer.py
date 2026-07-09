import numpy as np

from porkchop import evaluate_transfer, julian_date, porkchop_grid


def test_mars_2026_window_is_found():
    """The Earth->Mars late-2026 launch window must appear in the scan with a
    plausible minimum departure C3 (historically ~ 11-20 km^2/s^2)."""
    grid = porkchop_grid(
        "earth",
        "mars",
        julian_date(2026, 9, 1),
        julian_date(2027, 2, 1),
        tof_min_days=140,
        tof_max_days=340,
        n_depart=25,
        n_arrive=25,
    )
    best = grid.best()
    assert 5.0 < best.c3_depart < 25.0
    assert 1.5 < best.vinf_arrive < 5.0
    tof = best.arrive_jd - best.depart_jd
    assert 140 <= tof <= 340


def test_transfer_point_consistency():
    point = evaluate_transfer(
        "earth", "mars", julian_date(2026, 11, 1), julian_date(2027, 8, 1)
    )
    assert point.c3_depart > 0
    assert point.vinf_arrive > 0
    assert point.total_dv_proxy == np.sqrt(point.c3_depart) + point.vinf_arrive


def test_grid_shape_and_nan_masking():
    grid = porkchop_grid(
        "earth",
        "venus",
        julian_date(2026, 1, 1),
        julian_date(2026, 3, 1),
        tof_min_days=80,
        tof_max_days=200,
        n_depart=10,
        n_arrive=12,
    )
    assert grid.c3.shape == (12, 10)
    assert np.isfinite(grid.c3).any()
    # points outside the TOF band must be NaN
    assert np.isnan(grid.c3).any()
