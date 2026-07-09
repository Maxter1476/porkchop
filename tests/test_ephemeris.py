import numpy as np
import pytest

from porkchop import AU_KM, MU_SUN, julian_date, planet_state


def test_julian_date_epochs():
    assert julian_date(2000, 1, 1, 12.0) == pytest.approx(2451545.0)  # J2000
    assert julian_date(1970, 1, 1) == pytest.approx(2440587.5)  # Unix epoch
    assert julian_date(2026, 7, 8) == pytest.approx(2461229.5)


@pytest.mark.parametrize(
    ("planet", "a_au", "period_days"),
    [
        ("mercury", 0.387, 88.0),
        ("venus", 0.723, 224.7),
        ("earth", 1.000, 365.25),
        ("mars", 1.524, 687.0),
        ("jupiter", 5.203, 4332.6),
    ],
)
def test_distances_and_periods(planet, a_au, period_days):
    from porkchop import PLANETS

    jd = julian_date(2026, 1, 1)
    r, v = planet_state(planet, jd)
    # Distance within the planet's perihelion-aphelion range
    dist_au = np.linalg.norm(r) / AU_KM
    ecc = PLANETS[planet][0][1]
    assert a_au * (1 - ecc) * 0.99 <= dist_au <= a_au * (1 + ecc) * 1.01
    # Vis-viva: recover the semi-major axis, then the period, from the state
    a_from_state = 1.0 / (2.0 / np.linalg.norm(r) - np.dot(v, v) / MU_SUN)
    period = 2.0 * np.pi * np.sqrt(a_from_state**3 / MU_SUN) / 86400.0
    assert period == pytest.approx(period_days, rel=0.01)


def test_earth_speed_about_30_kms():
    _, v = planet_state("earth", julian_date(2026, 3, 15))
    assert np.linalg.norm(v) == pytest.approx(29.8, abs=0.8)


def test_orbit_closes_over_one_period():
    """Earth's position one sidereal year apart moves by only a small angle."""
    jd = julian_date(2026, 1, 1)
    r0, _ = planet_state("earth", jd)
    r1, _ = planet_state("earth", jd + 365.256)
    cos_angle = np.dot(r0, r1) / (np.linalg.norm(r0) * np.linalg.norm(r1))
    assert np.degrees(np.arccos(np.clip(cos_angle, -1, 1))) < 1.0


def test_earth_mars_synodic_period():
    """Heliocentric longitudes realign every ~780 days (the synodic period)."""

    def separation(jd):
        re, _ = planet_state("earth", jd)
        rm, _ = planet_state("mars", jd)
        return np.arctan2(re[1], re[0]) - np.arctan2(rm[1], rm[0])

    jd0 = julian_date(2026, 1, 1)
    base = separation(jd0)
    # scan 700..860 days for the next alignment with the same phase
    days = np.arange(700.0, 860.0, 0.25)
    diffs = [abs(np.remainder(separation(jd0 + d) - base + np.pi, 2 * np.pi) - np.pi) for d in days]
    realign = days[int(np.argmin(diffs))]
    # Mars's eccentric orbit makes individual realignment intervals range
    # roughly 764-810 days around the 779.9-day mean synodic period.
    assert 755.0 <= realign <= 815.0


def test_unknown_planet_raises():
    with pytest.raises(KeyError):
        planet_state("pluto", 2451545.0)
