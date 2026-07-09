import numpy as np
import pytest

from porkchop import MU_EARTH, propagate, stumpff_c, stumpff_s


def test_stumpff_continuity_at_zero():
    for fn, limit in ((stumpff_c, 0.5), (stumpff_s, 1.0 / 6.0)):
        assert fn(0.0) == pytest.approx(limit)
        assert fn(1e-9) == pytest.approx(fn(-1e-9), rel=1e-6)


def test_circular_orbit_period():
    """A circular orbit propagated one period returns to its start."""
    r = 7000.0  # km
    v = np.sqrt(MU_EARTH / r)
    period = 2.0 * np.pi * np.sqrt(r**3 / MU_EARTH)
    r0 = np.array([r, 0.0, 0.0])
    v0 = np.array([0.0, v, 0.0])
    r1, v1 = propagate(r0, v0, period, MU_EARTH)
    assert np.allclose(r1, r0, atol=1e-3)
    assert np.allclose(v1, v0, atol=1e-6)


def test_quarter_period_circular():
    r = 7000.0
    v = np.sqrt(MU_EARTH / r)
    period = 2.0 * np.pi * np.sqrt(r**3 / MU_EARTH)
    r1, v1 = propagate(np.array([r, 0, 0]), np.array([0, v, 0]), period / 4, MU_EARTH)
    assert np.allclose(r1, [0, r, 0], atol=1e-3)
    assert np.allclose(v1, [-v, 0, 0], atol=1e-6)


@pytest.mark.parametrize("dt", [100.0, 3600.0, 86400.0, -3600.0])
def test_energy_and_momentum_conserved(dt):
    rng = np.random.default_rng(4)
    r0 = np.array([8000.0, 1000.0, -500.0])
    v0 = np.array([0.5, 7.0, 1.0]) + rng.normal(0, 0.1, 3)
    r1, v1 = propagate(r0, v0, dt, MU_EARTH)

    def energy(r, v):
        return 0.5 * np.dot(v, v) - MU_EARTH / np.linalg.norm(r)

    assert energy(r1, v1) == pytest.approx(energy(r0, v0), rel=1e-9)
    assert np.allclose(np.cross(r1, v1), np.cross(r0, v0), rtol=1e-9)


def test_hyperbolic_propagation():
    """Above escape speed the propagator must still conserve energy."""
    r0 = np.array([7000.0, 0.0, 0.0])
    v_esc = np.sqrt(2 * MU_EARTH / 7000.0)
    v0 = np.array([0.0, 1.2 * v_esc, 0.0])
    r1, v1 = propagate(r0, v0, 5 * 3600.0, MU_EARTH)
    e0 = 0.5 * np.dot(v0, v0) - MU_EARTH / np.linalg.norm(r0)
    e1 = 0.5 * np.dot(v1, v1) - MU_EARTH / np.linalg.norm(r1)
    assert e1 == pytest.approx(e0, rel=1e-9)
    assert np.linalg.norm(r1) > 7000.0
