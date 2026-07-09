import numpy as np
import pytest

from porkchop import MU_EARTH, MU_SUN, lambert, propagate


def test_vallado_example():
    """Vallado 'Fundamentals of Astrodynamics' Example 7-5 (geocentric).

    r1 = (15945.34, 0, 0) km, r2 = (12214.83899, 10249.46731, 0) km,
    tof = 76 min. Published answer: v1 = (2.058913, 2.915965, 0) km/s.
    """
    r1 = np.array([15945.34, 0.0, 0.0])
    r2 = np.array([12214.83899, 10249.46731, 0.0])
    v1, v2 = lambert(r1, r2, 76.0 * 60.0, MU_EARTH)
    assert np.allclose(v1, [2.058913, 2.915965, 0.0], atol=2e-4)


@pytest.mark.parametrize("period_fraction", [0.15, 0.45, 0.8])
def test_roundtrip_with_propagator(period_fraction):
    """Propagate a state, then Lambert between endpoints must recover v1, v2.

    This cross-validates the two independent algorithms against each other
    over elliptic geometries with plane changes. TOF stays below one orbital
    period — the solver is zero-revolution by design.
    """
    r0 = np.array([9000.0, 500.0, 300.0])
    v0 = np.array([-0.3, 6.2, 0.8])
    energy = 0.5 * np.dot(v0, v0) - MU_EARTH / np.linalg.norm(r0)
    a = -MU_EARTH / (2.0 * energy)
    period = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
    tof = period_fraction * period
    r1, v1 = propagate(r0, v0, tof, MU_EARTH)
    lv0, lv1 = lambert(r0, r1, tof, MU_EARTH)
    assert np.allclose(lv0, v0, atol=1e-5)
    assert np.allclose(lv1, v1, atol=1e-5)


def test_hohmann_limit():
    """A 180-degree coplanar circular-to-circular Lambert arc flown in the
    Hohmann transfer time must cost exactly the analytic Hohmann dv."""
    r_a, r_b = 1.0e8, 2.0e8  # km, heliocentric-ish
    a_t = (r_a + r_b) / 2.0
    tof = np.pi * np.sqrt(a_t**3 / MU_SUN)

    r1 = np.array([r_a, 0.0, 0.0])
    r2 = np.array([-r_b * np.cos(1e-4), r_b * np.sin(1e-4), 0.0])  # just under 180 deg
    v1, v2 = lambert(r1, r2, tof, MU_SUN)

    v_circ_a = np.sqrt(MU_SUN / r_a)
    dv_depart_analytic = v_circ_a * (np.sqrt(2 * r_b / (r_a + r_b)) - 1.0)
    dv_depart_lambert = np.linalg.norm(v1 - np.array([0.0, v_circ_a, 0.0]))
    assert dv_depart_lambert == pytest.approx(dv_depart_analytic, rel=2e-3)


def test_rejects_bad_inputs():
    r = np.array([7000.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        lambert(r, 2 * r, -10.0, MU_EARTH)
    with pytest.raises(ValueError):
        lambert(r, 2 * r, 3600.0, MU_EARTH)  # 0-degree transfer angle
