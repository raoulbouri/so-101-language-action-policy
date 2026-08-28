import numpy as np

from so101_sim.trajectory import quintic_scaling, quintic_segment, steps_for


def test_quintic_endpoints_and_smoothness():
    tau = np.linspace(0, 1, 1001)
    s = quintic_scaling(tau)
    assert np.isclose(s[0], 0.0) and np.isclose(s[-1], 1.0)
    ds = np.gradient(s, tau)
    dds = np.gradient(ds, tau)
    # Zero velocity and acceleration at both ends is the whole point. np.gradient
    # falls back to a one-sided stencil at the boundary, so compare the endpoint
    # acceleration against the peak rather than against an absolute epsilon.
    assert abs(ds[0]) < 1e-3 and abs(ds[-1]) < 1e-3
    peak = np.abs(dds).max()
    assert abs(dds[0]) < 0.01 * peak
    assert abs(dds[-1]) < 0.01 * peak


def test_quintic_second_derivative_vanishes_analytically():
    # s''(tau) = 60t - 180t^2 + 120t^3, which is exactly 0 at both endpoints
    for tau in (0.0, 1.0):
        assert abs(60 * tau - 180 * tau**2 + 120 * tau**3) < 1e-12


def test_quintic_is_monotonic():
    s = quintic_scaling(np.linspace(0, 1, 500))
    assert np.all(np.diff(s) >= -1e-12)


def test_segment_shape_excludes_the_start_frame():
    seg = quintic_segment(np.zeros(3), np.ones(3), 10)
    assert seg.shape == (10, 3)
    assert np.allclose(seg[-1], 1.0)
    assert not np.allclose(seg[0], 0.0)


def test_steps_for_is_at_least_one():
    assert steps_for(0.0, 50.0) == 1
    assert steps_for(1.0, 50.0) == 50
