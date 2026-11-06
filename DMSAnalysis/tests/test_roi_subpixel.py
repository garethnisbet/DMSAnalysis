"""The sub-pixel ROI integrator and the splatted simulated image.

The legacy integrator (`msroi`) samples each ROI at whole-pixel offsets, takes
the nearest pixel, and crosses the line along one chord normal per ROI; the
simulated lines are set on whole pixels.  So the simulated centres, and the
fit's chi^2, move in steps as the parameters change.  With a bin step
(`roi_sampler` / `roi_profiles`, `dmsfit_ico_hkl.setRoiBinStep`) the strip is
sampled at sub-pixel bins along the local normal and the lines are splatted
bilinearly.  Pinned here:

* a line's sub-pixel position is recovered, and a curved line is not broadened;
* the x coordinate and its orientation are the legacy ones (centres, manual
  centres and sessions keep their meaning);
* the batched gathers equal a direct bilinear interpolation;
* splatting conserves weight and the first moment;
* the engine's chi^2 is smooth and symmetric about the truth, where the legacy
  one is a staircase; and the legacy path is unchanged when no step is set;
* the slider builds, fits and exports with the step, and integrates data and
  simulation through the same one.

Run standalone:
    python -m DMSAnalysis.tests.test_roi_subpixel
or under pytest:
    pytest DMSAnalysis/tests/test_roi_subpixel.py
"""

import json
import os

import numpy as np
from scipy import ndimage

from .. import ts_quasi as ts
from .test_dms_curves import cubic_kwargs, make_engine
from .test_fit_eval_speedups import _kernel_from_lines

SHAPE = (200, 200)


def _line_plane(r0, c0, ang, length, shape=SHAPE):
    """A one-pixel-wide rasterised straight path, as the ROI builder lays one."""
    t = np.linspace(-length / 2, length / 2, int(4 * length))
    r = np.round(r0 + t * np.sin(ang)).astype(int)
    c = np.round(c0 + t * np.cos(ang)).astype(int)
    p = np.zeros(shape)
    p[r, c] = 1
    return p


def _line_image(r0, c0, ang, offset, sig=1.5, shape=SHAPE):
    """Gaussian-profile straight line through (r0, c0) shifted `offset` px
    along its normal (-sin, cos) rotated to (row, col) = (cos, -sin)."""
    rr, cc = np.mgrid[:shape[0], :shape[1]].astype(float)
    n = np.array([np.cos(ang), -np.sin(ang)])      # normal in (row, col)
    d = (rr - r0) * n[0] + (cc - c0) * n[1] - offset
    return 100.0 * np.exp(-0.5 * (d / sig) ** 2) + 5.0


def _centre(img, plane, width, step):
    x, y, _ = ts.roi_curve(img, plane, width, step)
    coef, _, _ = ts.peakfit(x, y, 'gauss', None)
    return coef


# ── the integrator ─────────────────────────────────────────────────────────────

def test_sub_pixel_offsets_are_recovered_linearly():
    plane = _line_plane(100, 100, 0.37, 120)
    base = _centre(_line_image(100, 100, 0.37, 0.0), plane, 21, 0.25)[2]
    for off in np.linspace(-0.9, 0.9, 13):
        c = _centre(_line_image(100, 100, 0.37, off), plane, 21, 0.25)[2]
        # the chord normal of this path points along -n, so x runs against off
        assert abs(abs(c - base) - abs(off)) < 0.02, (off, c - base)


def test_x_axis_and_orientation_are_the_legacy_ones():
    plane = _line_plane(100, 100, 0.37, 120)
    x_new, _, _ = ts.roi_curve(np.zeros(SHAPE), plane, 21, 0.25)
    x_old, _, _ = ts.roi_curve(np.zeros(SHAPE), plane, 21, None)
    assert x_new[0] == x_old[0] == 0
    assert abs(x_new[-1] - x_old[-1]) < 1e-9
    a = _line_image(100, 100, 0.37, 0.0)
    b = _line_image(100, 100, 0.37, 2.0)
    d_new = _centre(b, plane, 21, 0.25)[2] - _centre(a, plane, 21, 0.25)[2]
    d_old = _centre(b, plane, 21, None)[2] - _centre(a, plane, 21, None)[2]
    assert np.sign(d_new) == np.sign(d_old) and abs(abs(d_new) - 2.0) < 0.05


def test_a_curved_line_is_not_broadened():
    """The legacy chord normal cuts a curved line obliquely toward the ROI's
    ends, widening the profile by ~1/cos of the tilt; the local normal does
    not.  The effect is modest (the strip already follows the line pixel by
    pixel) — a few percent on this 63-degree arc."""
    rad, sig = 120.0, 1.5
    r0, c0 = 190.0, 100.0
    a = np.linspace(np.pi / 2 - 0.55, np.pi / 2 + 0.55, 2000)
    rr, cc = r0 - rad * np.sin(a), c0 + rad * np.cos(a)
    plane = np.zeros(SHAPE)
    plane[np.round(rr).astype(int), np.round(cc).astype(int)] = 1
    R, C = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    img = 100 * np.exp(-0.5 * ((np.hypot(R - r0, C - c0) - rad) / sig) ** 2)
    s_new = abs(_centre(img, plane, 31, 0.25)[0])
    s_old = abs(_centre(img, plane, 31, None)[0])
    assert abs(s_new - sig) < 0.15, s_new
    assert s_old > s_new + 0.02, (s_old, s_new)


def test_a_centre_outside_its_strip_is_not_a_measurement():
    """A Gaussian dragged off the curve (here by a ramp with no peak) must
    not become a target: it is marked not found."""
    plane = _line_plane(100, 100, 0.37, 120)
    rr, cc = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    ramp = 0.5 * rr + 0.3 * cc
    x = ts.roi_curve(ramp, plane, 21, 0.25)[0]
    assert np.isnan(ts._centre_in_strip([1, 1, x[-1] + 5, 0], x)[2])
    assert np.isnan(ts._centre_in_strip([1, 1, -3.0, 0], x)[2])
    assert ts._centre_in_strip([1, 1, 7.5, 0], x)[2] == 7.5
    kernel = plane[:, :, None]
    co = ts.multiroifit(ramp, kernel, 21, 10, 'gauss', None, step=0.25)[0]
    assert np.isnan(co[0, 2]) or x[0] <= co[0, 2] <= x[-1]


def test_batched_gathers_equal_direct_interpolation():
    rng = np.random.default_rng(1)
    img = rng.random(SHAPE)
    planes = [_line_plane(100, 100, a, 80) for a in (0.1, 0.9, 2.0)]
    smp = [ts.roi_sampler(p, 25, 0.25) for p in planes] + [None]
    got = ts.roi_profiles(img, smp)
    for s, g in zip(smp, got):
        if s is None:
            assert g.size == 0
            continue
        P = s['coords'].reshape(-1, 2)
        inb = ((P[:, 0] >= 0) & (P[:, 0] <= SHAPE[0] - 1) &
               (P[:, 1] >= 0) & (P[:, 1] <= SHAPE[1] - 1))
        v = np.zeros(len(P))
        v[inb] = ndimage.map_coordinates(img, P[inb].T, order=1)
        ref = v.reshape(s['coords'].shape[:2]).sum(1)
        assert np.allclose(g, ref, rtol=0, atol=1e-9)


def test_splat_conserves_weight_and_position():
    rng = np.random.default_rng(2)
    r = rng.uniform(5, 195, 500)
    c = rng.uniform(5, 195, 500)
    w = rng.uniform(0.2, 1.0, 500)
    img = ts.splat_points(SHAPE, r, c, w)
    assert abs(img.sum() - w.sum()) < 1e-9
    for k in range(5):
        one = ts.splat_points(SHAPE, r[k:k+1], c[k:k+1])
        R, C = np.mgrid[:SHAPE[0], :SHAPE[1]]
        assert abs((one * R).sum() - r[k]) < 1e-9
        assert abs((one * C).sum() - c[k]) < 1e-9


# ── the engine ─────────────────────────────────────────────────────────────────

def _engine_case(method):
    kw = cubic_kwargs(numsteps=400)
    dms = make_engine(method=method, **kw)
    kernel = _kernel_from_lines(make_engine(method='circle', **kw),
                                np.shape(dms.imdata))
    dms.kernel, dms.width, dms.simsigma = kernel, 21, 2.0
    slots = list(ts.reduced_param_indices('cubic', True, False))
    return dms, dms.ig_full[slots].copy(), slots.index(11)    # dxrot


def _chi2_scan(dms, x0, j, step):
    dms.setRoiBinStep(step)
    dms.imcalc(x0)
    dms.centres = dms._simcoeffs()[:, 2:3].copy()
    hs = np.linspace(-0.01, 0.01, 41)
    out = []
    for h in hs:
        x = x0.copy()
        x[j] += h
        out.append(dms.fit(x))
    return np.array(out)


def test_engine_chi2_is_smooth_and_symmetric_with_sub_pixel_bins():
    for method in ('circle', 'sweep'):
        dms, x0, j = _engine_case(method)
        v = _chi2_scan(dms, x0, j, 0.25)
        assert v[20] < 1e-12
        d = np.diff(v)
        assert np.all(d[:20] < 0) and np.all(d[20:] > 0), method
        assert np.allclose(v, v[::-1], rtol=0.02, atol=1e-9), method


def test_legacy_engine_is_a_staircase_and_unchanged():
    """The motivating defect, and proof that no step means the old path."""
    dms, x0, j = _engine_case('sweep')
    v = _chi2_scan(dms, x0, j, None)
    assert len(np.unique(np.round(v, 12))) < len(v)   # flat steps
    fresh, x1, _ = _engine_case('sweep')              # never told a step
    fresh.imcalc(x1)
    dms.setRoiBinStep(None)
    dms.imcalc(x0)
    assert np.array_equal(fresh.imsim, dms.imsim)
    assert np.array_equal(fresh._simcoeffs(), dms._simcoeffs())


def test_bin_step_setting_names():
    assert ts.roi_bin_step(None) is None
    assert ts.roi_bin_step(0) is None
    assert ts.roi_bin_step('legacy') is None
    assert ts.roi_bin_step('0.5') == 0.5


# ── the slider ─────────────────────────────────────────────────────────────────

def _wait(win, app, attr):
    w = getattr(win, attr)
    while w is not None and w.isRunning():
        w.wait(100)
        app.processEvents()
    for _ in range(5):
        app.processEvents()


def test_slider_builds_fits_and_exports_with_the_bin_step():
    from ..qt import QtWidgets
    from .gui_harness import slider_on
    config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'configs', 'fit_conventional_tetragonal_PMN_PT_example.json')
    sl = slider_on(config)
    win = sl.win
    app = QtWidgets.QApplication.instance()
    saved_pm = win._peak_method
    try:
        if win._peak_method == 'ridge':
            win._peak_combo.setCurrentIndex(win._peak_combo.findData('gauss'))
        _wait(win, app, '_build_worker')
        win._on_clear_picks()
        win._apply_reflections(json.load(open(config))['crystal']['reflist_hkl'])
        win._sb_roibin.setValue(0.25)
        assert sl.roibin == 0.25
        assert win._on_build_curves()
        _wait(win, app, '_build_worker')
        assert win._fit_dms.roibin == 0.25 and win._kernel_roibin == 0.25
        n_old = len(np.arange(int(np.round(-win._kernel_width / 2)),
                              int(np.round(win._kernel_width / 2))))
        assert len(win._linedatax[0]) == int((n_old - 1) / 0.25) + 1
        # Moving the spinbox without a rebuild must not change how the live
        # simulated curves are integrated: they follow the kernel's step.
        win._sb_roibin.setValue(0.0)
        assert sl.roibin is None and win._kernel_roibin == 0.25
        win._try_draw_sim_lines()
        assert len(win._last_sim_lines[0][0]) == len(win._linedatax[0])
        assert win._build_workflow_config()['roi']['bin_step_per_zoom'] is None
        win._sb_roibin.setValue(0.5)
        data = win._session_dict()
        assert data['roi_bin_step'] == 0.5
        win._sb_roibin.setValue(0.0)
        win._set_roibin(data['roi_bin_step'])
        assert sl.roibin == 0.5 and abs(win._sb_roibin.value() - 0.5) < 1e-9
    finally:
        win._sb_roibin.setValue(ts.ROI_BIN_STEP)
        win._peak_combo.setCurrentIndex(win._peak_combo.findData(saved_pm))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok ', name)
