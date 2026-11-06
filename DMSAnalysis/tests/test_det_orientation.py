"""The detector mounting toggle: vertical or horizontal scattering.

The Pilatus 100K rides on the detector arm.  In vertical scattering 2θ runs
along its long axis and an azimuthal scan sweeps the DMS lines across the short
axis; in horizontal scattering the two swap.  The engines work in a frame with
component 0 normal to the scattering plane and component 2 in it, and
``ts.det_to_pixels`` puts those on the image: rows/cols in the vertical mode
(what every existing config is), cols/rows in the horizontal one.  The mounting
rides in bit 1 of ``scatv`` so every positional argument tuple carries it.

Run standalone:
    python -m DMSAnalysis.tests.test_det_orientation
or under pytest:
    pytest DMSAnalysis/tests/test_det_orientation.py
"""

import os
import json
import tempfile

import numpy as np

from .. import ts_quasi as ts


# ── the mapping ────────────────────────────────────────────────────────────────

def test_vertical_is_the_historical_map():
    """Vertical reproduces the formula every engine used before the toggle."""
    p0, p2 = np.array([3.0, -7.5]), np.array([11.0, 2.25])
    for sv in (0, 1):
        r, c = ts.det_to_pixels(p0, p2, 100.0, 200.0, ts.scatv_code(sv, 'vertical'))
        assert np.allclose(r, 100.0 + p0)
        assert np.allclose(c, 200.0 + (p2 if sv == 1 else -p2))


def test_horizontal_swaps_the_axes():
    p0, p2 = np.array([3.0, -7.5]), np.array([11.0, 2.25])
    for sv in (0, 1):
        r, c = ts.det_to_pixels(p0, p2, 100.0, 200.0, ts.scatv_code(sv, 'horizontal'))
        assert np.allclose(r, 100.0 + (p2 if sv == 1 else -p2))
        assert np.allclose(c, 200.0 + p0)


def test_inverse_round_trips_in_both_modes():
    rng = np.random.default_rng(0)
    p0, p2 = rng.normal(size=50) * 80, rng.normal(size=50) * 80
    for orient in ts.DET_ORIENTATIONS:
        for sv in (0, 1):
            code = ts.scatv_code(sv, orient)
            assert ts.scatv_orientation(code) == (sv, orient)
            r, c = ts.det_to_pixels(p0, p2, 104.6, 257.6, code)
            q0, q2 = ts.pixels_to_det(r, c, 104.6, 257.6, code)
            assert np.allclose(q0, p0) and np.allclose(q2, p2)


def test_orientation_names():
    assert ts.det_orientation(None) == 'vertical'
    assert ts.det_orientation(' Horizontal ') == 'horizontal'
    try:
        ts.det_orientation('sideways')
    except ValueError:
        pass
    else:
        raise AssertionError('bad name accepted')


# ── the engine ─────────────────────────────────────────────────────────────────

def _engine(code, size=401):
    lat = [4.0, 4.0, 4.0, 90.0, 90.0, 90.0]
    hkl = np.array([0.0, 0.0, 2.0])
    energy = 8.0
    thb = ts.bragg(lat, hkl, energy).th()[0]
    refs = np.array([[1, 1, 1], [1, 0, 2], [0, 1, 1], [-1, 1, 2], [2, 0, 2]],
                    dtype=float)
    ig = np.array(lat + [0, 0, 0, 0, 1200.0, 3.0, 2.0, 5.0, energy] + [0.0] * 9)
    c = (size - 1) / 2.0
    eng = ts.dmsfit_ico_hkl(
        np.matrix(refs), [thb - 27, thb + 10, 200], np.round(hkl), [-180, 180],
        45, np.zeros((1, 1)), np.zeros((1, 1, 1)), hkl,
        np.matrix([[1, 0, 0], [0, 0, 1]]), np.zeros((size, size)), 4.5,
        [1.0, 0.0, 0.0], 0.0, c, c, code, 'cubic', True, False,
        ig[10], ig[11], ig[12], ig[13], ig[14], np.matrix(np.zeros_like(refs)),
        list(ig[15:24]), ig[0])
    eng.setIGFull(ig)
    eng.setCalLattice(ig[:6])
    eng.setLattice(ig[:6])
    eng.imcalc(ig[list(ts.imcalc_param_indices('cubic', True, False))])
    return eng, c


def test_engine_lines_transpose_between_modes():
    """On a square image with the origin at its centre, the horizontal lines
    are the vertical lines transposed about the origin — and both modes put
    something on the plate."""
    for sv in (0, 1):
        ev, c = _engine(ts.scatv_code(sv, 'vertical'))
        eh, _ = _engine(ts.scatv_code(sv, 'horizontal'))
        n_on = 0
        for (cv, rv), (ch, rh) in zip(ev.dmslines, eh.dmslines):
            ok = np.isfinite(rv) & np.isfinite(rh)
            n_on += ok.sum()
            # horizontal: rows ← ±p2, cols ← p0; vertical: rows ← p0, cols ← ±p2
            assert np.allclose(rh[ok] - c, cv[ok] - c), sv
            assert np.allclose(ch[ok] - c, rv[ok] - c), sv
        assert n_on > 20, n_on


# ── the slider ─────────────────────────────────────────────────────────────────

def _config(tmpdir, orient=None):
    geo = {'hkl': [0.0, 0.0, 2.0], 'psi': 0.0,
           'px_unscaled': 100, 'py_unscaled': 200, 'scatv': 0}
    if orient is not None:
        geo['det_orientation'] = orient
    cfg = {
        'scan': {'scannum': 1, 'scanpath': tmpdir + os.sep,
                 'datapoint': 0, 'datapoint0': 0},
        'geometry': geo,
        'display': {'zoomval': 1, 'colourlim': [0, 1000], 'colmap': 'gray'},
        'roi': {'width_per_zoom': 45, 'comwidth_per_zoom': 5},
        'computation': {'numsteps': 60, 'simsigma_per_zoom': 4.5,
                        'thrange_delta': [-27, 10], 'bravais': 'cubic',
                        'curve_method': 'sweep'},
        'crystal': {'initial_guess_base': [4.0, 4.0, 4.0, 90.0, 90.0, 90.0]
                                          + [0.0] * 18,
                    'reflist_hkl': [[1, 1, 1], [0, 1, 1]]},
        'flags': {'save': 0, 'fit': 0, 'firstplot': 0, 'detoptimize': 1,
                  'energyopt': 0},
    }
    path = os.path.join(tmpdir, 'cfg.json')
    with open(path, 'w') as fh:
        json.dump(cfg, fh)
    return path


def test_slider_toggle_reaches_engines_export_and_session():
    from .gui_harness import slider_on
    sl = slider_on(_config(tempfile.mkdtemp()))
    win = sl.win
    try:
        win._det_combo.setCurrentIndex(win._det_combo.findData('horizontal'))
        assert sl.det_orient == 'horizontal'
        assert sl.scatv & ts.SCATV_HORIZONTAL
        for eng in (win._dms, win._dms_full, win._sel_dms, win._fit_dms):
            if eng is not None:
                assert eng.scatv == sl.scatv
        # a clicked pixel back-projects through the same map the overlay uses
        d = win._pixel_to_direction(150.0, 260.0)
        assert d is not None and np.isclose(np.linalg.norm(d), 1.0)

        cfg = win._build_workflow_config()
        assert cfg['geometry']['det_orientation'] == 'horizontal'
        assert cfg['geometry']['scatv'] in (0, 1)      # the sign bit only

        data = win._session_dict()
        assert data['det_orientation'] == 'horizontal'
        win._det_combo.setCurrentIndex(win._det_combo.findData('vertical'))
        assert sl.det_orient == 'vertical'
        win._restore_from_dict(data)
        assert sl.det_orient == 'horizontal'
        assert win._det_combo.currentData() == 'horizontal'

        # a session from before the toggle restores as vertical
        data.pop('det_orientation')
        win._restore_from_dict(data)
        assert sl.det_orient == 'vertical'
    finally:
        win._det_combo.setCurrentIndex(win._det_combo.findData('vertical'))


def test_flip_2theta_toggle():
    """The Flip 2θ box is the scatv sign: it reaches every engine, the export
    and the session, and mirrors the overlay along the 2θ axis."""
    from .gui_harness import slider_on
    sl = slider_on(_config(tempfile.mkdtemp()))
    win = sl.win
    try:
        win._chk_flip2th.setChecked(False)
        d0 = win._pixel_to_direction(120.0, 300.0)
        win._chk_flip2th.setChecked(True)
        assert sl.scatv & 1
        for eng in (win._dms, win._dms_full, win._sel_dms, win._fit_dms):
            if eng is not None:
                assert eng.scatv == sl.scatv
        # mirrored about the origin column: the same direction is now found on
        # the other side of py
        py = float(win._dms.py)
        d1 = win._pixel_to_direction(120.0, 2 * py - 300.0)
        assert np.allclose(d0, d1), (d0, d1)

        assert win._build_workflow_config()['geometry']['scatv'] == 1
        data = win._session_dict()
        assert data['scatv'] == 1
        win._chk_flip2th.setChecked(False)
        win._restore_from_dict(data)
        assert win._chk_flip2th.isChecked() and sl.scatv & 1
    finally:
        win._chk_flip2th.setChecked(False)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('all passed')
