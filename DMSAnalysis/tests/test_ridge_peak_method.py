"""The slider's 'Ridge + cone' peak method, driven through the real GUI.

Selecting it makes Build curves measure the checked reflections' lines along
their predicted arcs and install the ridge + cone residual as the fit engine's
objective (`dmsfit_ico_hkl.setObjective`), so every optimiser in FitWorker
scores it unchanged.  Pinned here on a synthetic image rendered at a known
geometry, from a perturbed start:

* the build measures points and the live readout reports the ridge chi^2;
* a fit drives the lines onto the measured points and lands near the truth;
* the post-fit rebuild keeps the same points (the fit's reference), and the
  run record says what it was scored on;
* switching back to a ROI method removes the objective.

Run standalone:
    python -m DMSAnalysis.tests.test_ridge_peak_method
or under pytest:
    pytest DMSAnalysis/tests/test_ridge_peak_method.py
"""

import json
import os
import tempfile

import numpy as np
from scipy import ndimage

from .gui_harness import slider_on
from .. import ridgefit as rf

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'fit_conventional_tetragonal_PMN_PT_example.json')


def _render(model, x, shape, amp=200.0, bg=50.0, sig=2.0, seed=0):
    im = np.zeros(shape)
    for _, rr, cc in model.predicted_lines(x, shape, spacing=0.25):
        ok = np.isfinite(rr)
        r, c = rr[ok], cc[ok]
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (1, 0, fr * (1 - fc)),
                          (0, 1, (1 - fr) * fc), (1, 1, fr * fc)):
            np.add.at(im, (np.clip(r0 + dr, 0, shape[0] - 1),
                           np.clip(c0 + dc, 0, shape[1] - 1)), w * 0.25)
    im = ndimage.gaussian_filter(im, sig) * amp * np.sqrt(2 * np.pi) * sig + bg
    return np.random.default_rng(seed).poisson(im).astype(float)


def _config_session(sl, cfg_path):
    """A session dict that puts the window in the state a fresh start on
    ``cfg_path`` would: its crystal type, scan, geometry and reflections.

    The GUI harness is a singleton, so the window may have been built on
    another test's config, and earlier tests leave it changed (a different
    scan loaded, the detector toggles moved) even when it was built on this
    one.  This test renders a synthetic image at the
    tetragonal config's geometry and measures ridges on it; on another
    config's geometry (detector distance, hkl, beam centre) most of its lines
    miss the plate and the ridge holds a fraction of the points.  Restoring a
    session is the app's own path for switching all of that at once."""
    cfg = json.load(open(cfg_path))
    geo, comp, roi = cfg['geometry'], cfg['computation'], cfg.get('roi', {})
    ig = np.array(cfg['crystal']['initial_guess_base'], dtype=float)
    zoom = cfg['display']['zoomval']
    ig[10] = ig[10] / 2 * zoom          # the slider's startup conversions:
    scanpath = os.path.join(sl.dat2config.resolve_scanpath(
        cfg['scan']['scanpath'], cfg_path), '')   # restore appends the file name
    try:                                # detdist halved, energy made absolute
        exp = cfg.get('experiment') or sl.dat2config.extract_metadata(
            os.path.join(scanpath, '%s.dat' % cfg['scan']['scannum']),
            cfg['scan']['datapoint'], cfg['scan']['datapoint0'])
        ig[14] += float(exp['energy'])
    except Exception:                   # no scan on disk: keep the window's
        ig[14] += float(sl.win.ig[14])
    refs = cfg['crystal']['reflist_hkl']
    return {
        'bravais': comp['bravais'],
        'pc_transform': comp.get('pseudocubic_transform', 1),
        'scan': {'scanpath': scanpath,
                 'scannum': cfg['scan']['scannum'],
                 'datapoint': cfg['scan']['datapoint'],
                 'datapoint0': cfg['scan']['datapoint0']},
        'hkl': geo['hkl'], 'psi': geo['psi'],
        'px': geo['px_unscaled'] * zoom, 'py': geo['py_unscaled'] * zoom,
        'initial_guess': ig.tolist(),
        'ref_6d': refs, 'ref_6d_checked': [True] * len(refs),
        'peak_method': comp.get('peak_method', 'gauss'),
        'curve_method': comp.get('curve_method', 'sweep'),
        'roi_bin_step': sl.ts.roi_bin_from_config(roi, zoom),
        'det_orientation': geo.get('det_orientation', 'vertical'),
        'scatv': geo.get('scatv', 0),
        'curves_built': False,
    }


def _wait(win, app, attr):
    w = getattr(win, attr)
    while w is not None and w.isRunning():
        w.wait(100)
        app.processEvents()
    for _ in range(5):
        app.processEvents()


def test_ridge_peak_method_builds_scores_and_fits():
    from ..qt import QtWidgets
    sl = slider_on(CONFIG)
    win = sl.win
    app = QtWidgets.QApplication.instance()
    saved = (win._imdata, win._peak_method, win._active_method, win.ig.copy())
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)                    # the fit's run record goes to ./Processing
    # Always reset: even when this config built the window, earlier tests
    # (scan loads, detector toggles) leave it somewhere else.
    before = win._session_dict()
    win._restore_from_dict(_config_session(sl, CONFIG))
    _wait(win, app, '_build_worker')
    try:
        win._on_clear_picks()
        win._apply_reflections(json.load(open(CONFIG))['crystal']['reflist_hkl'])
        sel = win._checked_ref_6d()
        assert sel.shape[0] > 0
        win._sync_ig()
        truth = win.ig.copy()
        rl, rl2 = sl.build_reflist_from_6d(sel)
        dms = sl.make_overlay_dms(np.array(rl), np.array(rl2), win._hkl,
                                  win._imdata, win._psirange, win._thrange,
                                  win._azir, win._psi, win._px, win._py, truth)
        win._imdata = _render(rf.ConeModel(dms), sl.extract_reduced(truth),
                              win._imdata.shape)

        win._peak_combo.setCurrentIndex(win._peak_combo.findData('ridge'))
        assert win._peak_method == 'ridge'
        # With ROIs left over from another test, the switch rebuilds on its own.
        _wait(win, app, '_build_worker')
        win._sliders['rotx'].setValue(truth[11] + 0.05)
        win._sliders['a'].setValue(truth[0] + 0.002)
        win._sync_ig()

        assert win._on_build_curves()
        _wait(win, app, '_build_worker')
        assert win._ridge_active()
        n_pts = len(win._ridge['row'])
        assert n_pts > 100
        assert win._fit_dms.objective is not None
        assert 'ridge: %d pts' % n_pts in win._lbl_resid.text()
        start = win._ridge_fitter.chi2(sl.extract_reduced(win.ig))

        win._active_method = 'LSQ'
        win._do_fit()
        _wait(win, app, '_fit_worker')
        _wait(win, app, '_build_worker')
        end = win._ridge_fitter.chi2(sl.extract_reduced(win.ig))
        assert end < 0.01 * start
        rms = float(np.sqrt(np.mean(win._ridge_fitter.pixel_residuals(
            sl.extract_reduced(win.ig)) ** 2)))
        assert rms < 0.1, rms
        assert abs(win.ig[0] - truth[0]) < 1e-4
        # The post-fit rebuild kept the reference points.
        assert len(win._ridge['row']) == n_pts
        assert win._ridge_scatter is not None and win._ridge_scatter.isVisible()
        txt = open(os.path.join(sorted(
            os.path.join('Processing', d) for d in os.listdir('Processing'))[-1],
            'Result.txt')).read()
        assert 'scored on %d ridge points' % n_pts in txt

        # Back to a ROI method: the objective goes with the rebuild.
        win._peak_combo.setCurrentIndex(win._peak_combo.findData('gauss'))
        _wait(win, app, '_build_worker')
        assert win._fit_dms.objective is None
        assert not win._ridge_active()
        assert not win._ridge_scatter.isVisible()
    finally:
        os.chdir(cwd)
        win._imdata, _, win._active_method, _ = saved
        win._peak_combo.setCurrentIndex(win._peak_combo.findData(saved[1]))
        _wait(win, app, '_build_worker')
        win._restore_from_dict(before)   # hand the next test the window it had
        _wait(win, app, '_build_worker')


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main([__file__, '-q']))
