"""The DMS overlay is drawn on the cones the ridge objective scores.

The overlay lines (`dmsfit_ico_hkl.dmslines`) used to be rounded to whole
pixels before the beam centre was added, so even the analytic circle method
drew every point on a pixel vertex, up to ~0.7 px off the cone that
`ridgefit.ConeModel.project` (the same map, unrounded) scores the ridge points
against.  The lines are now unrounded; only the simulated image's pixel index
is.  Pinned here, for both curve methods and both lattice modes:

* every drawn point back-projects onto its own reflection's cone to machine
  precision;
* the points are sub-pixel, not snapped to the grid;
* the simulated image is still indexed on whole pixels.

Run standalone:
    python -m DMSAnalysis.tests.test_overlay_subpixel
or under pytest:
    pytest DMSAnalysis/tests/test_overlay_subpixel.py
"""

import numpy as np

from .. import ridgefit as rf
from .. import ts_quasi as ts
from .test_dms_curves import make_engine, cubic_kwargs, ico_kwargs

SHAPE = (1200, 1200)
CASES = [('cubic', cubic_kwargs, 'cubic'), ('ico', ico_kwargs, 'icosahedral')]
RUNS = [('circle', 100), ('sweep', 1000)]


def _engine(kwf, bravais, method, numsteps):
    dms = make_engine(method=method, numsteps=numsteps, shape=SHAPE, **kwf())
    x = dms.inputarray[ts.imcalc_param_indices(bravais, True, False)].copy()
    return dms, rf.ConeModel(dms), x


def _drawn(dms):
    """(reflection, rows, cols) for every reflection with points on the plate."""
    for j, (ex, ey) in enumerate(dms.dmslines):
        ex, ey = np.asarray(ex, float), np.asarray(ey, float)
        ok = np.isfinite(ex) & np.isfinite(ey)
        if ok.any():
            yield j, ey[ok], ex[ok]


def _cone_distance(U, C, owner):
    """Angular distance (rad) of unit vectors U to the nearest of `owner`'s
    cones.  A cone is the circle {centre c + radius r, normal n} on the sphere."""
    best = np.full(len(U), np.inf)
    for k in np.where(C['owner'] == owner)[0]:
        w = U - C['c'][k]
        h = w @ C['n'][k]
        rho = np.linalg.norm(w - np.outer(h, C['n'][k]), axis=1)
        best = np.minimum(best, np.hypot(h, rho - C['r'][k]))
    return best


def test_overlay_points_lie_on_the_scored_cones():
    for tag, kwf, bravais in CASES:
        for method, numsteps in RUNS:
            dms, model, x = _engine(kwf, bravais, method, numsteps)
            g, C = model.circles(x)
            assert C is not None
            n_pts, worst = 0, 0.0
            for j, rows, cols in _drawn(dms):
                d = _cone_distance(model.backproject(g, rows, cols), C, j)
                assert np.isfinite(d).all(), (
                    '%s %s ref %d: drawn line has no cone in the model'
                    % (tag, method, j))
                n_pts += len(d)
                worst = max(worst, float(d.max()))
            assert n_pts > 100, '%s %s: too little on the plate' % (tag, method)
            # 1e-9 rad is ~3e-6 px at the fixture's 3000 px detector distance;
            # the old whole-pixel rounding put points up to ~2e-4 rad off.
            assert worst < 1e-9, ('%s %s: drawn point %.2e rad off its cone'
                                  % (tag, method, worst))


def test_overlay_points_are_not_snapped_to_the_pixel_grid():
    for tag, kwf, bravais in CASES:
        for method, numsteps in RUNS:
            dms, _, _ = _engine(kwf, bravais, method, numsteps)
            P = np.concatenate([np.stack([r, c], 1) for _, r, c in _drawn(dms)])
            off_grid = np.mean(np.abs(P - np.round(P)).max(axis=1) > 1e-3)
            assert off_grid > 0.9, ('%s %s: only %.0f%% of drawn points are '
                                    'off a pixel vertex' % (tag, method,
                                                            100 * off_grid))


def test_simulated_image_is_still_on_whole_pixels():
    for tag, kwf, bravais in CASES:
        for method, numsteps in RUNS:
            dms, _, _ = _engine(kwf, bravais, method, numsteps)
            r, c = dms.dmsindex
            assert len(r) > 0
            assert np.issubdtype(np.asarray(r).dtype, np.integer)
            assert np.issubdtype(np.asarray(c).dtype, np.integer)
            assert np.count_nonzero(dms.imsim) > 0


def test_slider_dms_scatter_is_drawn_from_the_unrounded_lines():
    """The slider's DMS-lines scatter used to be drawn from `dmsindex` (the
    simulated image's whole-pixel raster), so its dots stayed on pixel vertices
    after the lines themselves were unrounded.  Driven through the real GUI."""
    import os
    import time
    from ..qt import QtWidgets
    from .gui_harness import slider_on
    config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'configs',
                          'fit_conventional_tetragonal_PMN_PT_example.json')
    sl = slider_on(config)
    win = sl.win
    app = QtWidgets.QApplication.instance()
    win._dms_scatter.setData(x=[], y=[])
    win._do_update()
    deadline = time.time() + 30
    x = np.array([])
    while time.time() < deadline:
        win._worker.idle.wait(0.1)
        app.processEvents()
        x, y = win._dms_scatter.getData()
        if x is not None and len(x):
            break
    assert x is not None and len(x) > 100, 'the overlay update never landed'
    P = np.stack([np.asarray(y, float), np.asarray(x, float)], 1)
    off_grid = np.mean(np.abs(P - np.round(P)).max(axis=1) > 1e-3)
    assert off_grid > 0.9, ('only %.0f%% of the DMS-lines dots are off a pixel '
                            'vertex' % (100 * off_grid))
    # and they are the engine's lines, point for point
    rows, cols = sl.discovery_points(win._dms.dmslines)
    assert len(rows) == len(P)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok ', name)
