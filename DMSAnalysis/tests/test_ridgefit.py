"""The ridge + cone fit (`ridgefit`): measure the lines, fit the cones.

Pins what the method rests on:

* the unrounded detector map and its inverse are exact, and agree with the
  engine's own overlay curves;
* the residual is smooth in the parameters.  Two bugs made it jump: a cone's
  "outside" was defined by a normal that could flip, and a point lying between
  two nearly coincident cones of one reflection (the two halves of a theta
  range spanning zero) tied between them.  Either flips a residual's sign under
  a 1e-12 perturbation and stalls the optimiser partway along a valley;
* the ridge extraction finds a sub-pixel offset and ignores module gaps;
* end to end, a synthetic image rendered at known parameters is fitted back to
  them, from a perturbed start, within the reported uncertainties.

Run standalone:
    python -m DMSAnalysis.tests.test_ridgefit
or under pytest:
    pytest DMSAnalysis/tests/test_ridgefit.py
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .. import ridgefit as rf
from .. import ts_quasi as ts
from .test_dms_curves import make_engine, cubic_kwargs

SHAPE = (1200, 1200)


def _case():
    dms = make_engine(**cubic_kwargs(), method='circle', shape=SHAPE)
    slots = ts.imcalc_param_indices('cubic', True, False)
    names = [rf.IG_SLOT_NAMES[i] for i in slots]
    return dms, rf.ConeModel(dms), dms.inputarray[slots].copy(), names


def _render(model, x, amp=200.0, bg=50.0, sig=2.0, seed=0):
    """Poisson image of Gaussian-profile lines along the exact cones at `x`.
    Bilinear splatting of a dense sampling keeps each line centred sub-pixel."""
    im = np.zeros(SHAPE)
    for _, rr, cc in model.predicted_lines(x, SHAPE, spacing=0.25):
        ok = np.isfinite(rr)
        r, c = rr[ok], cc[ok]
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (1, 0, fr * (1 - fc)),
                          (0, 1, (1 - fr) * fc), (1, 1, fr * fc)):
            np.add.at(im, (np.clip(r0 + dr, 0, SHAPE[0] - 1),
                           np.clip(c0 + dc, 0, SHAPE[1] - 1)), w * 0.25)
    im = ndimage.gaussian_filter(im, sig) * amp * np.sqrt(2 * np.pi) * sig + bg
    return np.random.default_rng(seed).poisson(im).astype(float)


def test_backprojection_inverts_the_detector_map():
    dms, m, x, _ = _case()
    g, _ = m.circles(x)
    V = g.vecs[np.isfinite(g.vecs).all(1)]
    r, c = m.project(g, V)
    on = (r >= 0) & (r < SHAPE[0]) & (c >= 0) & (c < SHAPE[1])
    assert on.sum() > 100
    U = m.backproject(g, r[on], c[on])
    assert np.abs(U - V[on]).max() < 1e-12


def test_predicted_lines_match_the_engine_overlay():
    """Same curves as the overlay (unrounded; the slack is the arc ends and
    the reference polyline's own sampling)."""
    dms, m, x, _ = _case()
    by = {}
    for o, rr, cc in m.predicted_lines(x, SHAPE):
        ok = np.isfinite(rr)
        by.setdefault(o, []).append(np.stack([rr[ok], cc[ok]], 1))
    d_all = []
    for j, (ex, ey) in enumerate(dms.dmslines):
        ok = np.isfinite(ex)
        if ok.any() and j in by:
            d, _ = cKDTree(np.concatenate(by[j])).query(np.stack([ey[ok], ex[ok]], 1))
            d_all.append(d)
    d_all = np.concatenate(d_all)
    assert len(d_all) > 1000
    assert np.percentile(d_all, 99.5) <= 1.0
    assert d_all.max() < 4.0          # a few arc-end points


def test_residual_is_smooth_under_tiny_perturbations():
    dms, m, x, names = _case()
    img = _render(m, x)
    ridge = rf.extract_ridge(img, m.predicted_lines(x, SHAPE), 45.0)
    f = rf.RidgeConeFit(m, ridge)
    r0 = f.pixel_residuals(x)
    for nm in ('chicor', 'dyrot', 'a', 'psicor'):
        d = np.zeros_like(x)
        d[names.index(nm)] = 1.0
        for h in (1e-12, 1e-9):
            dr = f.pixel_residuals(x + h * d) - r0
            assert np.abs(dr).max() < 1e-4, (nm, h, np.abs(dr).max())


def test_ridge_finds_a_subpixel_offset_and_skips_gaps():
    """A straight line 0.37 px off its prediction, with a module gap across it."""
    img = np.full((200, 300), 100.0)
    rows = np.arange(200)[:, None]
    img += 400 * np.exp(-0.5 * ((rows - 100.37) / 1.8) ** 2)
    img[:, 140:150] = -1                                    # module gap
    cc = np.linspace(5, 295, 600)
    lines = [(0, np.full_like(cc, 100.0), cc)]
    R = rf.extract_ridge(img, lines, 30.0, step=8.0)
    assert len(R['row']) > 20
    assert np.all(np.abs(R['col'] - 145) > 4), 'sampled across the gap'
    assert abs(np.median(R['row']) - 100.37) < 0.02
    assert np.all(np.abs(R['row'] - 100.37) < 5 * R['err'] + 0.02)


def test_synthetic_fit_recovers_the_geometry():
    dms, m, xt, names = _case()
    img = _render(m, xt)
    x0 = xt.copy()
    for nm, v in (('a', 0.002), ('psicor', 0.05), ('detdist', 10.0),
                  ('dxrot', 0.05), ('dzrot', 0.05), ('thcor', 0.02)):
        x0[names.index(nm)] += v
    x = x0
    for _ in range(2):
        ridge = rf.extract_ridge(img, m.predicted_lines(x, SHAPE), 45.0)
        f = rf.RidgeConeFit(m, ridge)
        res = f.run(x)
        x = res.x
    assert f.chi2(x) <= f.chi2(xt) + 1e-6, 'stopped short of the minimum'
    err, _ = rf.parameter_errors(res, len(ridge['row']))
    z = np.abs(x - xt) / err
    assert np.all(z < 4), dict(zip(names, np.round(z, 2)))
    assert abs(x[names.index('a')] - xt[names.index('a')]) < 1e-4


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main([__file__, '-q']))
