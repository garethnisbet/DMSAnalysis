"""The triple-intersection residual must be a property of the lattice, not of
how finely the Kossel lines happen to be sampled.

A Kossel line is a cone of exit directions — exactly a circle on the sphere —
and `kosscalc`'s 0..360 sweep only samples it.  Crossing the sampled polylines
crosses chords, so the crossing is off by the chord sagitta and `triple_spread`
(a squared distance) by its square; the residual then depended on
`computation.resolution`, and an optimiser could reach machine zero by tuning
the cell to the polygon rather than closing the triple.  The engine now solves
each pair as a cone-cone intersection (`sphere_circle_plane`,
`cone_pair_directions`), so the same geometry scores the same number at any
resolution — which is also why tripslider's live label (live resolution) and
its status bar (fit resolution) used to disagree.

Run standalone:
    python -m DMSAnalysis.tests.test_tripfit_intercepts
or under pytest.
"""

import copy
import json
import os

import numpy as np

from .. import ts_quasi as ts
from .test_tripfit_quasi import CONFIG, PMN_CONFIG, _cfg, _params, _window

RESOLUTIONS = (50, 100, 400, 1000, 4000)


def _engine(cfg, group, resolution):
    geo, gc = cfg['geometry'], cfg['intersections'][group]
    return ts.tripfit(np.array([geo['hkl']], dtype=float),
                      np.array(gc['reflist'], dtype=float), geo['azir'],
                      resolution, cfg['computation']['bravais'],
                      gc['energy'], float(gc.get('target', 0.0)),
                      params=_params(cfg),
                      tau=float(cfg['crystal'].get('tau_approx', ts.TAU_APPROX)))


def _reduced(cfg):
    return _params(cfg)[ts.tripfit_free_slots(cfg['computation']['bravais'])]


def _unproject(pt):
    '''Inverse stereographic projection: the unit direction a plane point came
    from, so a crossing can be tested against the cones it is meant to solve.'''
    x, y = float(pt[0]), float(pt[1])
    r2 = x * x + y * y
    return np.array([2 * x, 2 * y, r2 - 1.0]) / (r2 + 1.0)


# ── the helpers ─────────────────────────────────────────────────────────────────
def test_circle_plane_is_exact_and_sign_agnostic():
    n0 = np.array([0.3, -0.5, 0.81])
    n0 /= np.linalg.norm(n0)
    c0 = 0.42
    u = np.cross(n0, [1.0, 0.0, 0.0]); u /= np.linalg.norm(u)
    w = np.cross(n0, u)
    r = np.sqrt(1 - c0 ** 2)
    for npts in (3, 17, 500):                     # three points already fix it
        t = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        v = c0 * n0 + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)
        n, c, resid = ts.sphere_circle_plane(v)
        assert resid < 1e-14
        s = np.sign(n @ n0)                       # the sign of (n, c) is free
        assert np.allclose(s * n, n0, atol=1e-12)
        assert abs(s * c - c0) < 1e-12


def test_circle_plane_rejects_points_that_fix_no_circle():
    for bad in (np.zeros((5, 3)),                        # coincident
                np.column_stack((np.linspace(0, 1, 9),   # collinear
                                 np.zeros(9), np.zeros(9)))):
        try:
            ts.sphere_circle_plane(bad)
        except ValueError:
            continue
        raise AssertionError('a degenerate locus was accepted as a circle')


def test_cone_pair_solves_both_plane_conditions():
    n1 = np.array([0.0, 0.0, 1.0])
    n2 = np.array([0.6, 0.0, 0.8])
    for c1, c2 in ((0.3, 0.1), (-0.2, 0.1), (0.0, 0.0)):
        v = ts.cone_pair_directions(n1, c1, n2, c2)
        assert len(v) == 2
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
        assert np.allclose(v @ n1, c1) and np.allclose(v @ n2, c2)


def test_cone_pair_reports_no_crossing_and_refuses_coaxial_cones():
    n1 = np.array([0.0, 0.0, 1.0])
    n2 = np.array([0.0, 0.6, 0.8])
    # cones whose half-angles differ by more than their axes do: no crossing
    assert len(ts.cone_pair_directions(n1, 0.99, n2, -0.99)) == 0
    assert len(ts.cone_pair_directions(n1, -0.2, np.array([0.6, 0.0, 0.8]),
                                       0.55)) == 0
    try:
        ts.cone_pair_directions(n1, 0.3, n1, 0.4)
    except ValueError:
        return
    raise AssertionError('coaxial cones were given an isolated crossing')


# ── the engine ──────────────────────────────────────────────────────────────────
def test_residual_is_the_same_at_every_resolution():
    for path in (PMN_CONFIG, CONFIG):
        cfg = _cfg(path)
        x = _reduced(cfg)
        for gi in range(len(cfg['intersections'])):
            vals = []
            for res in RESOLUTIONS:
                tf = _engine(cfg, gi, res)
                vals.append(tf.fit(x))
                # the loci really are circles, which is what makes this exact
                assert tf.circle_residual is not None
                assert tf.circle_residual < ts.KOSSEL_CIRCLE_TOL
            vals = np.array(vals)
            assert np.all(vals > 0)
            spread = abs(vals - vals[0]).max() / vals[0]
            assert spread < 1e-8, (path, gi, vals)


def test_crossings_lie_on_the_two_cones_they_solve():
    for path in (PMN_CONFIG, CONFIG):
        cfg = _cfg(path)
        tf = _engine(cfg, 0, 200)
        tf.kosselcalc(_reduced(cfg))
        planes = [ts.sphere_circle_plane(np.asarray(vr))[:2]
                  for vr in (tf.vr0, tf.vr1, tf.vr2)]
        pts = tf._intercepts()
        for pt, (i, j) in zip(pts, ((0, 1), (0, 2), (1, 2))):
            v = _unproject(pt)
            for k in (i, j):
                n, c = planes[k]
                assert abs(v @ n - c) < 1e-12, (path, i, j, k)


def test_polyline_crossings_converge_to_the_analytic_ones():
    '''The old sampled-polyline path is still the fallback, so it must agree —
    in the limit.  Its error shrinks with resolution while the analytic value
    does not move, which is the whole point of the change.'''
    cfg = _cfg(PMN_CONFIG)
    x = _reduced(cfg)
    errs = []
    for res in (100, 400, 1600):
        tf = _engine(cfg, 1, res)
        tf.kosselcalc(x)
        exact = ts.triple_spread(tf._intercepts())
        P = tf._polyline_intercepts()
        best = min(ts.triple_spread((a, b, c))
                   for a in P[0] for b in P[1] for c in P[2])
        errs.append(abs(best - exact) / exact)
    # ~1/res^2, so 16x resolution is ~3 orders: it converges, and on the
    # analytic value — which meanwhile has not moved at all.
    assert errs[0] > 100 * errs[-1], errs
    assert errs[-1] < 0.1, errs


def test_a_locus_that_is_not_a_circle_falls_back_to_the_polylines():
    cfg = _cfg(PMN_CONFIG)
    tf = _engine(cfg, 0, 400)
    tf.kosselcalc(_reduced(cfg))
    assert tf._circle_intercepts() is not None
    # bend one locus off its plane by far more than KOSSEL_CIRCLE_TOL
    bent = np.asarray(tf.vr1).copy()
    bent[::3, 2] += 1e-6
    tf.vr1 = np.matrix(bent / np.linalg.norm(bent, axis=1, keepdims=True))
    assert tf._circle_intercepts() is None
    assert tf.circle_residual is None      # the marker every fallback leaves
    pts = tf._intercepts()                      # still answers, from the sweep
    assert pts.shape == (3, 2) and np.all(np.isfinite(pts))


# ── the GUI ─────────────────────────────────────────────────────────────────────
def test_gui_live_and_fit_residuals_agree():
    '''The reported symptom: tripslider's control-panel sum is computed at the
    live resolution and its status bar at the fit resolution.  Those are now
    the same number.'''
    cfg = _cfg(PMN_CONFIG)
    cfg['computation']['live_resolution'] = 120
    cfg['computation']['resolution'] = 1500
    win = _window(cfg)
    try:
        win._redraw()
        live = float(win._total_lbl.text().split(':')[1].split('(')[0])
        fitres = win._fit_objective(win._reduced())      # at the fit resolution
        assert live > 0
        # the label is written to 4 decimals, so that is all this can ask of it
        # — and it is still 12 orders tighter than the disagreement it guards
        assert abs(live - fitres) <= 1e-4 * fitres, (live, fitres)
    finally:
        win.close()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  ', name)
