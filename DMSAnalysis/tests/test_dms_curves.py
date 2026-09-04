"""The analytic-circle DMS curve method against the sampled theta sweep.

A DMS locus is a cone of exit directions, so it is exactly a circle in
exit-direction space; ``curve_method='circle'`` fits that circle and re-draws it
at detector resolution instead of joining the scanned points.  These tests pin
the three claims that makes:

* the sampled points really do lie on a circle, to machine precision, in both
  lattice modes and with a phason strain or a chi correction — *provided* the
  runs are cut where the scan vector reverses (a theta range spanning zero);
* the circle curves cover everything the sweep draws at the same scan
  resolution, so switching method can only add smoothness, never lose a line;
* a coarse circle run reproduces a fine sweep, which is the point of the option.

Run standalone:
    python -m DMSAnalysis.tests.test_dms_curves
or under pytest:
    pytest DMSAnalysis/tests/test_dms_curves.py
"""

import numpy as np

from .. import ts_quasi as ts


# ── engine setup shared by the tests ──────────────────────────────────────────

CUBIC = dict(lattice=[4.022, 4.022, 4.022, 90.0, 90.0, 90.0],
             hkl=[1.0, 1.0, 1.0], energy=12.0, azir=[1, -1, 0],
             bravais='cubic')

ICO_6D = np.array([
    [-1, -1, -2, -1,  1,  1], [-1,  1, -1, -2, -1,  1],
    [ 1, -1, -1,  1,  2,  1], [ 1,  2,  1, -1, -1,  1],
    [ 2,  1,  1,  1,  1,  1], [-1,  0, -2, -2,  0,  1],
    [ 0,  2,  0, -2, -1,  1], [ 2,  0,  0,  1,  2,  1],
    [ 2,  2,  1,  0,  0,  1], [ 0, -1, -2,  0,  2,  1]], dtype=float)

PHASON = [0.001228, 0.000730, 0.000491, 0.000507, -0.000951,
          -0.002741, -0.000441, -0.001405, 0.002354]


def cubic_reflist(n=40):
    """Enough of the depth-2 hkl set that some of it lands on the test detector
    (the first two dozen entries miss it entirely at this geometry)."""
    r = ts.hklgen_3d(2).astype(float)
    return r[np.any(r != 0, axis=1)][:n]


def make_engine(lattice, hkl, energy, azir, reflist, reflist2=None,
                bravais='cubic', psi=-180.0, thrange=(-27, 10), numsteps=200,
                psicor=0.0, chicor=0.0, thetacor=0.0, phason=None,
                method='sweep', shape=(1200, 1200), detdist=3000.0,
                detrot=(0.0, 0.0, 0.0)):
    """A `dmsfit_ico_hkl` in calculator mode, as the slider's overlay builds it.

    `detrot` matters more than it looks: an untilted plate catches none of the
    icosahedral fixture's exit beams, and the few pixels that survive the bounds
    filter all land on the beam centre — which would make the curve comparisons
    below compare a few hundred copies of one point and pass on anything.
    """
    thb = ts.bragg(lattice, hkl, energy).th()[0]
    reflist  = np.asarray(reflist, dtype=float)
    reflist2 = np.zeros_like(reflist) if reflist2 is None else np.asarray(reflist2, float)
    ig = np.zeros(24)
    ig[:6] = lattice
    ig[6], ig[7], ig[8] = psicor, chicor, thetacor
    ig[10:14] = [detdist, detrot[0], detrot[1], detrot[2]]
    ig[14] = energy
    ig[15:24] = [0.0] * 9 if phason is None else phason
    dms = ts.dmsfit_ico_hkl(
        np.matrix(reflist), [thb + thrange[0], thb + thrange[1], numsteps],
        np.round(hkl), [psi - 180, psi + 180], 45,
        np.zeros((1, 1)), np.zeros((1, 1, 1)), hkl,
        np.matrix([[1, 0, 0], [0, 0, 1]]), np.zeros(shape), 0.0, azir, psi,
        600, 600, 0, bravais, True, False,
        ig[10], ig[11], ig[12], ig[13], ig[14],
        np.matrix(reflist2), list(ig[15:24]), ig[0])
    dms.setLattice(list(ig[:6]))
    dms.setIGFull(ig)
    dms.setCurveMethod(method)
    if bravais in ts.CONVENTIONAL_SYSTEMS:
        reduced = ig[ts.reduced_param_indices(bravais, True, False)]
    else:
        # Slots [0, 6,7,8,9, 10..13, 15..23] — the icosahedral branch of imcalc
        # reads detdist from inputs[5], so slot 9 (unused, always zero) has to be
        # in the vector to keep everything after it aligned.  Dropping it fed the
        # engine detrot as its detector distance.
        reduced = np.concatenate([[ig[0]], ig[6:10], ig[10:14], ig[15:24]])
    dms.imcalc(reduced)
    return dms


def ico_kwargs(**over):
    rl, rl2 = ts.Projection6d(ICO_6D).reflection_6d()
    aq = 6.461053
    kw = dict(lattice=[aq, aq, aq, 90.0, 90.0, 90.0],
              hkl=[2.27931876, 3.70249186, 1.29579814], energy=9.0,
              azir=[0, 0, 1], reflist=np.asarray(rl), reflist2=np.asarray(rl2),
              bravais='icosahedral', detrot=(40.0, 0.0, 0.0))
    kw.update(over)
    return kw


def cubic_kwargs(**over):
    kw = dict(CUBIC, reflist=cubic_reflist())
    kw.update(over)
    return kw


def points(line):
    """The finite (x, y) points of one dmslines entry."""
    x = np.asarray(line[0], dtype=float)
    y = np.asarray(line[1], dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return np.stack([x[m], y[m]], axis=1)


def distinct_pixels(dms):
    """How many distinct pixels the drawn curves actually occupy.  A comparison
    over a curve that has collapsed onto one point proves nothing."""
    drawn = [points(l) for l in dms.dmslines]
    drawn = [p for p in drawn if len(p)]
    if not drawn:
        return 0
    return len(np.unique(np.concatenate(drawn), axis=0))


def covered_within(A, B):
    """Worst distance from a point of A to the nearest point of B (pixels)."""
    if len(A) == 0:
        return 0.0
    if len(B) == 0:
        return np.inf
    worst = 0.0
    for chunk in np.array_split(A, max(1, len(A) // 2000 + 1)):
        d = np.sqrt(((chunk[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
        worst = max(worst, float(d.max()))
    return worst


# ── the circle is exact ───────────────────────────────────────────────────────

def _worst_circle_deviation(**kw):
    dms = make_engine(method='circle', **kw)
    return dms.circle_residual


def test_locus_is_a_circle_to_machine_precision():
    """Both lattice modes, at psi, with a phason strain and with a chi
    correction.  ~1e-13 is float64 noise on a unit vector."""
    for tag, kw in [
            ('cubic',          cubic_kwargs()),
            ('cubic psi',      cubic_kwargs(psi=37.0)),
            ('cubic psicor',   cubic_kwargs(psicor=2.5)),
            ('cubic chicor',   cubic_kwargs(chicor=0.05)),
            ('ico',            ico_kwargs()),
            ('ico phason',     ico_kwargs(phason=PHASON)),
            ('ico psi',        ico_kwargs(psi=20.0))]:
        dev = _worst_circle_deviation(**kw)
        # > 0 as well as small: a run that was never fitted reports 0.0, which
        # would let this pass on an empty scene.
        assert dev is not None and 0 < dev < 1e-9, \
            '%s: circle deviation %g' % (tag, dev)


def test_theta_range_spanning_zero_is_still_exact():
    """The cubic case above has thb = 12.85 deg, so the default [-27, +10] window
    crosses theta = 0 and the scan vector reverses mid-sweep.  Cutting the runs
    there is what keeps the fit exact — without it the deviation is ~0.2 rad."""
    thb = ts.bragg(CUBIC['lattice'], CUBIC['hkl'], CUBIC['energy']).th()[0]
    assert thb < 27.0, 'test premise: the default theta window must span zero'
    assert 0 < _worst_circle_deviation(**cubic_kwargs(thrange=(-27, 10))) < 1e-9
    # ... and the same range that does not cross zero agrees.
    assert 0 < _worst_circle_deviation(**cubic_kwargs(thrange=(-10, 10))) < 1e-9


def test_theta_correction_shears_the_locus_and_is_reported():
    """A theta correction offsets the exit polar angle after the azimuth was
    solved, so the locus is no longer exactly planar.  It stays small (~2e-4 rad
    at 1 deg, sub-pixel at a 3000 px detector distance) and, crucially, is
    reported rather than hidden."""
    dev = _worst_circle_deviation(**cubic_kwargs(thetacor=1.0))
    assert 1e-9 < dev < 1e-2


# ── the circle curves reproduce the sweep ─────────────────────────────────────

def test_circles_cover_the_sweep_at_the_same_resolution():
    """Every point the sampled sweep draws is on a circle curve, to within the
    engine's whole-pixel rounding (sqrt(2) = one diagonal pixel).  Switching
    method can add resolution but must never lose a curve."""
    for tag, kw in [('cubic', cubic_kwargs()), ('ico', ico_kwargs())]:
        for numsteps in (80, 400):
            swept   = make_engine(method='sweep',  numsteps=numsteps, **kw)
            circled = make_engine(method='circle', numsteps=numsteps, **kw)
            assert len(swept.dmslines) == len(circled.dmslines)
            drawn = 0
            for j, (ls, lc) in enumerate(zip(swept.dmslines, circled.dmslines)):
                S, C = points(ls), points(lc)
                if len(S) == 0:
                    continue
                drawn += 1
                d = covered_within(S, C)
                assert d <= 1.5, ('%s n=%d ref %d: sweep point %.2f px from any '
                                  'circle point' % (tag, numsteps, j, d))
            assert drawn > 0, '%s n=%d: nothing on the detector to compare' % (
                tag, numsteps)
            spread = distinct_pixels(swept)
            assert spread > 50, ('%s n=%d: the sweep occupies only %d distinct '
                                 'pixels — nothing meaningful to compare'
                                 % (tag, numsteps, spread))


def test_coarse_circles_reproduce_a_fine_sweep():
    """The point of the option: a scan an order of magnitude coarser still puts
    the curve where a fine sweep puts it."""
    for tag, kw in [('cubic', cubic_kwargs()), ('ico', ico_kwargs())]:
        fine   = make_engine(method='sweep',  numsteps=4000, **kw)
        coarse = make_engine(method='circle', numsteps=100,  **kw)
        drawn = 0
        for j, (lf, lc) in enumerate(zip(fine.dmslines, coarse.dmslines)):
            C = points(lc)
            if len(C) == 0:
                continue
            drawn += 1
            d = covered_within(C, points(lf))
            assert d <= 8.0, ('%s ref %d: coarse circle point %.2f px off the '
                              'fine sweep' % (tag, j, d))
        assert drawn > 0, '%s: nothing on the detector to compare' % tag
        assert distinct_pixels(fine) > 50, (
            '%s: the reference sweep occupies too few distinct pixels' % tag)


def test_simulated_image_follows_the_curves():
    """The simulated image is drawn from the same points as the curves, so the
    circle method fills in the line rather than leaving the sweep's gaps."""
    kw = cubic_kwargs(numsteps=100)
    swept   = make_engine(method='sweep',  **kw)
    circled = make_engine(method='circle', **kw)
    assert np.count_nonzero(circled.imsim) > np.count_nonzero(swept.imsim)


# ── the plumbing ──────────────────────────────────────────────────────────────

def test_curve_method_names():
    assert ts.dms_curve_method('circle') == 'circle'
    assert ts.dms_curve_method('Circles') == 'circle'
    assert ts.dms_curve_method('analytic') == 'circle'
    assert ts.dms_curve_method('sweep') == 'sweep'
    # a missing / blank config value is the default, not an error
    assert ts.dms_curve_method(None) == 'sweep'
    assert ts.dms_curve_method('') == 'sweep'
    for bad in ('spline', 'polyline', 'ellipse'):
        try:
            ts.dms_curve_method(bad)
        except ValueError:
            continue
        raise AssertionError('accepted an unknown curve method %r' % bad)


def test_sweep_is_the_default_and_reports_no_residual():
    dms = make_engine(**cubic_kwargs(numsteps=100))
    assert dms.curvemethod == 'sweep'
    assert dms.circle_residual is None


def test_plane_basis_is_perpendicular_and_deterministic():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(50, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    u = ts.dms_plane_basis(n)
    assert np.allclose(np.einsum('ij,ij->i', u, n), 0.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
    assert np.array_equal(u, ts.dms_plane_basis(n))
    # a normal along x falls back to the second rule rather than dividing by zero
    ux = ts.dms_plane_basis([[1.0, 0.0, 0.0]])
    assert np.isfinite(ux).all() and np.allclose(np.linalg.norm(ux), 1.0)


def test_arc_points_round_trip_a_known_circle():
    c = np.array([[0.1, -0.2, 0.3]])
    n = np.array([[0.0, 0.0, 1.0]])
    r = np.array([0.4])
    pts, counts = ts.dms_arc_points(c, r, n, np.array([0.0]), np.array([1.0]),
                                    np.array([64]))
    assert counts[0] == 64 and len(pts) == 64
    assert np.allclose(np.linalg.norm(pts - c, axis=1), r[0])
    assert np.allclose((pts - c) @ n[0], 0.0, atol=1e-12)
    # and the fit recovers what was drawn
    fc, fr, fn, a0, a1, resid = ts.dms_fit_arcs(pts, [np.arange(64)])
    assert np.allclose(fc[0], c[0], atol=1e-9)
    assert abs(fr[0] - r[0]) < 1e-9
    assert abs(abs(a1[0] - a0[0]) - 1.0) < 1e-9
    assert resid[0] < 1e-9


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
