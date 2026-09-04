"""How ``roibuilder_ico_hkl`` turns one reflection's locus into its two ROIs.

The builder gets the locus as a pixel path in scan order and has to end up with
two kernel planes, each a strip along half of *one* line.  The path it is handed
is not always tidy: the engine walks both psi solutions and concatenates them
(often the same detector line twice), and a locus can leave the physical region
or the plate and come back, so the raw index can hold several far-apart pieces.

These tests pin the three steps that handle that — dedupe, cut on the gaps, and
rasterise — because getting them wrong is not visible in the fit, only in the
ROI curves it is scored on.

Run standalone:
    python -m DMSAnalysis.tests.test_roi_kernels
or under pytest:
    pytest DMSAnalysis/tests/test_roi_kernels.py
"""

import numpy as np

from .. import ts_quasi as ts


def _arc(n=200, r0=300.0, c0=400.0, rad=250.0, a0=0.2, a1=1.4):
    """A sampled arc as (N,2) integer pixels, in path order."""
    a = np.linspace(a0, a1, n)
    return np.stack([r0 + rad * np.sin(a), c0 + rad * np.cos(a)], axis=1).astype(int)


def test_a_doubled_psi_branch_collapses_to_one_line():
    """Both psi solutions can trace the same detector line, so the raw index is
    that line twice.  Deduped it is one path, in the original order."""
    p = _arc()
    doubled = np.concatenate([p, p])
    out = ts.roi_dedupe_path(doubled)
    uniq = {tuple(v) for v in p}
    assert len(out) == len(uniq)
    assert {tuple(v) for v in out} == uniq
    # first-seen order preserved: consecutive entries stay neighbours
    d = np.linalg.norm(np.diff(out.astype(float), axis=0), axis=1)
    assert d.max() < 4, 'dedupe scrambled the path order'


def test_runs_are_cut_at_the_gaps_and_ordered_longest_first():
    """A locus that leaves the detector and comes back is several pieces; the
    builder must see them as separate, longest first."""
    long_piece = _arc(n=200)
    far_piece = _arc(n=60) + np.array([900, -300])
    runs = ts.roi_split_runs(np.concatenate([long_piece, far_piece]))
    assert len(runs) == 2
    assert len(runs[0]) == 200 and len(runs[1]) == 60
    assert list(runs[0]) == list(range(200))


def test_a_short_stub_is_not_a_run():
    """Two or three stray pixels are not a piece of line worth a ROI."""
    p = np.concatenate([_arc(n=120), _arc(n=2) + np.array([800, 800])])
    runs = ts.roi_split_runs(p)
    assert len(runs) == 1 and len(runs[0]) == 120


def test_rasterise_fills_between_samples_and_stays_on_the_plate():
    """Scan samples are pixels apart; the kernel must be a continuous path, and
    nothing outside the image may be written."""
    from scipy import ndimage
    shape = (700, 800)
    p = _arc(n=40)                      # deliberately coarse: ~10 px between points
    im = ts.roi_rasterise(p, shape)
    _, ncomp = ndimage.label(im > 0, structure=np.ones((3, 3)))
    assert ncomp == 1, 'rasterised path is in %d pieces' % ncomp
    # and walking it never has to jump
    step = np.linalg.norm(np.diff(ts.roi_walk_path(np.argwhere(im > 0)).astype(float),
                                  axis=0), axis=1).max()
    assert step <= np.sqrt(2) + 1e-9, 'rasterised path has holes (max step %g)' % step
    # A path running off the plate is clipped, not crashed, and keeps the part
    # that is on it.
    im2 = ts.roi_rasterise(_arc(n=40) + np.array([200, 0]), shape)
    assert im2.shape == shape and 0 < im2.sum() < im.sum()


def test_the_halves_never_span_a_gap():
    """The property the whole thing exists for: with the locus in two pieces,
    neither ROI may bridge them.  Cutting the *sorted* index at its median (the
    old behaviour) does exactly that — the assertion below fails on it — and the
    bridge is integrated as if it were line."""
    # The two pieces overlap in row but sit 900 px apart in column — the case a
    # detector-axis sort interleaves.
    a = _arc(n=200)
    b = _arc(n=180) + np.array([0, 900])
    idx = np.concatenate([a, b])
    run = ts.roi_split_runs(ts.roi_dedupe_path(idx))[0]
    pts = ts.roi_dedupe_path(idx)[run]
    half = len(pts) // 2
    for part in (pts[:half], pts[half:]):
        d = np.linalg.norm(np.diff(part.astype(float), axis=0), axis=1)
        assert d.max() < 10, 'a half spans a %.0f px jump' % d.max()
    # and the old recipe really would have bridged it
    srt = idx[np.argsort(idx[:, 0])]
    d_old = np.linalg.norm(np.diff(srt[:len(srt) // 2].astype(float), axis=0), axis=1)
    assert d_old.max() > 100


def _seg_dist(P, A, B):
    """Distance from each point in P (N,2) to the nearest segment A->B (M,2)."""
    d = B - A
    L = np.maximum((d * d).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * d[None]).sum(2) / L[None], 0, 1)
    proj = A[None] + t[:, :, None] * d[None]
    return np.linalg.norm(P[:, None, :] - proj, axis=2).min(1)


def test_the_roi_engine_and_the_fit_engine_find_the_same_lines():
    """The ROI builder runs `dmscalc_ico_hkl`; the overlay and the fit run
    `dmsfit_ico_hkl`.  They must agree on where a reflection's line is, or the
    ROI is laid along a line that is not the one being fitted.

    They did not: the fit engine drops the theta steps with no physical Ewald
    solution (|sin| > 1, or a negative discriminant), while the ROI engine
    clamped both and turned every one of them into a solution.  The invented
    points make a smooth curve of their own somewhere else on the plate, so a
    ROI could end up hundreds of pixels from its line — on the TestExample
    session, [-1 1 1] drew a vertical arc and got a horizontal ROI 835 px away.
    """
    from .test_dms_curves import cubic_kwargs, ico_kwargs, make_engine

    common = dict(numsteps=400, thrange=(-27, 10), psi=-180.0, detdist=3000.0,
                  shape=(1200, 1200))
    fixtures = [cubic_kwargs(detrot=(0.0, 0.0, 0.0), **common),
                ico_kwargs(**common)]          # ico_kwargs sets its own detrot
    for kw in fixtures:
        reflist = np.asarray(kw['reflist'], dtype=float)
        reflist2 = np.asarray(kw['reflist2'], float) if kw.get('reflist2') is not None \
            else np.zeros_like(reflist)
        thb = ts.bragg(kw['lattice'], kw['hkl'], kw['energy']).th()[0]
        hkllist = ts.pilkhlrange(kw['lattice'], kw['hkl'], kw['energy'],
                                 thb + kw['thrange'][0],
                                 thb + kw['thrange'][1]).hklscan(kw['numsteps'])
        # Exactly the geometry the fit engine is given, or the two are not
        # comparable and the test measures its own inconsistency.
        ig = np.zeros(24)
        ig[:6] = kw['lattice']
        ig[10:14] = [kw['detdist']] + list(kw['detrot'])
        ig[14] = kw['energy']

        checked = 0
        for j in range(len(reflist)):
            fit = make_engine(**dict(kw, reflist=reflist[j:j+1],
                                     reflist2=reflist2[j:j+1]))
            x = np.asarray(fit.dmslines[0][0], float)
            y = np.asarray(fit.dmslines[0][1], float)
            pts = np.stack([y, x], axis=1)
            ok = ~np.isnan(pts).any(1)
            pairs = [(pts[i], pts[i+1]) for i in range(len(pts) - 1)
                     if ok[i] and ok[i+1]]
            if len(pairs) < 5:
                continue                  # this reflection misses the detector
            calc = ts.dmscalc_ico_hkl(
                reflist[j:j+1], hkllist, np.round(kw['hkl']), 1,
                [kw['psi'] - 180, kw['psi'] + 180], 100, kw['hkl'],
                np.matrix([[1, 0, 0], [0, 0, 1]]), np.zeros(kw['shape']), 0.0,
                kw['azir'], kw['psi'], 600, 600, 0,
                ig[10], ig[11], ig[12], ig[13], ig[14],
                reflist2[j:j+1], list(ig[15:24]))
            calc.crystal_system = (kw['bravais']
                                   if kw['bravais'] in ts.CONVENTIONAL_SYSTEMS
                                   else None)
            idx = np.asarray(calc.roiindex(ig), dtype=float)
            if len(idx) < 5:
                continue
            A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
            worst = _seg_dist(idx, A, B).max()
            assert worst <= 3.0, (
                '%s reflection %s: the ROI engine puts points %.0f px off the '
                'line the fit engine draws'
                % (kw['bravais'], reflist[j], worst))
            checked += 1
        assert checked >= 3, ('%s fixture put too few lines on the detector'
                              % kw['bravais'])


def test_outline_of_a_built_half_is_a_clean_strip():
    """End to end with the drawing side: a half-path rasterised into a kernel
    plane, then outlined, contains everything msroi integrates through it."""
    from .test_roi_outline import _inside
    shape = (700, 800)
    p = _arc(n=200)
    half = len(p) // 2
    k = ts.roi_rasterise(p[:half], shape)
    img = np.random.default_rng(0).random(shape)
    _, pts = ts.msroi(img, k, 21)
    rows, cols = ts.roi_outline(k, 21)
    assert _inside(rows, cols, pts) == 1.0


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  ', name)
    print('all roi kernel tests passed')
