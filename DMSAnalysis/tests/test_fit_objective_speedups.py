"""The fit objective's fast paths must reproduce the slow ones bit for bit.

Two things made every objective evaluation of the image fit slow, and both were
replaced with something cheaper that is *exactly* equal, not approximately:

* the simulated image was smeared with ``ndimage.convolve`` over the whole
  detector frame (~0.25 s, ~80% of an evaluation), although only the line
  pixels are non-zero — ``convolve_binary`` adds their kernel weights directly,
  in ndimage's own order, so every float sum is the same;
* every ROI re-derived the pixels it integrates with an ``np.where`` over a
  whole detector-sized kernel plane — ``msroi_sampler`` works them out once and
  the engine keeps one per ROI.

The ROI kernel itself is now stored sparsely (``RoiKernel``, ~0.6 MB where the
dense stack was ~830 MB), which is what makes it cheap to send the engine to the
worker processes the slider's multi-start fits now run in.

"Exactly" is the point: the optimisers are sensitive to the last bit of the
objective, and these tests compare with ``array_equal``, not a tolerance.

Run standalone:
    python -m DMSAnalysis.tests.test_fit_objective_speedups
or under pytest:
    pytest DMSAnalysis/tests/test_fit_objective_speedups.py
"""

import copy

import numpy as np
from scipy import ndimage

from .. import ts_quasi as ts
from .test_dms_curves import CUBIC, cubic_reflist


def _binary(shape, rows, cols):
    im = np.zeros(shape)
    im[rows, cols] = 1
    return im


def test_convolve_binary_is_ndimage_convolve_bit_for_bit():
    """Every sigma the configs use and a couple either side — sigma <= 1 is
    where ndimage drops the epsilon-sized corner weights from its footprint —
    with lit pixels on, and within the kernel radius of, every edge and corner,
    where 'reflect' reads a pixel's mirror image.  Duplicated pixels too: the
    engine's index repeats a pixel whenever both psi branches land on it."""
    rng = np.random.default_rng(0)
    e = np.arange(9)
    for shape in [(300, 211), (211, 300), (40, 37)]:
        H, W = shape
        rows = np.concatenate([rng.integers(0, H, 400), e, H - 1 - e, e, H - 1 - e,
                               rng.integers(0, H, 20), np.zeros(20, int)])
        cols = np.concatenate([rng.integers(0, W, 400), e, W - 1 - e, W - 1 - e, e,
                               np.full(20, W - 1), rng.integers(0, W, 20)])
        rows = np.concatenate([rows, rows[:50]])
        cols = np.concatenate([cols, cols[:50]])
        for sigma in (0.5, 1.0, 2.0, 4.0, 4.5, 9.0):
            K = ts.makekernel('gauss', 15, sigma)
            ref = ndimage.convolve(_binary(shape, rows, cols), K)
            for frac in (1.0, 0.0):          # the sparse path, then the dense fallback
                got = ts.convolve_binary(shape, rows, cols, K, dense_fraction=frac)
                assert np.array_equal(got, ref), (
                    '%s sigma=%g dense_fraction=%g: %d pixels differ'
                    % (shape, sigma, frac, np.count_nonzero(got != ref)))


def test_convolve_binary_of_nothing_is_zero():
    got = ts.convolve_binary((50, 60), [], [], ts.makekernel('gauss', 15, 2.0))
    assert got.shape == (50, 60) and not got.any()


def _reference_msroi(img, kernel, width):
    """msroi as it was written before the sampler was factored out of it."""
    vs_idx = np.where(kernel > 0)
    dv = np.array([[vs_idx[0][-1] - vs_idx[0][0], vs_idx[1][-1] - vs_idx[1][0]]],
                  dtype=float)
    v = (dv @ np.array([[0, 1], [-1, 0]])).flatten()
    v = v / np.linalg.norm(v)
    vs = np.stack([vs_idx[0], vs_idx[1]], axis=1).astype(float)
    irange = np.arange(int(np.round(-width / 2.0)), int(np.round(width / 2.0)))
    shifted = np.round(vs[np.newaxis] + np.outer(irange, v)[:, np.newaxis]).astype(int)
    valid = ((shifted[:, :, 0] > 0) & (shifted[:, :, 0] < img.shape[0]) &
             (shifted[:, :, 1] >= 0) & (shifted[:, :, 1] < img.shape[1]))
    r0 = np.clip(shifted[:, :, 0], 0, img.shape[0] - 1)
    r1 = np.clip(shifted[:, :, 1], 0, img.shape[1] - 1)
    v1 = np.where(valid, img[r0, r1], 0.0).sum(axis=1, keepdims=True)
    w_idx, n_idx = np.where(valid)
    return v1, shifted[w_idx, n_idx], v


def _scene(shape=(700, 650), numsteps=400, simsigma=4.5, width=45):
    """A fit engine with real ROIs (from roibuilder, as fit.py builds them) and
    targets extracted from a nudged, noisy copy of its own simulation."""
    kw = dict(CUBIC)
    lattice, hkl, energy, azir = kw['lattice'], np.array(kw['hkl']), kw['energy'], kw['azir']
    reflist = np.asarray(cubic_reflist(18), float)
    reflist2 = np.zeros_like(reflist)
    px, py, detdist, psi = shape[0] // 2, shape[1] // 2, 1300.0, -180.0
    thb = ts.bragg(lattice, hkl, energy).th()[0]
    thr = [thb - 27, thb + 10]
    hkllist = ts.pilkhlrange(lattice, hkl, energy, thr[0], thr[1]).hklscan(numsteps)
    ig = np.zeros(24)
    ig[:6] = lattice
    ig[10] = detdist
    ig[14] = energy
    detvects = np.matrix([[1, 0, 0], [0, 0, 1]])
    psirange = [psi - 360, psi + 360]
    kernel = ts.roibuilder_ico_hkl(
        (reflist, hkllist, np.round(hkl), 1, psirange, 0, hkl, detvects, shape,
         simsigma, azir, psi, px, py, 0, detdist, 0.0, 0.0, 0.0, energy, ig,
         reflist2, [0.0] * 9, 'cubic'))

    def engine(centres, imdata):
        d = ts.dmsfit_ico_hkl(
            np.matrix(reflist), [thr[0], thr[1], numsteps], np.round(hkl), psirange,
            width, centres, kernel, hkl, detvects, imdata, simsigma, azir, psi,
            px, py, 0, 'cubic', True, False, detdist, 0.0, 0.0, 0.0, energy,
            np.matrix(reflist2), [0.0] * 9, ig[0])
        d.setLattice(list(ig[:6]))
        d.setIGFull(ig)
        return d

    x0 = ig[ts.reduced_param_indices('cubic', True, False)]
    truth = x0.copy()
    truth[1] += 0.3
    sim = engine(np.zeros((kernel.shape[2], 1)), np.zeros(shape))
    sim.imcalc(truth)
    imdata = sim.imsim * 100 + np.random.default_rng(1).normal(0, 1.0, shape)
    centres = np.array([ts.multiroifit(imdata, kernel, width, 0.02)[0][:, 2]]).T
    rng = np.random.default_rng(7)
    xs = [x0] + [x0 + rng.normal(0, 0.02, x0.shape) for _ in range(3)]
    return engine(centres, imdata), xs


def test_msroi_is_unchanged_by_the_refactor():
    dms, xs = _scene()
    assert dms.kernel.shape[2] >= 4, 'test premise: the scene must have ROIs'
    dms.imcalc(xs[0])
    for i in range(dms.kernel.shape[2]):
        ref = _reference_msroi(dms.imsim, dms.kernel[:, :, i], dms.width)
        got = ts.msroi2(dms.imsim, dms.kernel[:, :, i], dms.width)
        for a, b in zip(got, ref):
            assert np.array_equal(a, b)


def test_the_objective_is_the_slow_objective_bit_for_bit():
    """The engine's residuals, scored through the cached samplers and the
    sparse smear, against the same geometry scored the old way: a dense binary
    image, ndimage.convolve, and msroi derived from scratch for every ROI."""
    dms, xs = _scene()
    for x in xs:
        fast = dms.residuals(x)
        binary = _binary(dms.imsim.shape, *dms.dmsindex)
        slow_im = ndimage.convolve(binary, ts.makekernel('gauss', 15, dms.simsigma))
        assert np.array_equal(dms.imsim, slow_im)
        centres = []
        for i in range(dms.kernel.shape[2]):
            y = _reference_msroi(slow_im, dms.kernel[:, :, i], dms.width)[0][:, 0]
            try:
                centres.append(ts.peakfit(np.arange(len(y)), y, dms.peakmethod,
                                          dms.peaksig)[0][2])
            except Exception:
                centres.append(np.nan)
        slow, _, _ = ts.centre_residuals(centres, dms.centres[:, 0],
                                         dms._roi_fail_penalty())
        assert np.array_equal(fast, slow)


def test_the_roi_cache_follows_the_kernel():
    """Swapping in a different kernel (or width) must not reuse samplers built
    for the old one, and a deep copy — each parallel start takes one — must
    score exactly as the original does."""
    dms, xs = _scene()
    before = dms.fit(xs[1])
    twin = copy.deepcopy(dms)
    assert twin.fit(xs[1]) == before

    dms.kernel = dms.kernel[:, :, :2].copy()
    fresh = copy.deepcopy(dms)
    fresh.__dict__.pop('_roi_sampler_cache', None)
    fresh.centres = fresh.centres[:2]
    dms.centres = dms.centres[:2]
    assert dms.fit(xs[1]) == fresh.fit(xs[1])

    dms.width = 31
    fresh.width = 31
    fresh.__dict__.pop('_roi_sampler_cache', None)
    assert dms.fit(xs[1]) == fresh.fit(xs[1])


# ── sparse ROI storage ────────────────────────────────────────────────────────

def test_the_sparse_kernel_is_the_dense_kernel():
    """roibuilder returns a RoiKernel; everything that reads a kernel must see
    exactly the dense stack it replaced."""
    dms, _ = _scene()
    K = dms.kernel
    assert isinstance(K, ts.RoiKernel)
    dense = np.asarray(K)
    assert dense.shape == K.shape
    for i in range(K.shape[2]):
        assert np.array_equal(K[:, :, i], dense[:, :, i])
        for a, b in zip(K.pixels(i), np.where(dense[:, :, i] > 0)):
            assert np.array_equal(a, b)
    sub = K[:, :, [1, 3]]
    assert isinstance(sub, ts.RoiKernel) and sub.shape[2] == 2
    assert np.array_equal(np.asarray(sub), dense[:, :, [1, 3]])
    assert np.array_equal(np.asarray(K.copy()), dense)
    assert K.nbytes < dense.nbytes / 100


def test_a_dense_kernel_and_a_sparse_one_score_the_same():
    """The engine and multiroifit/multiroifit2 accept either form and give
    identical results, including the ROI mask stack they return."""
    dms, xs = _scene()
    dense = copy.deepcopy(dms)
    dense.kernel = np.asarray(dms.kernel)
    for x in xs:
        assert np.array_equal(dms.residuals(x), dense.residuals(x))
    img = dms.imdata
    for fn, extra in ((ts.multiroifit, (10,)), (ts.multiroifit2, (0.02, None))):
        a = fn(img, dms.kernel, dms.width, *extra)
        b = fn(img, dense.kernel, dms.width, *extra)
        assert np.array_equal(a[0], b[0], equal_nan=True)
        for u, v in zip(a[1:4], b[1:4]):
            assert np.array_equal(u, v)
        assert np.array_equal(np.asarray(a[4]), np.asarray(b[4]))
        assert np.array_equal(np.sum(a[4], 2), np.sum(np.asarray(b[4]), 2))


# ── multi-start in worker processes ───────────────────────────────────────────

def test_process_starts_match_in_process_starts():
    """A start is a pure function of the engine and its starting point, so it
    must come back the same from a worker process as from this one."""
    from joblib import Parallel, delayed
    dms, xs = _scene()
    x0 = xs[0]
    free = np.arange(len(x0))
    steps = np.full(len(x0), 1e-2)
    starts = [np.zeros(len(x0)), np.full(len(x0), 0.5)]
    kw = dict(options={'rhobeg': 1.0, 'maxiter': 40})
    here = [ts.run_scaled_start(dms, x0, free, x0, steps, s, 'COBYLA', 1e-6, **kw)
            for s in starts]
    there = Parallel(n_jobs=2)(
        delayed(ts.run_scaled_start)(dms, x0, free, x0, steps, s, 'COBYLA', 1e-6, **kw)
        for s in starts)
    for (r1, f1, z1), (r2, f2, z2) in zip(here, there):
        assert np.array_equal(r1.x, r2.x) and r1.fun == r2.fun and r1.nfev == r2.nfev
        assert f1 == f2 and np.array_equal(z1, z2)
        assert f1 <= r1.fun


def test_stop_reaches_a_worker_process():
    from joblib import Parallel, delayed
    dms, xs = _scene()
    x0 = xs[0]
    flag = ts.StopFlag()
    try:
        flag.set()
        try:
            Parallel(n_jobs=2)(
                delayed(ts.run_scaled_start)(dms, x0, np.arange(len(x0)), x0, None,
                                             np.zeros(len(x0)), 'COBYLA', 1e-6,
                                             stop=flag)
                for _ in range(2))
        except ts.FitStopped:
            pass
        else:
            raise AssertionError('a set StopFlag did not stop the worker')
    finally:
        flag.close()
    flag.set()                      # after close: a late Stop must not raise


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
