"""The two fit-evaluation speedups must not change what a fit scores.

One evaluation of `dmsfit_ico_hkl.fit` used to spend almost all its time on two
steps, neither of them physics:

* blurring the simulated image with a 15x15 Gaussian (`ndimage.convolve`), and
* searching every full kernel plane for its ROI pixels (`np.where` in `msroi`),
  repeated on every evaluation although the kernel never changes during a fit.

The blur is now two 1-D passes (`blur_gauss`) and the ROI pixels are found once,
when the kernel is set (`roi_kernel_indices`).  These tests pin that both give
what the old code gave, and that the engine scores a fit identically.

Run standalone:
    python -m DMSAnalysis.tests.test_fit_eval_speedups
or under pytest:
    pytest DMSAnalysis/tests/test_fit_eval_speedups.py
"""

import numpy as np
from scipy import ndimage

from .. import ts_quasi as ts
from .test_dms_curves import cubic_kwargs, make_engine


def _old_blur(img, sigma):
    return ndimage.convolve(img, ts.makekernel('gauss', 15, sigma))


def test_separable_blur_matches_the_2d_convolution():
    rng = np.random.default_rng(0)
    img = np.zeros((300, 257))
    img[rng.integers(0, 300, 800), rng.integers(0, 257, 800)] = 1.0
    img[0, :] = 1.0                      # exercise the 'reflect' boundary
    img[:, -1] = 1.0
    for sigma in (0.7, 1.5, 3.0, 8.0):
        old = _old_blur(img, sigma)
        new = ts.blur_gauss(img, 15, sigma)
        assert np.allclose(new, old, rtol=0, atol=1e-12 * old.max()), sigma


def test_precomputed_indices_integrate_exactly_as_the_plane():
    rng = np.random.default_rng(1)
    img = rng.random((200, 220))
    kernel = np.zeros((200, 220, 3))
    t = np.linspace(0, 1, 150)
    kernel[(40 + 100 * t).astype(int), (30 + 160 * t**2).astype(int), 0] = 1
    kernel[(150 - 80 * t).astype(int), (60 + 120 * t).astype(int), 1] = 1
    kernel[5:7, 0:220, 2] = 1            # a ROI along the plate edge
    idx = ts.roi_kernel_indices(kernel)
    for i in range(kernel.shape[2]):
        v1a, v2a = ts.msroi(img, kernel[:, :, i], 31)
        v1b, v2b = ts.msroi(img, idx[i], 31)
        assert np.array_equal(v1a, v1b)
        assert np.array_equal(v2a, v2b)


def _kernel_from_lines(dms, shape):
    """Two planes per drawn line (its two halves), as the ROI builder lays them."""
    planes = []
    for x, y in dms.dmslines:
        x = np.asarray(x, float); y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        r, c = np.round(y[ok]).astype(int), np.round(x[ok]).astype(int)
        on = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        r, c = r[on], c[on]
        if len(r) < 40:
            continue
        h = len(r) // 2
        for rr, cc in ((r[:h], c[:h]), (r[h:], c[h:])):
            p = np.zeros(shape); p[rr, cc] = 1; planes.append(p)
    return np.stack(planes, axis=2)


def test_the_engine_scores_a_fit_as_the_old_code_did():
    kw = cubic_kwargs(numsteps=400)
    dms = make_engine(**kw)
    shape = np.shape(dms.imdata)
    kernel = _kernel_from_lines(dms, shape)
    assert kernel.shape[2] >= 6, 'fixture put too few lines on the detector'

    dms.kernel = kernel                  # the setter caches the ROI pixels
    dms.width = 45
    dms.centres = np.full((kernel.shape[2], 1), 22.0)
    ig = dms.ig_full
    x = ig[ts.reduced_param_indices('cubic', True, False)]
    x = x * (1 + 1e-4 * np.arange(len(x)))   # off the drawn geometry a little

    dms.simsigma = 1.5
    new_fit = dms.fit(x)
    new_coeffs = dms._simcoeffs()

    # The old path, by hand: the unblurred image, the 2-D convolution, and a
    # full-plane search per ROI.
    dms.simsigma = 0.0
    dms.imcalc(x)
    old_im = _old_blur(dms.imsim, 1.5)
    old_c = []
    for i in range(kernel.shape[2]):
        s, _ = ts.msroi(old_im, kernel[:, :, i], dms.width)
        coef, _, _ = ts.peakfit(np.arange(len(s)), s[:, 0], dms.peakmethod,
                                dms.peaksig)
        old_c.append(coef[2])
    old_c = np.array(old_c)
    assert np.allclose(new_coeffs[:, 2], old_c, rtol=0, atol=1e-6)
    old_fit = float(np.sum((old_c - 22.0) ** 2))
    assert abs(new_fit - old_fit) <= 1e-6 * max(1.0, old_fit)


def test_reassigning_the_kernel_refreshes_the_indices():
    """ridgefit sets `eng.kernel` directly; the cached pixels must follow."""
    dms = make_engine(**cubic_kwargs(numsteps=100, shape=(100, 100)))
    k = np.zeros((100, 100, 2)); k[10, 10:20, 0] = 1; k[50:60, 5, 1] = 1
    dms.kernel = k
    assert len(dms._kernel_idx) == 2
    assert np.array_equal(dms._kernel_idx[1][0], np.arange(50, 60))


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  ', name)
    print('all passed')
