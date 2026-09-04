"""``ts_quasi.roi_outline`` — the drawn ROI is the region ``msroi`` integrates.

The slider's "ROIs" overlay draws these outlines on the detector image, so the
one thing they must not do is describe a strip the integrated curve did not come
from.  Each test builds a kernel path, integrates through it with ``msroi``, and
checks the pixels ``msroi`` actually summed all fall inside the outline.

Run standalone:
    python -m DMSAnalysis.tests.test_roi_outline
or under pytest:
    pytest DMSAnalysis/tests/test_roi_outline.py
"""

import numpy as np

from .. import ts_quasi as ts


SHAPE = (240, 240)


def _kernel(slope, row_dominant):
    """One-pixel-wide straight path, laid out the way `roibuilder_ico_hkl` lays
    one: a contiguous run along the dominant axis, interpolated on the other."""
    k = np.zeros(SHAPE)
    t = np.arange(40, 200)
    o = (120 + slope * (t - 120)).astype(int)
    m = (o >= 0) & (o < SHAPE[1])
    if row_dominant:
        k[t[m], o[m]] = 1
    else:
        k[o[m], t[m]] = 1
    return k


def _inside(ring_rows, ring_cols, pts, pad=1.5):
    """Fraction of `pts` (N,2 row/col) inside the closed outline, with a pixel of
    slack for the integer rounding msroi does and the outline does not."""
    from matplotlib.path import Path
    poly = Path(np.stack([np.asarray(ring_rows), np.asarray(ring_cols)], axis=1))
    return poly.contains_points(np.asarray(pts, dtype=float), radius=pad).mean()


def test_outline_contains_the_integrated_pixels():
    """Every pixel msroi summed lies inside the drawn outline, at either
    orientation (a shallow line's kernel is column-ordered by np.where, which is
    the case the outline has to re-sort before offsetting)."""
    img = np.random.default_rng(0).random(SHAPE)
    for row_dominant in (True, False):
        for slope in (0.15, 1.0, 4.0):
            k = _kernel(slope, row_dominant)
            for width in (5, 21, 45):
                _, pts = ts.msroi(img, k, width)
                rows, cols = ts.roi_outline(k, width)
                frac = _inside(rows, cols, pts)
                assert frac == 1.0, (
                    'slope %.2f row_dominant=%s width=%d: %.3f of the '
                    'integrated pixels fall outside the outline'
                    % (slope, row_dominant, width, frac))


def test_outline_width_matches_the_integration_width():
    """The outline is `width` pixels across, measured perpendicular to the path
    — the same span msroi's offset range covers."""
    k = _kernel(0.4, True)
    for width in (5, 21, 45):
        rows, cols = ts.roi_outline(k, width)
        ring = np.stack([rows, cols], axis=1)
        n = (len(ring) - 1) // 2
        e1, e2 = ring[:n], ring[n:2 * n][::-1]
        sep = np.linalg.norm(e1 - e2, axis=1)
        expected = float(len(np.arange(int(round(-width / 2.0)),
                                       int(round(width / 2.0)))) - 1)
        assert np.allclose(sep, expected), (
            'width %d: outline spans %.2f-%.2f px, expected %.2f'
            % (width, sep.min(), sep.max(), expected))


def test_degenerate_kernels_return_nothing():
    """An empty or single-pixel kernel has no direction to offset along; the
    caller gets an empty outline rather than a NaN one it would draw."""
    for k in (np.zeros(SHAPE), ):
        rows, cols = ts.roi_outline(k, 21)
        assert np.size(rows) == 0 and np.size(cols) == 0
    k = np.zeros(SHAPE); k[10, 10] = 1
    rows, cols = ts.roi_outline(k, 21)
    assert np.size(rows) == 0


def test_the_two_halves_of_a_line_draw_as_displaced_strips():
    """The builder's contract the overlay relies on: one reflection gives two
    kernel planes, each half of its line — that pair, moving in opposite senses
    when the line rotates, is what makes the fit sensitive to a rotation and not
    only to a shift.  Drawn, they must come out as two strips displaced along
    the line, each covering its own half and neither straying into the other's."""
    k = _kernel(0.4, True)
    idx = np.argwhere(k > 0)
    half = len(idx) // 2
    k1 = np.zeros(SHAPE); k1[idx[:half, 0], idx[:half, 1]] = 1
    k2 = np.zeros(SHAPE); k2[idx[half:, 0], idx[half:, 1]] = 1
    rings = [np.stack(ts.roi_outline(kk, 21), axis=1) for kk in (k1, k2)]
    assert all(len(r) for r in rings)
    centroids = [r.mean(axis=0) for r in rings]
    assert np.linalg.norm(centroids[0] - centroids[1]) > 21, \
        'the two half-ROIs are drawn on top of each other'
    # Each outline holds its own half's pixels and not the other's.
    img = np.random.default_rng(1).random(SHAPE)
    for ring, own, other in ((rings[0], k1, k2), (rings[1], k2, k1)):
        _, pts_own = ts.msroi(img, own, 21)
        _, pts_other = ts.msroi(img, other, 21)
        assert _inside(ring[:, 0], ring[:, 1], pts_own) == 1.0
        assert _inside(ring[:, 0], ring[:, 1], pts_other) < 0.15


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  ', name)
    print('all roi_outline tests passed')
