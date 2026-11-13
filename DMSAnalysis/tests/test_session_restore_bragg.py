"""A session restore re-derives the θ window from the *restored* hkl.

`_restore_from_dict` reloads the scan first, while the window still holds the
previous hkl and energy, so the θ window, the hkl scan and `hklint` were all
centred on the previous primary reflection and never recomputed.  Restoring a
session of another crystal made the window NaN (sin θ > 1), and Build curves
then laid every ROI over the whole detector and ran out of memory sampling it.
"""

import os

import numpy as np

from .gui_harness import slider_on

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'fit_conventional_tetragonal_PMN_PT_example.json')


def _thb(sl, win):
    return sl.ts.bragg(win._lattice, win._hkl.copy(), win.ig[14]).th()[0]


def test_restore_recentres_the_theta_window_on_the_restored_hkl():
    sl = slider_on(CONFIG)
    win = sl.win
    before = win._session_dict()
    try:
        session = dict(before, curves_built=False)
        # Leave the window on a primary reflection with no Bragg condition, as
        # another crystal's session would, then restore a valid one over it.
        win._hkl[:] = np.asarray(before['hkl']) * 50
        win._thrange = [np.nan, np.nan]
        session['hkl'] = list(np.asarray(before['hkl']) * 1.05)
        win._restore_from_dict(session)

        thb = _thb(sl, win)
        assert np.isfinite(thb)
        assert np.allclose(win._thrange, [thb - 27, thb + 10])
        assert np.array_equal(win._hklint, np.round(win._hkl))
        assert np.array_equal(sl.hklint, np.round(win._hkl))
    finally:
        win._restore_from_dict(before)


def test_build_refuses_a_nan_theta_window():
    """The worker fails fast instead of allocating a full-frame sampler."""
    sl = slider_on(CONFIG)
    win = sl.win
    errors = []
    w = sl.BuildWorker.__new__(sl.BuildWorker)
    w._ig, w._sel6d = np.asarray(win.ig, float), np.array([[1, 0, 0]])
    w._thrange, w._numsteps = [np.nan, np.nan], 10
    w._reuse_kernel = None
    w.error = type('S', (), {'emit': staticmethod(errors.append)})()
    sl.BuildWorker.run(w)
    assert errors and 'θ range' in errors[0]
