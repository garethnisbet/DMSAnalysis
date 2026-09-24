"""|q_perp| beside psi_err in the Geo 3-click / nearest-ref rankings.

``_ewald_scores`` is purely geometric: it asks whether a reflection's cone can
pass through the clicked directions, never whether that reflection is
observable.  6D indexing makes the reflection set dense, so several candidates
routinely score alike and the ranking cannot separate them.  For an icosahedral
quasicrystal the structure factor falls off steeply with the perpendicular-space
component, so |q_perp| is the tie-breaker — reported next to psi_err rather than
folded into it, since a composite score would hide a poor geometric match behind
a plausible |q_perp|.

The risk this pins is *alignment*: ``_perp_strengths`` indexes
``full_reflist``/``full_reflist2`` with positions derived from
``full_reflist_6d``, so the three arrays must stay row-aligned and the helper
must preserve the caller's order (the candidates arrive sorted by psi_err, not
in list order).

Run standalone:
    python -m DMSAnalysis.tests.test_perp_strengths
or under pytest:
    pytest DMSAnalysis/tests/test_perp_strengths.py
"""

import os

import numpy as np

from .gui_harness import slider_on

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'SelectedLines_GoodInitialGuess.json')

# One icosahedral star (symmetry-equivalent, so equal |q_perp|) plus two
# reflections of different perpendicular-space character.  Used directly rather
# than whatever reflections the window happens to hold, because the GUI harness
# is a singleton and another test's config may have built the window.
REF_6D = np.array([
    [-1, -1, -2, -1,  1,  1],
    [-1,  1, -1, -2, -1,  1],
    [ 2,  1,  1,  1,  1,  1],
    [-1,  0, -2, -2,  0,  1],
    [ 1,  0, -3, -1,  3,  4],
])


def _with_reflist(win, sl, ref_6d):
    """Install `ref_6d` as the window's candidate pool, projected as the
    quasicrystal branch projects it, and return a restore callable."""
    saved = (sl.CONVENTIONAL, win.full_reflist, win.full_reflist2,
             win.full_reflist_6d)
    sl.CONVENTIONAL = False
    rl, rl2 = sl.build_reflist_from_6d(ref_6d)
    win.full_reflist, win.full_reflist2, win.full_reflist_6d = rl, rl2, ref_6d

    def restore():
        (sl.CONVENTIONAL, win.full_reflist, win.full_reflist2,
         win.full_reflist_6d) = saved
    return restore


def test_conventional_has_no_perpendicular_space():
    """A conventional crystal has no cut-and-projection, so the helper reports
    nothing rather than a column of zeros."""
    sl  = slider_on(CONFIG)
    win = sl.win
    saved = sl.CONVENTIONAL
    sl.CONVENTIONAL = True
    try:
        assert win._perp_strengths([0]) is None
    finally:
        sl.CONVENTIONAL = saved


def test_norms_match_the_projected_components():
    """|q_perp| and the perp/par ratio are the norms of the projected
    components, row-aligned with the 6D indices they came from."""
    sl  = slider_on(CONFIG)
    win = sl.win
    restore = _with_reflist(win, sl, REF_6D)
    try:
        idx = np.arange(len(REF_6D))
        qperp, ratio = win._perp_strengths(idx)
        assert qperp.shape == (len(REF_6D),)
        expect_perp = np.linalg.norm(np.asarray(win.full_reflist2, float), axis=1)
        expect_par  = np.linalg.norm(np.asarray(win.full_reflist,  float), axis=1)
        assert np.allclose(qperp, expect_perp)
        assert np.allclose(ratio, expect_perp / expect_par)
        assert np.all(qperp > 0), 'a quasicrystal reflection has perp component'
    finally:
        restore()


def test_symmetry_equivalents_share_qperp():
    """The three reflections of one icosahedral star are equivalent, so they
    carry the same |q_perp| — and a reflection of different character does not.
    This is the discrimination the column exists to show."""
    sl  = slider_on(CONFIG)
    win = sl.win
    restore = _with_reflist(win, sl, REF_6D)
    try:
        qperp, _ = win._perp_strengths(np.arange(len(REF_6D)))
        star = qperp[:3]
        assert np.allclose(star, star[0]), 'equivalents must share |q_perp|'
        assert qperp[3] > star[0] * 1.5, 'a weaker class must rank clearly higher'
    finally:
        restore()


def test_order_follows_the_caller_not_the_list():
    """Candidates reach the helper sorted by psi_err, so the returned rows must
    follow the index vector — a misalignment here would label every candidate
    with another reflection's |q_perp|."""
    sl  = slider_on(CONFIG)
    win = sl.win
    restore = _with_reflist(win, sl, REF_6D)
    try:
        straight, _ = win._perp_strengths(np.arange(len(REF_6D)))
        picked = [3, 0, 4]
        shuffled, _ = win._perp_strengths(picked)
        assert np.allclose(shuffled, straight[picked])
    finally:
        restore()


def test_rankings_report_qperp():
    """Both identify paths run end to end and put |q_perp| on the pick label.

    Only meaningful when the window really is on a quasicrystal (the harness is
    a singleton, so another test's config may own it) — the engines the searches
    drive are built from that config, not from the patched reflist.
    """
    sl  = slider_on(CONFIG)
    win = sl.win
    if sl.CONVENTIONAL:
        return
    ref = win.full_reflist_6d[0]
    arc = win._plot_arc(ref, '#00cccc')
    assert arc is not None and len(arc._x_data) >= 3
    xs, ys = arc._x_data, arc._y_data
    # Spread across the whole arc: bunched clicks are exactly the degenerate
    # case the column is there to break.
    pts = [(float(ys[i]), float(xs[i]))
           for i in (0, len(xs) // 2, len(xs) - 1)]

    win._psi_tol = 5.0
    win._run_geo_search(pts)
    label = win._lbl_pick.text()
    assert 'q' in label and '=' in label, label
    assert label.startswith('[%s]' % ' '.join('%d' % v for v in ref)), label

    win._run_nearest_ref(pts)
    label = win._lbl_pick.text()
    assert label.startswith('[%s]' % ' '.join('%d' % v for v in ref)), label
    assert 'q' in label, label


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('all passed')
