"""The Reflist |q_perp| slider: narrowing the identify pool, live.

A quasicrystal's structure factor falls off steeply with the perpendicular-space
component, so most of the dense 6D reflection set is indexable but not
observable.  The slider drops those from what the three-click identify will
*consider*, which is what leaves it a pool it can choose within (see
`test_perp_strengths`).

It deliberately does **not** touch the generated reflection list, the overlay
slice or the fit.  That is what makes it live: filtering the list meant a full
`_regenerate_reflist` per slider step — at Depth 3 that is ~100k reflections
reprojected and the overlay engine rebuilt, tens of ms, on every one of a
2000-step slider's events.  Masking a cached norm array is ~0.02 ms, so the cut
can act while the slider is being dragged.

What is pinned here:

* **The reflist is untouched.**  The regression the live rework had to avoid
  re-introducing.
* **The mask matches the cached norms**, which must stay row-aligned with
  ``full_reflist_6d`` — the identify indexes ``full_reflist``/``full_reflist2``
  by positions found in it.
* **The cut is absolute, the range is not.**  |q_perp| does not depend on Depth,
  so a chosen ceiling keeps meaning the same thing when Depth grows the set; an
  *untouched* slider must equally not inherit the old maximum as a cutoff.
* **A ceiling that passes nothing is ignored**, not honoured — there would be
  nothing left to identify against.
* **Arcs survive a re-filter.**  A candidate hidden by a tighter cut comes back
  without a second imcalc, and one the user has selected is never hidden at all.

Run standalone (drives the real search and re-filter):
    python -m DMSAnalysis.tests.test_qperp_filter
or under pytest:
    pytest DMSAnalysis/tests/test_qperp_filter.py
"""

import os

import numpy as np

from .gui_harness import slider_on

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'SelectedLines_GoodInitialGuess.json')

QPERP = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def _mode(sl, conventional, qperp=None):
    """Patch the module's crystal mode and, optionally, the cached |q_perp|,
    returning a restore callable.

    The mask short-circuits on a conventional crystal, and the GUI harness is a
    singleton so another test's config may own the window — hence the patch
    rather than a config that guarantees the mode.  The restore regenerates,
    which puts the cache, the slider range and the reflist back in step.
    """
    win = sl.win
    saved = (sl.CONVENTIONAL, win._qperp_cut, win._qperp_all, win.full_reflist_6d)
    sl.CONVENTIONAL = bool(conventional)
    if qperp is not None:
        win._qperp_all = np.asarray(qperp, dtype=float)
        win.full_reflist_6d = np.arange(len(qperp) * 6).reshape(len(qperp), 6)

    def restore():
        (sl.CONVENTIONAL, win._qperp_cut, win._qperp_all,
         win.full_reflist_6d) = saved
        win._regenerate_reflist()
    return restore


def test_range_starts_at_the_smallest_qperp():
    """The slider spans the values that exist, so every position leaves
    something to identify against.  Ranging from zero gave the bottom of the
    travel a dead zone that passed nothing and fell back to the whole list —
    which showed the same candidates at the bottom as at the top."""
    sl = slider_on(CONFIG)
    assert sl.DMSSlider._qperp_range(np.array([0.5, 2.0])) == (0.5, 2.0)


def test_range_is_never_zero_width():
    """FloatSlider maps value to position by dividing through the span, so a
    zero-width range raises on the first setValue.  A conventional crystal (all
    zeros), an empty list, and a single-shell set all reach this."""
    rng = slider_on(CONFIG).DMSSlider._qperp_range
    for q in (np.zeros(4), np.zeros(0), np.full(3, 0.7)):
        lo, hi = rng(q)
        assert hi > lo, (q, lo, hi)


def test_equivalent_reflections_are_not_split_by_float_noise():
    """Symmetry equivalents have mathematically equal |q_perp| but their norms
    differ in the last bits, so an exact compare admits one member of a star and
    drops another — which reads as the cut behaving erratically."""
    le = slider_on(CONFIG).DMSSlider._qperp_le
    star = 0.16692554 * np.array([1.0, 1.0 + 2e-16, 1.0 - 3e-16])
    assert le(star, star.min()).all()
    assert not le(np.array([0.2839]), 0.16692554).any()


def test_mask_selects_on_the_cached_norms():
    sl  = slider_on(CONFIG)
    win = sl.win
    restore = _mode(sl, conventional=False, qperp=QPERP)
    try:
        win._qperp_cut = 3.0
        assert np.array_equal(win._qperp_mask(), QPERP <= 3.0)
        win._qperp_cut = None
        assert win._qperp_mask().all()
    finally:
        restore()


def test_conventional_crystal_has_nothing_to_cut_on():
    sl  = slider_on(CONFIG)
    win = sl.win
    restore = _mode(sl, conventional=True, qperp=QPERP)
    try:
        win._qperp_cut = 1.5
        assert win._qperp_mask().all()
    finally:
        restore()
    assert not win._sl_qperp.isEnabled() or not sl.CONVENTIONAL


def test_a_ceiling_below_everything_keeps_the_smallest_shell():
    """Not nothing — and, the regression this guards, not *everything*.  Falling
    back to the whole list is what made the bottom of the slider show the same
    candidates as the top."""
    sl  = slider_on(CONFIG)
    win = sl.win
    restore = _mode(sl, conventional=False, qperp=QPERP)
    try:
        win._qperp_cut = 0.1                   # below every |q_perp| here
        mask = win._qperp_mask()
        assert mask.sum() == 1 and mask[0], mask
        assert not mask.all(), 'must not fall back to the full pool'
    finally:
        restore()


def test_the_pool_shrinks_monotonically_down_the_slider():
    """The property the user sees: dragging down never brings candidates back.
    Pinned end to end, because it is a composition of the range, the mask and
    the top-of-range-is-off rule — each of which was individually defensible
    while together they made the bottom behave like the top."""
    sl  = slider_on(CONFIG)
    win = sl.win
    if sl.CONVENTIONAL:
        return
    saved = win._qperp_cut
    try:
        win._qperp_cut = None
        win._regenerate_reflist()
        q = win._qperp_values()
        lo, hi = win._sl_qperp._min, win._sl_qperp._max
        pools = []
        for f in np.linspace(0.0, 1.0, 9):
            win._on_qperp_cut_changed(float(lo + f * (hi - lo)))
            pools.append(int(win._qperp_mask().sum()))
        assert all(a <= b for a, b in zip(pools, pools[1:])), pools
        assert pools[0] > 0, 'the bottom must leave something to identify'
        assert pools[0] < pools[-1], 'the bottom must be a real cut'
        assert pools[-1] == q.size, 'the top must be off'
    finally:
        win._qperp_cut = saved
        win._regenerate_reflist()


def test_a_stale_cache_does_not_mask():
    """The norms are recomputed per regeneration; if they ever fall out of step
    with the list, masking by position would be reading another reflection's
    row, so the mask stands down instead."""
    sl  = slider_on(CONFIG)
    win = sl.win
    restore = _mode(sl, conventional=False, qperp=QPERP)
    try:
        win._qperp_cut = 3.0
        win._qperp_all = QPERP[:2]             # shorter than the list
        assert win._qperp_mask().all()
    finally:
        restore()


def test_top_of_range_reads_as_off():
    """Dragging back to the top records None, not that number — otherwise the
    next Depth increase would inherit it as a real cutoff."""
    sl  = slider_on(CONFIG)
    win = sl.win
    saved = win._qperp_cut
    try:
        top = win._sl_qperp._max
        win._on_qperp_cut_changed(top)
        assert win._qperp_cut is None
        half = top / 2.0
        win._on_qperp_cut_changed(half)
        assert win._qperp_cut == half
    finally:
        win._qperp_cut = saved
        win._regenerate_reflist()


def test_the_cut_never_touches_the_reflist():
    """The whole point of the live rework: moving the slider must not regenerate
    or shorten the generated list, only narrow what the identify searches."""
    sl  = slider_on(CONFIG)
    win = sl.win
    if sl.CONVENTIONAL:
        return
    saved = win._qperp_cut
    try:
        win._qperp_cut = None
        win._regenerate_reflist()
        before = np.array(win.full_reflist_6d, copy=True)
        q = win._qperp_values()
        assert q.size == len(before)

        win._on_qperp_cut_changed(float(np.median(q)))
        assert np.array_equal(win.full_reflist_6d, before), 'reflist was modified'
        assert len(win.full_reflist) == len(win.full_reflist2) == len(before)
        assert int(win._qperp_mask().sum()) < len(before), 'nothing was masked'
    finally:
        win._qperp_cut = saved
        win._regenerate_reflist()


def test_cut_survives_a_depth_change():
    """The ceiling is absolute, so a Depth change extends the range past it
    rather than resetting it — and the cache re-ranges to the new set."""
    sl  = slider_on(CONFIG)
    win = sl.win
    if sl.CONVENTIONAL:
        return
    saved_auto, saved_cut = win._use_auto, win._qperp_cut
    try:
        win._use_auto = True
        win._qperp_cut = None
        win._sb_depth.setValue(2)
        q = win._qperp_values()
        assert q.size and win._sl_qperp.val == win._sl_qperp._max

        cut = float(np.median(q))
        win._on_qperp_cut_changed(cut)
        pool_2 = int(win._qperp_mask().sum())
        assert pool_2 < q.size

        win._sb_depth.setValue(3)
        assert win._qperp_cut == cut, 'an absolute ceiling must survive'
        assert win._sl_qperp._max > cut, 'the range must extend past it'
        assert win._qperp_all.size == len(win.full_reflist_6d)
        assert win._qperp_all[win._qperp_mask()].max() <= cut
    finally:
        win._use_auto, win._qperp_cut = saved_auto, saved_cut
        win._sb_depth.setValue(1)


def test_refilter_hides_and_restores_candidate_arcs():
    """A tighter cut drops candidates from the drawn set; loosening it brings
    them back from the arc cache rather than re-tracing them."""
    sl  = slider_on(CONFIG)
    win = sl.win
    if sl.CONVENTIONAL:
        return
    saved_cut, saved_tol = win._qperp_cut, win._psi_tol
    try:
        win._qperp_cut = None
        win._regenerate_reflist()
        ref = win.full_reflist_6d[0]
        arc = win._plot_arc(ref, '#00cccc')
        assert arc is not None and len(arc._x_data) >= 3
        xs, ys = arc._x_data, arc._y_data
        pts = [(float(ys[i]), float(xs[i])) for i in (0, len(xs) // 2, len(xs) - 1)]

        win._psi_tol = 12.0
        win._run_geo_search(pts)
        drawn = dict(win._geo_arcs)
        assert drawn, 'the search drew no candidates'
        n_before = sum(1 for a in drawn.values() if a.isVisible())

        q = win._qperp_values()
        win._on_qperp_cut_changed(float(q.min()) * 1.05)
        win._refilter_geo_candidates()
        assert sum(1 for a in win._geo_arcs.values() if a.isVisible()) < n_before

        win._on_qperp_cut_changed(win._sl_qperp._max)
        win._refilter_geo_candidates()
        assert sum(1 for a in win._geo_arcs.values() if a.isVisible()) == n_before
        # Same items throughout: nothing was re-traced.
        assert all(win._geo_arcs.get(r) is a for r, a in drawn.items())
    finally:
        win._qperp_cut, win._psi_tol = saved_cut, saved_tol
        win._on_clear_picks()
        win._regenerate_reflist()


def test_a_selected_candidate_stops_being_refiltered():
    """Selecting a candidate hands the arc to the user; a later re-filter must
    not hide a reflection that is in the selected list."""
    sl  = slider_on(CONFIG)
    win = sl.win
    if sl.CONVENTIONAL:
        return
    try:
        win._qperp_cut = None
        win._regenerate_reflist()
        row = 0
        ref = win.full_reflist_6d[row]
        arc = win._plot_arc(ref, '#00cccc')
        assert arc is not None
        win._geo_arcs[row] = arc
        win._add_arc_to_list(ref, arc)
        assert row not in win._geo_arcs, 'a selected arc must leave the cache'
    finally:
        win._on_clear_picks()
        win._regenerate_reflist()


def test_the_drawn_set_is_capped_and_says_so():
    """The pool is nested as the cut loosens, but the *drawn* set is not: only
    `GEO_MAX_DRAWN` arcs are traced and the ranking is over the current pool, so
    letting more candidates in can push a drawn one out.

    That is deliberate.  Ranking over the unfiltered candidates instead would
    make the drawn set nested, but it would show almost nothing at a useful cut
    — the best-scoring candidates are mostly the high-|q_perp| reflections the
    cut exists to remove, so on the example config a cut that leaves 73
    candidates would draw 1 line rather than 10.  What was actually wrong was
    that the cap was invisible, so the churn looked like a fault.
    """
    sl  = slider_on(CONFIG)
    win = sl.win
    if sl.CONVENTIONAL:
        return
    saved_cut, saved_tol, saved_auto = win._qperp_cut, win._psi_tol, win._use_auto
    try:
        win._use_auto = True
        win._qperp_cut = None
        win._sb_depth.setValue(2)
        arc = None
        for row in range(len(win.full_reflist_6d)):
            arc = win._plot_arc(win.full_reflist_6d[row], '#00cccc')
            if arc is not None and len(arc._x_data) > 10:
                break
        assert arc is not None
        xs, ys = arc._x_data, arc._y_data
        pts = [(float(ys[i]), float(xs[i])) for i in (0, len(xs) // 2, len(xs) - 1)]
        win._psi_tol = 5.0
        win._run_geo_search(pts)

        # Enough candidates that the cap binds, and the label must say so.
        assert 'showing' in win._lbl_pick.text(), win._lbl_pick.text()
        assert len(win._geo_arcs) <= win.GEO_MAX_DRAWN

        # The pool itself stays nested all the way down — that is the property
        # the cut guarantees, and the one the cap does not.
        lo, hi = win._sl_qperp._min, win._sl_qperp._max
        pools = []
        for f in (0.0, 0.3, 0.6, 1.0):
            win._on_qperp_cut_changed(float(lo + f * (hi - lo)))
            pools.append(set(np.where(win._qperp_mask())[0]))
        assert all(a <= b for a, b in zip(pools, pools[1:])), 'pool must nest'
    finally:
        win._qperp_cut, win._psi_tol = saved_cut, saved_tol
        win._use_auto = saved_auto
        win._on_clear_picks()
        win._sb_depth.setValue(1)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('all passed')
