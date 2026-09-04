"""Right-click on a DMS line removes it — even when another line has no points.

Each selected reflection owns an arc whose on-detector points are cached on the
item (``_x_data`` / ``_y_data``) for hit-testing.  An arc can hold *no* points:
the reflection's line is off the plate at the current geometry (a scan load
moves plenty of them there), or the arc was created empty by a bulk add and the
overlay pass has not traced it yet.

``_nearest_arc_at`` used to take ``min()`` over those arrays unguarded, so one
empty arc raised ``ValueError: zero-size array to reduction operation minimum``
inside the mouse-click slot — and right-click then removed *nothing*, whichever
line was clicked.  ``_nearest_selectable`` (middle-click) always guarded this;
the removal path did not.

Run standalone:
    python -m DMSAnalysis.tests.test_arc_picking
or under pytest:
    pytest DMSAnalysis/tests/test_arc_picking.py
"""

import os

import numpy as np

from .gui_harness import slider_on

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'fit_conventional_tetragonal_PMN_PT_example.json')


def _fake_arc(win, ref, xs, ys):
    """A drawn arc as the picking code sees one: an item in ``_pick_items``
    with a reflection in ``_arc_to_6d`` and its points cached for hit-testing.
    Built directly so the test does not depend on which reflections happen to
    land on the plate at the config's geometry."""
    import pyqtgraph as pg
    arc = pg.ScatterPlotItem(x=np.asarray(xs, float), y=np.asarray(ys, float),
                             size=3, pen=None, brush=pg.mkBrush('#00cccc'))
    arc._x_data = np.asarray(xs, dtype=float)
    arc._y_data = np.asarray(ys, dtype=float)
    arc._colour = pg.mkColor('#00cccc')
    win._vb.addItem(arc)
    win._pick_items.append(arc)
    win._arc_to_6d[id(arc)] = np.asarray(ref)
    win._add_arc_to_list(np.asarray(ref), arc)
    return arc


def _scene_pos(win, x, y):
    from PyQt5 import QtCore
    return win._vb.mapViewToScene(QtCore.QPointF(float(x), float(y)))


def _right_click(win, scene_pos):
    """Drive the real handler, as ``sigMouseClicked`` does."""
    from PyQt5 import QtCore

    class Ev:
        def button(self):   return QtCore.Qt.RightButton
        def scenePos(self): return scene_pos

    win._on_scene_clicked(Ev())


def _clean(win):
    win._on_clear_picks()


def test_right_click_removes_the_line_under_it():
    win = slider_on(CONFIG).win
    _clean(win)
    a = _fake_arc(win, [0, 0, 2], np.arange(100, 200), np.full(100, 300.0))
    b = _fake_arc(win, [1, 1, 0], np.arange(100, 200), np.full(100, 900.0))
    assert win._arc_list.count() == 2

    _right_click(win, _scene_pos(win, 150, 900))
    assert win._arc_list.count() == 1
    assert id(b) not in win._arc_to_6d          # the clicked one went
    assert id(a) in win._arc_to_6d              # the other stayed
    _clean(win)


def test_an_empty_arc_does_not_block_removal():
    """The regression: an off-plate reflection in the list must not disarm
    right-click for the lines that *are* drawn."""
    win = slider_on(CONFIG).win
    _clean(win)
    empty = _fake_arc(win, [3, 3, 0], [], [])
    good  = _fake_arc(win, [0, 0, 2], np.arange(100, 200), np.full(100, 300.0))
    assert win._arc_list.count() == 2

    _right_click(win, _scene_pos(win, 150, 300))
    assert win._arc_list.count() == 1, 'right-click removed nothing'
    assert id(good) not in win._arc_to_6d
    assert id(empty) in win._arc_to_6d          # nothing is near it to remove
    _clean(win)


def test_click_far_from_every_line_removes_nothing():
    win = slider_on(CONFIG).win
    _clean(win)
    _fake_arc(win, [0, 0, 2], np.arange(100, 200), np.full(100, 300.0))
    _right_click(win, _scene_pos(win, 1200, 1500))
    assert win._arc_list.count() == 1
    _clean(win)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('all passed')
