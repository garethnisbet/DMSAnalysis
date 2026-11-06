"""Load reflections takes every reflection in the file.

A hand-edited list — reflections appended to ``ref_6d`` with
``ref_6d_checked`` left at its old length — used to load only as many
reflections as there were check states, because ``_apply_reflections`` zipped
the two lists.  Missing check states now default to checked.  A list whose index
length does not match the crystal type (3 for conventional, 6 for icosahedral)
is refused with a message instead of loading nothing.

Run standalone:
    python -m DMSAnalysis.tests.test_load_reflections
or under pytest:
    pytest DMSAnalysis/tests/test_load_reflections.py
"""

import os
import json
import tempfile

from .test_det_orientation import _config


def _load(win, data):
    from ..qt import QtWidgets
    path = os.path.join(tempfile.mkdtemp(), 'refl.json')
    with open(path, 'w') as fh:
        json.dump(data, fh)
    orig = QtWidgets.QFileDialog.getOpenFileName
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (path, ''))
    try:
        win._on_load_reflections()
    finally:
        QtWidgets.QFileDialog.getOpenFileName = orig
    return [win._arc_list.item(i) for i in range(win._arc_list.count())]


def test_short_checked_list_loads_every_reflection():
    from .gui_harness import slider_on
    from ..qt import QtCore
    sl = slider_on(_config(tempfile.mkdtemp()))
    if not sl.CONVENTIONAL:
        return          # the shared window was built on a 6D config
    refs = [[1, 1, 1], [0, 1, 1], [1, 0, 2], [2, 0, 0], [1, 1, 3], [1, 1, 3]]
    items = _load(sl.win, {'ref_6d': refs, 'ref_6d_checked': [True, False]})
    assert len(items) == 5, [i.text() for i in items]     # duplicate merged
    states = [i.checkState() == QtCore.Qt.CheckState.Checked for i in items]
    assert states == [True, False, True, True, True], states
    assert 'Loaded 5 reflections' in sl.win._status.text()
    assert 'duplicate' in sl.win._status.text()


def test_wrong_index_length_is_refused_with_a_message():
    from .gui_harness import slider_on
    sl = slider_on(_config(tempfile.mkdtemp()))
    if not sl.CONVENTIONAL:
        return
    _load(sl.win, {'ref_6d': [[1, 1, 1]]})
    n = sl.win._arc_list.count()
    _load(sl.win, {'ref_6d': [[0, -1, -1, 5, -2, 0]]})
    assert sl.win._arc_list.count() == n              # selection left alone
    assert 'Not loaded' in sl.win._status.text()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('all passed')
