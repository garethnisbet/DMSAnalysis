"""The panel layout survives a restart.

The window is three panels — detector image | controls | integrated curves —
divided by one splitter.  Where the user drags those dividers is part of how
they work, not part of what they are analysing, so it is kept in Qt's own
per-user settings (``~/.config/DMSAnalysis/slider.conf`` on Linux) rather than
in the auto-saved session, and comes back whether or not the session is
resumed.  It is written on every drag as well as on exit, so a killed app does
not lose it.

Run standalone:
    python -m DMSAnalysis.tests.test_layout_persistence
or under pytest:
    pytest DMSAnalysis/tests/test_layout_persistence.py
"""

import os
import tempfile

from PyQt5 import QtCore

from .gui_harness import slider_on

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'fit_conventional_tetragonal_PMN_PT_example.json')


def _isolated_settings(win):
    """Point the window's layout store at a throwaway ini file, so a test run
    never reads or writes the user's real settings."""
    tmp = tempfile.mkdtemp(prefix='dmslayout_')
    st  = QtCore.QSettings(os.path.join(tmp, 'slider.ini'),
                           QtCore.QSettings.IniFormat)
    win._layout_settings = lambda: st
    return st


def test_divider_positions_survive_a_restart():
    win = slider_on(CONFIG).win
    st  = _isolated_settings(win)
    try:
        win._splitter.setSizes([500, 700, 300])
        want = win._splitter.sizes()          # what the minimum widths allow
        win._save_layout()
        st.sync()
        assert os.path.exists(st.fileName()), 'nothing was written'

        # the next launch starts on the defaults, then restores
        win._splitter.setSizes([820, 640, 420])
        assert win._splitter.sizes() != want
        win._restore_layout()
        assert win._splitter.sizes() == want, win._splitter.sizes()
    finally:
        del win._layout_settings


def test_a_drag_is_saved_without_waiting_for_exit():
    """Saving only in closeEvent would lose the layout whenever the app is
    killed, which is how a GUI on a beamline usually ends."""
    win = slider_on(CONFIG).win
    assert win._splitter.receivers(win._splitter.splitterMoved) > 0

    st = _isolated_settings(win)
    try:
        win._splitter.setSizes([600, 700, 400])
        want = win._splitter.sizes()
        # what the signal does, without synthesising a mouse drag
        win._save_layout()
        st.sync()
        win._splitter.setSizes([820, 640, 420])
        win._restore_layout()
        assert win._splitter.sizes() == want
    finally:
        del win._layout_settings


def test_nothing_saved_yet_leaves_the_defaults_alone():
    win = slider_on(CONFIG).win
    st  = _isolated_settings(win)          # empty store
    try:
        win._splitter.setSizes([700, 700, 300])
        before = win._splitter.sizes()
        win._restore_layout()              # must not blow up, must not move
        assert win._splitter.sizes() == before
    finally:
        del win._layout_settings


def test_a_window_off_every_screen_is_not_restored_there():
    """A geometry saved on a monitor that is no longer attached must not bring
    the window back where it cannot be seen."""
    win = slider_on(CONFIG).win
    assert win._on_a_screen()
    win.move(-30000, -30000)
    if win.frameGeometry().x() < -10000:    # the platform honoured the move
        assert not win._on_a_screen()
    win.move(40, 40)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('all passed')
