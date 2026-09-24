"""Import ``slider.py`` headlessly, once, for the GUI tests.

``slider.py`` builds and shows its window at import and then blocks in
``app.exec()``, so the launch has to be defused first: stub the event loop,
``sys.exit`` and the modal dialogs (the "restore previous session" prompt would
otherwise block forever on the first ``processEvents``).

The module is a singleton — the first caller's config is the one the window is
built on, and later callers get that same window.  Tests that need a particular
config must therefore not assume they are first; drive the window through
``_do_load_scan`` / the picking API instead.
"""

import os
import sys


_SLIDER = None

# Qt's default offscreen screen is 800x800 (Qt6) / 800x600 (Qt5).  Qt6's
# restoreGeometry clamps a window to its screen, so on that screen the 1900 px
# slider window cannot be restored as saved.  Qt6 reads a screen size from this
# file; Qt5 ignores it and does not clamp.
_SCREEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'offscreen_screen.json')


def offscreen():
    """Run Qt headless, on a screen the size of a real monitor."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen:configfile=' + _SCREEN)


def slider_on(cfg_path):
    """Return the imported ``DMSAnalysis.slider`` module, window built."""
    global _SLIDER
    if _SLIDER is not None:
        return _SLIDER
    offscreen()
    from ..qt import QtWidgets
    QtWidgets.QApplication.exec = lambda self: 0
    QtWidgets.QMessageBox.question = staticmethod(
        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No)
    argv, exit_ = sys.argv, sys.exit
    sys.argv = ['slider', cfg_path]
    sys.exit = lambda *a, **k: None
    try:
        from .. import slider as sl
    finally:
        sys.argv, sys.exit = argv, exit_
    _SLIDER = sl
    return sl
