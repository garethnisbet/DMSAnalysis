"""Import ``slider.py`` headlessly, once, for the GUI tests.

``slider.py`` builds and shows its window at import and then blocks in
``app.exec_()``, so the launch has to be defused first: stub the event loop,
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


def slider_on(cfg_path):
    """Return the imported ``DMSAnalysis.slider`` module, window built."""
    global _SLIDER
    if _SLIDER is not None:
        return _SLIDER
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.exec_ = lambda self: 0
    QtWidgets.QMessageBox.question = staticmethod(
        lambda *a, **k: QtWidgets.QMessageBox.No)
    argv, exit_ = sys.argv, sys.exit
    sys.argv = ['slider', cfg_path]
    sys.exit = lambda *a, **k: None
    try:
        from .. import slider as sl
    finally:
        sys.argv, sys.exit = argv, exit_
    _SLIDER = sl
    return sl
