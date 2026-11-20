"""The ROI panel's *Clear centres* button drops every manual ROI centre.

Right-click on a ROI panel assigns its centre (`_set_centre_override`); the
button puts each overridden ROI back on the centre the build located
(`imcoeffs[:, 2]`) in both the window and the fit engine, and drops overrides
restored from a session but not yet applied, so the next build does not bring
them back.
"""

import json
import os

import numpy as np

from .gui_harness import slider_on

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'fit_conventional_tetragonal_PMN_PT_example.json')


def _wait(win, app):
    w = win._build_worker
    while w is not None and w.isRunning():
        w.wait(100)
        app.processEvents()
    for _ in range(5):
        app.processEvents()


def test_clear_centres_restores_the_located_peaks():
    from ..qt import QtWidgets
    sl = slider_on(CONFIG)
    win = sl.win
    app = QtWidgets.QApplication.instance()
    saved_pm = win._peak_method
    try:
        if win._peak_method == 'ridge':
            win._peak_combo.setCurrentIndex(win._peak_combo.findData('gauss'))
        _wait(win, app)
        win._on_clear_picks()
        win._apply_reflections(json.load(open(CONFIG))['crystal']['reflist_hkl'])
        assert win._on_build_curves()
        _wait(win, app)
        auto = np.array(win._imcoeffs[:, 2], dtype=float)
        chi_auto = win._lbl_resid.text()

        win._set_centre_override(0, auto[0] + 7.0)
        win._set_centre_override(1, 3.0)
        assert win._centre_override_rois == {0, 1}
        assert win._session_dict()['manual_centres']
        win._pending_centre_overrides = {5: 1.0}   # restored, not yet built

        win._btn_clear_centres.click()
        assert win._centre_override_rois == set()
        assert win._pending_centre_overrides == {}
        assert np.allclose(win._centres[:2, 0], auto[:2], equal_nan=True)
        assert np.allclose(np.asarray(win._fit_dms.centres)[:2, 0], auto[:2],
                           equal_nan=True)
        assert win._session_dict()['manual_centres'] == {}
        assert win._lbl_resid.text() == chi_auto
        assert 'Cleared 3' in win._status.text()

        win._btn_clear_centres.click()
        assert 'No manual centres' in win._status.text()
    finally:
        win._peak_combo.setCurrentIndex(win._peak_combo.findData(saved_pm))
