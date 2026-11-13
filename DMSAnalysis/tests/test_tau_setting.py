"""τ for the 6D cut-and-projection is settable in the slider, as a formula.

The τ field in the Crystal type box takes a number or an arithmetic formula
(55/34, 0.5+0.5*5**0.5, (1+sqrt(5))/2, tau) and re-projects everything built
from the projection.  Only shown in the icosahedral modes; saved in sessions.
"""

import os

import numpy as np
import pytest

from .gui_harness import slider_on
from .. import ts_quasi as ts

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'SelectedLines_GoodInitialGuess.json')


@pytest.mark.parametrize('text, value', [
    ('55/34', 55 / 34),
    ('0.5+0.5*5**0.5', ts.TAU),
    ('(1+sqrt(5))/2', ts.TAU),
    ('tau', ts.TAU),
    ('-(-1.5)', 1.5),
    (' 89/55 ', 89 / 55),
])
def test_formulas_evaluate(text, value):
    assert ts.eval_number_expr(text) == pytest.approx(value, rel=1e-15)


@pytest.mark.parametrize('text', [
    '__import__("os")', 'open("x")', 'a+1', '1/0', '2**1000', '', '1,2',
    '[1]', 'sqrt(-1)', 'True',
])
def test_anything_else_is_refused(text):
    with pytest.raises(ValueError):
        ts.eval_number_expr(text)


@pytest.fixture
def ico_window():
    sl = slider_on(CONFIG)
    win = sl.win
    was, expr = sl.bravais, sl.tau_expr
    if sl.CONVENTIONAL:
        win._set_crystal_system('icosahedral')
    yield sl, win
    win._set_tau(expr)
    if sl.bravais != was:
        win._set_crystal_system(was)


def test_setting_tau_reprojects_the_reflections(ico_window):
    sl, win = ico_window
    assert win._tau_row.isVisibleTo(win)
    win._tau_edit.setText('0.5+0.5*5**0.5')
    win._on_tau_entered()
    assert sl.tau == ts.TAU
    par, perp = ts.Projection6dArrayApproximant(
        np.asarray(win.full_reflist_6d), ts.TAU).reflection_6d()
    assert np.allclose(win.full_reflist, par)
    assert np.allclose(win.full_reflist2, perp)
    assert np.allclose(win._dms_full.reflist, par)
    assert win._kernel is None and win._fit_dms is None
    assert win._cfg['crystal']['tau_expr'] == '0.5+0.5*5**0.5'
    assert win._session_dict()['tau_expr'] == '0.5+0.5*5**0.5'


def test_a_bad_formula_changes_nothing(ico_window):
    sl, win = ico_window
    win._set_tau('55/34')
    before = np.array(win.full_reflist, copy=True)
    win._tau_edit.setText('55/')
    win._on_tau_entered()
    assert sl.tau == 55 / 34
    assert np.array_equal(win.full_reflist, before)
    assert 'τ not changed' in win._status.text()
    win._tau_edit.setText('-1')
    win._on_tau_entered()
    assert sl.tau == 55 / 34


def test_hidden_for_a_conventional_crystal(ico_window):
    sl, win = ico_window
    win._set_crystal_system('cubic')
    assert not win._tau_row.isVisibleTo(win)
    win._set_crystal_system('icosahedral')
    assert win._tau_row.isVisibleTo(win)


def test_session_restores_tau(ico_window):
    sl, win = ico_window
    win._set_tau('tau')
    data = dict(win._session_dict(), curves_built=False)
    win._set_tau('55/34')
    win._restore_from_dict(data)
    assert sl.tau == ts.TAU and win._tau_edit.text() == 'tau'
    # a session from before the option was projected with 55/34
    data.pop('tau_expr')
    win._restore_from_dict(data)
    assert sl.tau == 55 / 34
