"""Per-parameter fit locks: the slider tick boxes in tripslider and
computation.locked, which the GUI writes and the batch tripfit app honours.

A locked parameter must come out of a fit exactly where it went in, while the
unlocked ones still refine — checked through the GUI's real background worker
and through the batch app.

Run standalone:
    python -m DMSAnalysis.tests.test_tripfit_locks
or under pytest.
"""

import copy
import json
import time

import numpy as np

from .. import ts_quasi as ts
from .test_tripfit_quasi import PMN_CONFIG, _cfg, _params, _run_batch, _select, _window

A, A11, A13, A32 = 0, 6, 8, 13          # parameter slots


def test_locked_names_map_to_slots_and_positions():
    assert ts.tripfit_locked_slots(['a13', 'a']) == [A, A13]
    assert ts.tripfit_locked_slots('alpha') == [3]
    assert ts.tripfit_locked_names({A13, A}) == ['a', 'a13']
    # reduced vector of icosahedral is [a, a11..a33]: a13 sits at position 3
    assert ts.tripfit_fit_positions('icosahedral', [A, A13]) == [1, 2, 4, 5, 6, 7, 8, 9]
    assert ts.tripfit_fit_positions('rhombohedral', [A]) == [1]
    try:
        ts.tripfit_locked_slots(['zeta'])
    except ValueError:
        return
    raise AssertionError('an unknown parameter name was accepted')


def test_boxes_are_on_the_refinable_sliders_only():
    win = _window(_cfg())
    try:
        assert all(fs._fit_chk is not None and fs.is_fit_enabled()
                   for fs in win._param_sliders.values())
        assert all(fs._fit_chk is None for fs in win._hkl_sliders)
        _select(win, 'icosahedral_fixed_a')      # a is shown but held by the type
        assert win._param_sliders[A]._fit_chk is None
        assert win._param_sliders[A11]._fit_chk is not None
    finally:
        win.close()


def test_fit_moves_only_the_unlocked_parameters():
    from PyQt5 import QtWidgets
    win = _window(_cfg())
    app = QtWidgets.QApplication.instance()
    try:
        # break the triples, then lock a and a11 before fitting
        win._params[A] += 0.005
        win._params[A13] += 0.003
        win._params[A32] -= 0.003
        win._param_sliders[A]._fit_chk.setChecked(False)
        win._param_sliders[A11]._fit_chk.setChecked(False)
        win._algo_combo.setCurrentText('L-BFGS-B')
        win._sp_fit.setValue(400)
        win._fit_res = 400
        for g in win._groups:
            g['tf'].set_params(win._params)
        start = win._params.copy()
        before = win._fit_objective(win._reduced())

        win._on_fit()
        worker = win._fit_worker
        assert worker is not None
        assert worker.wait(300000)
        for _ in range(500):                     # deliver the queued done signal
            app.processEvents()
            if win._fit_worker is None:
                break
            time.sleep(0.01)
        assert win._fit_worker is None

        assert win._params[A] == start[A]
        assert win._params[A11] == start[A11]
        assert win._params[A13] != start[A13]
        win._fit_res = 400
        after = win._fit_objective(win._reduced())
        assert after < before * 1e-2, (before, after)
    finally:
        win.close()


def test_fit_refuses_when_everything_is_locked():
    win = _window(_cfg())
    try:
        _select(win, 'cubic_no_strain')          # a is the only free parameter
        win._param_sliders[A]._fit_chk.setChecked(False)
        win._on_fit()
        assert win._fit_worker is None
        assert 'locked' in win._status.currentMessage()
    finally:
        win.close()


def test_locks_survive_rebuilds_and_round_trip_through_the_config():
    win = _window(_cfg())
    try:
        win._param_sliders[A13]._fit_chk.setChecked(False)
        _select(win, 'icosahedral_fixed_a')      # slider panel is rebuilt
        assert not win._param_sliders[A13].is_fit_enabled()
        _select(win, 'icosahedral')
        saved = win._export_config()
    finally:
        win.close()
    assert saved['computation']['locked'] == ['a13']
    win = _window(saved)
    try:
        assert not win._param_sliders[A13].is_fit_enabled()
        assert win._param_sliders[A11].is_fit_enabled()
    finally:
        win.close()

    win = _window(_cfg(PMN_CONFIG))
    try:
        assert 'locked' not in win._export_config()['computation']
    finally:
        win.close()


def test_batch_app_honours_locked_parameters():
    cfg = _cfg()
    cfg['flags']['fit'] = 1
    cfg['computation'].update({'opt_method': 'L-BFGS-B', 'resolution': 400,
                               'locked': ['a', 'a13']})
    cfg['crystal']['initial_guess'] = [v + 0.005 if i < 3 else v for i, v in
                                       enumerate(cfg['crystal']['initial_guess'])]
    cfg['crystal']['phason'][2] += 0.003      # a13
    cfg['crystal']['phason'][7] -= 0.003      # a32
    start = _params(cfg)[ts.tripfit_free_slots('icosahedral')]

    out = _run_batch(cfg)
    assert 'locked     : a, a13' in out
    text = out.split('reduced x  :')[1].split('lattice')[0]
    x = np.array(text.replace('[', ' ').replace(']', ' ').split(), dtype=float)
    assert np.isclose(x[0], start[0], atol=1e-7)       # a
    assert np.isclose(x[3], start[3], atol=1e-7)       # a13
    assert not np.isclose(x[8], start[8], atol=1e-7)   # a32 refined


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print('ok  %s' % fn.__name__)
    print('\n%d checks passed' % len(fns))


if __name__ == '__main__':
    _run()
