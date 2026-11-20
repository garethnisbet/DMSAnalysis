"""The number of cores the multistart optimisers use is set in the slider.

`n_parallel_starts` is both the number of perturbed starts and joblib's
`n_jobs`.  It used to come only from the config (default 4); the *cores*
spinbox next to the optimiser combo now sets it, clamped to the cores this
process may run on.  Saved in sessions and exported for fit.py.
"""

import os

from .gui_harness import slider_on

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'SelectedLines_GoodInitialGuess.json')


def test_clamp_cores():
    sl = slider_on(CONFIG)
    top = sl.cpu_cores()
    assert top >= 1
    assert sl.clamp_cores(0) == 1
    assert sl.clamp_cores(-3) == 1
    assert sl.clamp_cores(10 ** 6) == top
    assert sl.clamp_cores('junk') == min(4, top)
    assert sl.clamp_cores(None) == min(4, top)


def test_spinbox_sets_starts_session_and_export():
    sl = slider_on(CONFIG)
    win = sl.win
    start = sl.n_parallel_starts
    try:
        assert win._sb_cores.maximum() == sl.cpu_cores()
        assert win._sb_cores.value() == sl.n_parallel_starts
        win._sb_cores.setValue(1)
        assert sl.n_parallel_starts == 1
        assert win._session_dict()['n_parallel_starts'] == 1
        assert win._build_workflow_config()['computation']['n_parallel_starts'] == 1

        # a session restores its own setting ...
        data = win._session_dict()
        data['n_parallel_starts'] = sl.cpu_cores()
        win._restore_from_dict(data)
        assert sl.n_parallel_starts == sl.cpu_cores()
        assert win._sb_cores.value() == sl.cpu_cores()

        # ... one from a bigger machine is clamped, an older one keeps ours
        data['n_parallel_starts'] = 10 ** 6
        win._restore_from_dict(data)
        assert sl.n_parallel_starts == sl.cpu_cores()
        win._sb_cores.setValue(1)
        data.pop('n_parallel_starts')
        win._restore_from_dict(data)
        assert sl.n_parallel_starts == 1
    finally:
        win._sb_cores.setValue(start)
