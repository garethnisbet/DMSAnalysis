"""The slider's FitWorker runs its multi-start fits in worker processes.

The starts used to run on threads.  Once the objective stopped being dominated
by GIL-free C (the old full-frame convolve), threads took turns, so each start
now runs ``ts.run_scaled_start`` in a joblib worker process.  These tests drive
the real FitWorker — the Parallel call, the StopFlag, the collection of each
start's best point — rather than a stand-in:

* a finished fit reports exactly the best point the same starts reach when run
  in this process, one after another (a start is a pure function of the engine
  and its starting point, so where it runs cannot change what it finds);
* Stop ends a running fit promptly, from inside the worker processes.

Run standalone:
    python -m DMSAnalysis.tests.test_fitworker_processes
or under pytest:
    pytest DMSAnalysis/tests/test_fitworker_processes.py
"""

import os
import time

import numpy as np

from .. import ts_quasi as ts
from .gui_harness import slider_on
from .test_fit_objective_speedups import _scene

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'fit_conventional_tetragonal_PMN_PT_example.json')


def _worker(sl, n_starts=2):
    dms, xs = _scene()
    # As _do_fit does before handing the engine to FitWorker, which scores its
    # final candidates at the slider's numsteps.
    dms.hkllistrange[2] = sl.numsteps
    reduced = np.asarray(xs[0], dtype=float)
    steps = np.full(len(reduced), 1e-2)
    bounds = list(zip(reduced - 1.0, reduced + 1.0))
    w = sl.FitWorker(dms, reduced, bounds, 'COBYLA', n_starts, steps=steps)
    return w, dms, reduced, steps


def test_process_starts_find_what_in_process_starts_find():
    sl = slider_on(CONFIG)
    w, dms, reduced, steps = _worker(sl)
    got = {}
    w.done.connect(lambda r: got.setdefault('done', r))
    w.error.connect(lambda msg, t: got.setdefault('error', msg))
    w.run()                                  # synchronously, in this thread
    assert 'error' not in got, got.get('error')
    res = got['done']

    # The same two starts, the same options, run here one after another.
    ndim = len(reduced)
    starts = sl.perturbed_starts(2, ndim, np.random.default_rng(42))
    opts = dict(ts.tripfit_minimizer_options('COBYLA', sl.tolerance), rhobeg=1.0)
    ref = [ts.run_scaled_start(dms, reduced, np.arange(ndim), reduced, steps, s,
                               'COBYLA', sl.tolerance, options=opts)
           for s in starts]
    best = min(f for _, f, _ in ref)
    assert res['opt'] == best, (res['opt'], best)
    assert res['opt'] <= res['start_opt']
    assert res['method'] == 'COBYLA'


def test_stop_ends_a_running_fit_promptly():
    sl = slider_on(CONFIG)
    from ..qt import QtCore
    w, _, _, _ = _worker(sl, n_starts=2)
    got = {}
    # Direct connections: the slots run in the worker thread, so the signals
    # arrive without pumping the event loop — which would also run the
    # slider's queued startup task and its modal missing-scan prompt.
    direct = QtCore.Qt.ConnectionType.DirectConnection
    w.done.connect(lambda r: got.setdefault('done', True), direct)
    w.stopped.connect(lambda t: got.setdefault('stopped', t), direct)
    w.start()
    time.sleep(0.5)                          # let the worker processes get going
    t = time.perf_counter()
    w.stop()
    assert w.wait(60_000), 'the fit did not end after Stop'
    took = time.perf_counter() - t
    assert 'stopped' in got and 'done' not in got, got
    assert took < 30, 'Stop took %.1fs to end the fit' % took
    w.stop()                                 # a late Stop must be harmless


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
