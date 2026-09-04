"""A completed fit leaves a run record you can rerun from.

`_write_fit_snapshot` writes three files into
``Processing/<scannum>_dp<datapoint>_<YYYYMMDD-HHMMSS>/``:

* ``Result.txt``  — the solution, and the recipe: starting guess, optimiser,
  which slots were free, their bounds, and the sampling/peak settings
* ``IM_*.png``    — the detector image with the simulated DMS lines over it
* ``PLOT_*.svg``  — the integrated ROI curves as vector art

The record is written automatically when a fit finishes (after the post-fit
curve rebuild, so the curves in it are the refined ones); the manual button
writes the same three plus the code/config snapshot.  These tests drive the
writer over a small hand-made curve state, so they exercise the files without
running an optimisation.

Run standalone:
    python -m DMSAnalysis.tests.test_fit_snapshot
or under pytest:
    pytest DMSAnalysis/tests/test_fit_snapshot.py
"""

import os
import re
import glob
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .gui_harness import slider_on

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'configs', 'fit_conventional_tetragonal_PMN_PT_example.json')

NROI = 4          # two reflections, two ROIs each


def _curve_state(win):
    """Put a plausible built-curve state on the window: four ROIs, each with an
    experimental curve, a simulated one and both centres."""
    x = np.arange(0.0, 45.0)
    peak = lambda c: np.exp(-0.5 * ((x - c) / 4.0) ** 2) * 1000.0
    win._linedatax = [x.copy() for _ in range(NROI)]
    win._linedatay = [peak(20 + i) + 10.0 for i in range(NROI)]
    win._kernel    = np.zeros((8, 8, NROI))
    win._centres   = np.array([[20.0 + i] for i in range(NROI)])
    win._centre_override_rois = set()
    win._ref_6d_fit = np.array([[0, 0, 2], [1, 1, 0]])
    win._last_sim_lines = ([x.copy() for _ in range(NROI)],
                           [peak(21 + i) for i in range(NROI)])
    win._last_simcoefs = np.array([[1.0, 4.0, 21.0 + i] for i in range(NROI)])


def _fit_output(win, method='COBYLA'):
    ig = np.asarray(win.ig, dtype=float)
    start = ig.copy()
    if method != 'NoFit':
        start[10] += 12.0      # detector distance moved during the "fit"
    return {
        'opt': 0.1234, 'method': method, 'elapsed': 7.5, 'start_opt': 0.9,
        'res_x': np.array([1.0, 2.0]), 'inputarray': ig, 'dmsindex': None,
        'setup': {
            'start_ig': start, 'start_hkl': win._hkl.copy(),
            'start_psi': float(win._psi), 'method': method,
            'free_slots': [10, 14], 'all_slots': [0, 10, 14],
            'bounds': [(4.0, 4.1), (5000.0, 5100.0), (8.0, 8.5)],
            'n_parallel_starts': 4, 'numsteps': 120, 'tolerance': 1e-6,
            'peak_method': 'gauss', 'curve_method': 'sweep',
            'roi_width': 45.0, 'detoptimize': True, 'energyopt': False,
        },
    }


def _write(win, method='COBYLA', **kw):
    """Write a snapshot into a throwaway cwd; return (dir, [filenames])."""
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix='fitsnap_')
    try:
        os.chdir(tmp)
        outpath = win._write_fit_snapshot(_fit_output(win, method), **kw)
    finally:
        os.chdir(cwd)
    return outpath, sorted(os.listdir(outpath))


def test_snapshot_writes_three_files_in_a_scan_dp_time_folder():
    win = slider_on(CONFIG).win
    _curve_state(win)
    outpath, files = _write(win)

    name = os.path.basename(outpath)
    assert re.match(r'^%s_dp%d_\d{8}-\d{6}_COBYLA$'
                    % (win._scannum, int(win._datapoint)), name), name
    assert os.path.basename(os.path.dirname(outpath)) == 'Processing'

    stem = '%s_dp%d' % (win._scannum, int(win._datapoint))
    assert files == ['IM_%s.png' % stem, 'PLOT_%s.svg' % stem, 'Result.txt'], files
    for f in files:
        assert os.path.getsize(os.path.join(outpath, f)) > 0, f


def test_the_svg_holds_one_panel_per_roi():
    win = slider_on(CONFIG).win
    _curve_state(win)
    outpath, files = _write(win)
    svg = os.path.join(outpath, [f for f in files if f.endswith('.svg')][0])

    root = ET.parse(svg).getroot()
    assert root.tag.endswith('svg')
    text = open(svg).read()
    # every ROI's title, and both curves in each panel
    assert text.count('axes_') >= NROI, 'expected one axes group per ROI'
    for label in ('[0 0 2]', '[1 1 0]'):
        assert label.replace(' ', '') in text.replace(' ', ''), label


def test_result_txt_carries_the_rerun_recipe():
    win = slider_on(CONFIG).win
    _curve_state(win)
    outpath, _ = _write(win)
    txt = open(os.path.join(outpath, 'Result.txt')).read()

    assert 'scan %s' % win._scannum in txt
    assert 'datapoint %d' % int(win._datapoint) in txt
    # the outcome
    assert 'chi2 = 0.1234' in txt and '[COBYLA]' in txt
    # the recipe: method, free parameters, their bounds, the sampling settings
    assert 'fit setup (rerun with this)' in txt
    assert 'method       COBYLA' in txt
    assert 'detdist' in txt and 'energy' in txt
    assert '5000.00000000' in txt and '5100.00000000' in txt
    assert 'points 120' in txt and 'tolerance 1e-06' in txt
    assert 'ROI width 45 px' in txt and 'peaks gauss' in txt
    # the starting guess, and what the fit did to it
    start_line = [l for l in txt.splitlines()
                  if l.strip().startswith('initial_guess')]
    assert len(start_line) == 2, 'start and refined guesses both expected'
    assert 'moved by the fit (start → refined)' in txt
    assert 'per-ROI centres' in txt


def test_extras_add_the_reproducibility_snapshot():
    win = slider_on(CONFIG).win
    _curve_state(win)
    outpath, files = _write(win, extras=True)
    assert 'slider.py' in files and 'ts_quasi.py' in files
    assert 'res.x.txt' in files
    assert any(f.startswith('config_') and f.endswith('.json') for f in files), files


def test_no_fit_is_offered_and_records_itself_as_such():
    """'No Fit' runs no optimiser but still produces the run record — the way
    to export the current guess.  Its folder is tagged like any other method,
    and Result.txt says plainly that nothing was refined."""
    sl  = slider_on(CONFIG)
    win = sl.win
    assert 'NoFit' in sl.algo_methods
    assert len(sl.algo_display) == len(sl.algo_methods)

    _curve_state(win)
    outpath, files = _write(win, method='NoFit')
    assert os.path.basename(outpath).endswith('_NoFit'), outpath
    assert len(files) == 3, files

    txt = open(os.path.join(outpath, 'Result.txt')).read()
    assert '[NoFit]' in txt
    assert 'no optimiser ran' in txt
    assert 'would refine' in txt
    assert 'started from' not in txt          # there was no starting point to beat
    assert 'initial_guess = np.array([' in txt


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('all passed')
