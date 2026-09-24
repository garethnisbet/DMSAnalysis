"""Quasicrystal support in the tripfit engine, the batch app and tripslider.

A quasicrystal triple has to be handled exactly the way ``slider.py`` handles a
quasicrystal: 6D indices projected with the slider's tau approximant, a cubic
cell, and reflection vectors ``par + M·perp``.  The engine check therefore
compares against a conventional cubic engine fed those very vectors, instead of
against numbers this file would have to trust.

Run standalone:
    python -m DMSAnalysis.tests.test_tripfit_quasi
or under pytest.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from .. import ts_quasi as ts

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(PKG, 'configs', 'tripfit_icosahedral_AlPdMn_example.json')
PMN_CONFIG = os.path.join(PKG, 'configs', 'tripfit_rhombohedral_PMN_PT_example.json')


def _cfg(path=CONFIG):
    with open(path) as fh:
        return json.load(fh)


def _params(cfg):
    cr = cfg['crystal']
    return ts.tripfit_params(np.r_[cr['initial_guess'][:6], cr.get('phason', [0.0] * 9)])


def _engine(cfg, group=0, bravais=None, resolution=400):
    geo, gc = cfg['geometry'], cfg['intersections'][group]
    return ts.tripfit(np.array([geo['hkl']], dtype=float),
                      np.array(gc['reflist'], dtype=float), geo['azir'],
                      resolution, bravais or cfg['computation']['bravais'],
                      gc['energy'], 0.0, params=_params(cfg))


# ── engine ───────────────────────────────────────────────────────────────────────
def test_free_slots_match_the_image_fit_modes():
    phason = list(range(6, 15))
    assert ts.tripfit_free_slots('icosahedral') == [0] + phason
    assert ts.tripfit_free_slots('icosahedral_fixed_a') == phason
    assert ts.tripfit_free_slots('cubic_no_strain') == [0]
    for system in ts.CONVENTIONAL_SYSTEMS:        # conventional vectors unchanged
        assert ts.tripfit_free_slots(system) == ts.lattice_free_slots(system)


def test_tau_is_the_sliders():
    assert ts.TAU_APPROX == 55 / 34.


def test_quasi_lines_are_the_lines_of_the_sliders_vectors():
    cfg = _cfg()
    p = _params(cfg)
    q = _engine(cfg)
    q.kosselcalc(p[ts.tripfit_free_slots('icosahedral')])
    # slider.build_reflist_from_6d + the PhasonDistoArray step of dmsfit_ico_hkl
    ref6 = np.array(cfg['intersections'][0]['reflist'])
    par, perp = ts.Projection6dArrayApproximant(ref6, ts.TAU_APPROX).reflection_6d()
    vec = ts.PhasonDistoArray(np.array(par), np.array(perp), list(p[6:15])).qe1()
    c = ts.tripfit(q.hkl, np.matrix(vec), q.azir, q.resolution, 'cubic',
                   q.energy, 0.0)
    c.kosselcalc([p[0]])
    for s in ('st0', 'st1', 'st2'):
        np.testing.assert_array_equal(getattr(q, s), getattr(c, s))


def test_fixed_a_holds_a_and_no_strain_ignores_the_phason():
    cfg = _cfg()
    p = _params(cfg)
    ico = _engine(cfg)
    ico.kosselcalc(p[[0] + list(range(6, 15))])
    fixed = _engine(cfg, bravais='icosahedral_fixed_a')
    fixed.kosselcalc(p[6:15])                    # a comes from params
    np.testing.assert_array_equal(fixed.st0, ico.st0)

    strained = _engine(cfg, bravais='cubic_no_strain')
    strained.kosselcalc([p[0]])
    unstrained = copy.deepcopy(cfg)
    unstrained['crystal']['phason'] = [0.0] * 9
    ref = _engine(unstrained)
    ref.kosselcalc(np.r_[p[0], np.zeros(9)])
    np.testing.assert_array_equal(strained.st0, ref.st0)


def test_wrong_index_count_is_an_error_not_a_penalty():
    cfg = _cfg()
    for bravais, refl in (('icosahedral', np.eye(3)),
                          ('cubic', np.ones((3, 6)))):
        try:
            ts.tripfit_reflections(refl, bravais)
        except ValueError:
            continue
        raise AssertionError('%s accepted a %s reflist' % (bravais, refl.shape))
    try:
        ts.tripfit(np.array([cfg['geometry']['hkl']]), np.eye(3),
                   cfg['geometry']['azir'], 100, 'icosahedral', 6.3, 0.0)
    except ValueError:
        return
    raise AssertionError('the engine accepted Miller indices in a quasicrystal mode')


def test_example_triples_meet_and_a_fit_recovers_them():
    cfg = _cfg()
    p = _params(cfg)
    x0 = p[ts.tripfit_free_slots('icosahedral')]
    engines = [_engine(cfg, g, resolution=1000)
               for g in range(len(cfg['intersections']))]
    for tf in engines:
        assert tf.fit(x0) < 1e-8, tf.fit(x0)

    # Move a and two phason elements off the refinement and let the shared
    # optimiser bring all three triples back together.  The spread is far more
    # sensitive to the phason than to a (+0.01 A on a alone barely moves it).
    start = x0.copy()
    start[0] += 0.005
    start[3] += 0.003          # a13
    start[8] -= 0.003          # a32

    def total(x):
        return sum(tf.fit(x) for tf in engines)

    before = total(start)
    res = ts.run_tripfit_optimiser(total, start, 'L-BFGS-B',
                                   list(zip(start - 0.012, start + 0.012)), 1e-17)
    after = total(np.atleast_1d(res.x))
    assert before > 1e-7, before          # the perturbation really broke them
    assert after < before * 1e-3, (before, after)


# ── batch app ────────────────────────────────────────────────────────────────────
def _run_batch(cfg):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'cfg.json')
        with open(path, 'w') as fh:
            json.dump(cfg, fh)
        env = dict(os.environ, MPLBACKEND='Agg')
        out = subprocess.run([sys.executable, '-m', 'DMSAnalysis.tripfit', path],
                             cwd=os.path.dirname(PKG), env=env, timeout=300,
                             capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_batch_app_runs_a_quasicrystal_config():
    out = _run_batch(_cfg())
    assert 'phason' in out
    resid = float(out.split('residual   :')[1].split()[0])
    assert resid < 1e-7, out


# ── GUI ──────────────────────────────────────────────────────────────────────────
def _window(cfg):
    from .gui_harness import offscreen
    offscreen()
    from ..qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from .. import tripslider
    win = tripslider.TripSlider(copy.deepcopy(cfg), CONFIG)
    app.processEvents()
    return win


def _select(win, bravais):
    base, suffix = win._split_bravais(bravais)
    win._crystal_combo.setCurrentIndex(win._crystal_combo.findData(base))
    assert win._bravais == bravais


def test_gui_quasicrystal_mode_mirrors_the_slider():
    cfg = _cfg()
    win = _window(cfg)
    try:
        assert win._crystal_combo.currentText() == 'Icosahedral (quasicrystal)'
        assert sorted(win._param_sliders) == [0] + list(range(6, 15))
        # a clipped readout drops the sign of a negative phason element
        from ..qt import QtGui
        for fs in win._param_sliders.values():
            text = fs._fmt % (-abs(fs.val))
            fm = QtGui.QFontMetrics(fs._vl.font())
            assert fs._vl.width() >= fm.horizontalAdvance(text), text
        assert not win._pc_combo.isEnabled()
        assert '6D' in win._table.horizontalHeaderItem(1).text()
        groups, err = win._read_table()
        assert not err and all(len(r) == 6 for g in groups for r in g['reflist'])
        assert win._fit_objective(win._reduced()) < 1e-7

        _select(win, 'icosahedral_fixed_a')
        assert sorted(win._param_sliders) == [0] + list(range(6, 15))
        assert len(win._reduced()) == 9
        assert win._fit_objective(win._reduced()) < 1e-7    # same triples kept

        _select(win, 'cubic_no_strain')
        assert sorted(win._param_sliders) == [0]
    finally:
        win.close()


def test_gui_switching_indexing_family_sets_triples_aside():
    cfg = _cfg()
    win = _window(cfg)
    try:
        original, err = win._read_table()     # as the table holds them
        assert not err
        _select(win, 'cubic')
        assert win._pc_combo.isEnabled()
        assert 'h k l' in win._table.horizontalHeaderItem(1).text()
        assert all(len(r) == 3 for g in win._groups_cfg for r in g['reflist'])
        assert sorted(win._param_sliders) == [0]

        _select(win, 'icosahedral')
        assert win._groups_cfg == original
        assert win._fit_objective(win._reduced()) < 1e-7
    finally:
        win.close()


def test_gui_saved_quasicrystal_config_round_trips():
    cfg = _cfg()
    win = _window(cfg)
    try:
        win._param_sliders[7].valueChanged.emit(0.0011)   # a12
        saved = win._export_config()
        params = win._params.copy()
    finally:
        win.close()
    assert saved['computation']['bravais'] == 'icosahedral'
    assert saved['crystal']['phason'][1] == 0.0011
    assert saved['crystal']['tau_approx'] == ts.TAU_APPROX
    win = _window(saved)
    try:
        np.testing.assert_array_equal(win._params, params)
    finally:
        win.close()
    _run_batch(saved)                  # the batch app reads what the GUI writes


def test_gui_conventional_config_is_untouched():
    cfg = _cfg(PMN_CONFIG)
    win = _window(cfg)
    try:
        assert sorted(win._param_sliders) == ts.lattice_free_slots('rhombohedral')
        assert win._pc_combo.isEnabled()
        saved = win._export_config()
    finally:
        win.close()
    assert 'phason' not in saved['crystal']
    assert 'tau_approx' not in saved['crystal']
    assert saved['crystal']['initial_guess'] == cfg['crystal']['initial_guess']


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print('ok  %s' % fn.__name__)
    print('\n%d checks passed' % len(fns))


if __name__ == '__main__':
    _run()
