"""Loading a scan takes its numbers from the .dat.

A ``.dat`` measures the lattice, the energy, the primary reflection and the
azimuth.  Loading a *different* scan therefore replaces the slider state with
those values — the config no longer has to be hand-edited to follow the data,
and an exported JSON is built from what the file said (or from wherever the
user then drags the sliders).

Every load seeds, including a reload of the scan already open: a session
restored onto scan N (the auto-saved session does exactly that) must not make an
explicit Load of N's own .dat a no-op.  Untick "Seed sliders from .dat" to keep
a refinement instead — that is the branch where hkl follows the energy ratio
across a datapoint step rather than the file.

Run standalone:
    python -m DMSAnalysis.tests.test_scan_load_seeding
or under pytest:
    pytest DMSAnalysis/tests/test_scan_load_seeding.py
"""

import os
import json
import tempfile

import numpy as np

from .. import dat2config


# ── synthetic scans ────────────────────────────────────────────────────────────

def write_dat(path, lattice, azir, psi, hkl, energies, with_hkl=True,
              with_psi=True, hkl_columns=True, meta_energy=None):
    """A minimal Diamond-format .dat: metadata block, then one row per point.

    ``hkl`` is the value at the first point; it scales with the energy ratio
    down the scan, as a real hkl-tracking energy scan does.
    """
    meta = ['a=%.6f' % lattice[0], 'b=%.6f' % lattice[1], 'c=%.6f' % lattice[2],
            'alpha1=%.6f' % lattice[3], 'alpha2=%.6f' % lattice[4],
            'alpha3=%.6f' % lattice[5],
            'azih=%.6f' % azir[0], 'azik=%.6f' % azir[1], 'azil=%.6f' % azir[2],
            'en=%.6f' % (energies[0] if meta_energy is None else meta_energy)]
    if with_psi:
        meta.append('psi=%.6f' % psi)
    if with_hkl:
        meta += ['h=%.6f' % hkl[0], 'k=%.6f' % hkl[1], 'l=%.6f' % hkl[2]]

    cols = ['energy2']
    rows = [[e] for e in energies]
    if with_hkl and hkl_columns:
        cols += ['h', 'k', 'l']
        for row, e in zip(rows, energies):
            row += list(np.asarray(hkl, float) * e / energies[0])
    if with_psi:
        cols += ['psi']
        for row in rows:
            row.append(psi)

    with open(path, 'w') as fh:
        fh.write(' &SRS\n SRSRUN=%s,\n<MetaDataAtStart>\n'
                 % os.path.basename(path)[:-4])
        fh.write('\n'.join(meta) + '\n &END\n')
        fh.write('\t'.join(cols) + '\n')
        for row in rows:
            fh.write('\t'.join('%.6f' % v for v in row) + '\n')
    return path


SCAN_A = dict(scannum=1001,
              lattice=[4.00, 4.00, 4.00, 90.0, 90.0, 90.0],
              azir=[0.0, 0.0, 1.0], psi=-180.0,
              hkl=[1.0, 1.0, 3.0], energies=[8.0, 8.02, 8.04])
SCAN_B = dict(scannum=1002,
              lattice=[4.20, 4.20, 4.20, 90.0, 90.0, 90.0],
              azir=[0.0, 1.0, 2.0], psi=37.5,
              hkl=[0.0, 2.0, 2.0], energies=[9.0, 9.05, 9.10])


def make_scans(tmpdir):
    for s in (SCAN_A, SCAN_B):
        write_dat(os.path.join(tmpdir, '%d.dat' % s['scannum']),
                  s['lattice'], s['azir'], s['psi'], s['hkl'], s['energies'])


def write_config(tmpdir):
    """A config on SCAN_A whose geometry deliberately disagrees with the .dat,
    so anything the load seeds is visible."""
    cfg = {
        'scan': {'scannum': SCAN_A['scannum'], 'scanpath': tmpdir + os.sep,
                 'datapoint': 0, 'datapoint0': 0},
        'geometry': {'hkl': [2.0, 2.0, 2.0], 'psi': 0.0,
                     'px_unscaled': 700, 'py_unscaled': 500, 'scatv': 0},
        'display': {'zoomval': 1, 'colourlim': [0, 1000], 'colmap': 'gray'},
        'roi': {'width_per_zoom': 45, 'comwidth_per_zoom': 5},
        'computation': {'numsteps': 60, 'simsigma_per_zoom': 4.5,
                        'thrange_delta': [-27, 10], 'bravais': 'cubic',
                        'curve_method': 'sweep'},
        'crystal': {'initial_guess_base': [4.0, 4.0, 4.0, 90.0, 90.0, 90.0]
                                          + [0.0] * 18,
                    'reflist_hkl': [[0, 0, 2], [1, 1, 0]]},
        'flags': {'save': 0, 'fit': 0, 'firstplot': 0, 'detoptimize': 1,
                  'energyopt': 0},
    }
    path = os.path.join(tmpdir, 'cfg.json')
    with open(path, 'w') as fh:
        json.dump(cfg, fh)
    return path


# ── the reader ─────────────────────────────────────────────────────────────────

def test_extract_metadata_reads_hkl_and_psi():
    with tempfile.TemporaryDirectory() as tmp:
        make_scans(tmp)
        exp = dat2config.extract_metadata(
            os.path.join(tmp, '%d.dat' % SCAN_B['scannum']), 2, 0)
        assert np.allclose(exp['lattice'], SCAN_B['lattice'])
        assert np.isclose(exp['energy'],  SCAN_B['energies'][2])
        assert np.isclose(exp['energy0'], SCAN_B['energies'][0])
        # hkl is the value *at that datapoint*, not the metadata start value
        want = (np.asarray(SCAN_B['hkl'])
                * SCAN_B['energies'][2] / SCAN_B['energies'][0])
        assert np.allclose(exp['hkl'], want)
        assert np.isclose(exp['psi'], SCAN_B['psi'])


def test_metadata_hkl_tracks_the_datapoint():
    """A scan with no h/k/l columns — the fixed-angle energy scans this
    analysis lives on — carries one metadata hkl, recorded at the metadata
    energy.  The indices at a datapoint are that position scaled by the Bragg
    energy ratio, which is exactly what the columns hold on a scan that has
    them."""
    with tempfile.TemporaryDirectory() as tmp:
        en_meta = 8.5
        p = write_dat(os.path.join(tmp, '2002.dat'), SCAN_A['lattice'],
                      SCAN_A['azir'], SCAN_A['psi'], SCAN_A['hkl'],
                      SCAN_A['energies'], hkl_columns=False,
                      meta_energy=en_meta)
        exp = dat2config.extract_metadata(p, 2, 0)
        want = np.asarray(SCAN_A['hkl']) * SCAN_A['energies'][2] / en_meta
        assert np.allclose(exp['hkl'], want), exp['hkl']


def test_extract_metadata_omits_what_the_dat_lacks():
    """A scan with no h/k/l or psi must leave the keys out, not invent them."""
    with tempfile.TemporaryDirectory() as tmp:
        p = write_dat(os.path.join(tmp, '2001.dat'), SCAN_A['lattice'],
                      SCAN_A['azir'], 0.0, [0, 0, 0], SCAN_A['energies'],
                      with_hkl=False, with_psi=False)
        exp = dat2config.extract_metadata(p, 1, 0)
        assert 'hkl' not in exp and 'psi' not in exp
        assert np.isclose(exp['energy'], SCAN_A['energies'][1])


# ── the slider ─────────────────────────────────────────────────────────────────

def slider_on(cfg_path):
    """The headless slider, built on ``cfg_path`` if it is the first caller."""
    from .gui_harness import slider_on as _on
    return _on(cfg_path)


def slider_values(win):
    """(lattice a, energy, hkl, psi) as the GUI currently holds them."""
    win._sync_ig()
    return (float(win.ig[0]), float(win.ig[14]), win._hkl.copy(), float(win._psi))


def test_new_scan_seeds_from_the_dat():
    tmp = tempfile.mkdtemp()
    make_scans(tmp)
    sl  = slider_on(write_config(tmp))
    win = sl.win
    dp  = 2

    win._do_load_scan(os.path.join(tmp, '%d.dat' % SCAN_B['scannum']), dp, 0)
    a, en, hkl, psi = slider_values(win)

    assert np.isclose(a,  SCAN_B['lattice'][0]), a
    assert np.isclose(en, SCAN_B['energies'][dp]), en
    want = (np.asarray(SCAN_B['hkl'])
            * SCAN_B['energies'][dp] / SCAN_B['energies'][0])
    assert np.allclose(hkl, want), hkl
    assert np.isclose(psi, SCAN_B['psi']), psi
    # the engines are rebuilt on the seeded reflection, not the config's
    assert np.allclose(sl.hklint, np.round(want))
    # and the seeded values are what an export would carry
    cfg = win._build_workflow_config()
    assert np.allclose(cfg['geometry']['hkl'], want)
    assert np.isclose(cfg['geometry']['psi'], SCAN_B['psi'])
    assert np.allclose(cfg['experiment']['lattice'], SCAN_B['lattice'])


def test_reload_of_the_same_scan_also_seeds():
    """The scan already open is still a scan: loading it takes the .dat's
    numbers at that datapoint, refinement or not.  This is the case the
    auto-saved session lands in — restored onto scan N, then Load on N.dat."""
    tmp = tempfile.mkdtemp()
    make_scans(tmp)
    sl   = slider_on(write_config(tmp))
    win  = sl.win
    path = os.path.join(tmp, '%d.dat' % SCAN_B['scannum'])

    win._do_load_scan(path, 0, 0)
    # refine by hand: a lattice parameter and the primary reflection
    win._sliders['a'].setValue(SCAN_B['lattice'][0] + 0.05)
    for lbl, i in (('h', 0), ('k', 1), ('l', 2)):
        win._sliders[lbl].setValue(float(win._hkl[i]) + 0.01)

    win._do_load_scan(path, 2, 0)
    a, en, hkl, psi = slider_values(win)

    assert np.isclose(a, SCAN_B['lattice'][0]), a
    assert np.isclose(en, SCAN_B['energies'][2]), en
    want = (np.asarray(SCAN_B['hkl'])
            * SCAN_B['energies'][2] / SCAN_B['energies'][0])
    assert np.allclose(hkl, want), hkl


def test_seed_tickbox_off_keeps_the_sliders():
    """Unticked, a load brings the image and the azimuthal reference and
    nothing else — and a datapoint step still carries the refined hkl across on
    the energy ratio."""
    tmp = tempfile.mkdtemp()
    make_scans(tmp)
    sl  = slider_on(write_config(tmp))
    win = sl.win

    win._do_load_scan(os.path.join(tmp, '%d.dat' % SCAN_A['scannum']), 0, 0)
    win._sliders['a'].setValue(SCAN_A['lattice'][0] + 0.07)
    a0, en0, hkl0, _psi0 = slider_values(win)

    win._chk_seed_dat.setChecked(False)
    try:
        # a different scan: nothing but the image and azir moves
        win._do_load_scan(os.path.join(tmp, '%d.dat' % SCAN_B['scannum']), 1, 0)
        a1, en1, hkl1, _psi1 = slider_values(win)
        assert np.isclose(a1, a0), (a0, a1)
        assert np.allclose(hkl1, hkl0), (hkl0, hkl1)
        assert np.isclose(en1, en0), (en0, en1)
        assert win._scannum == SCAN_B['scannum']
        assert np.allclose(win._azir, SCAN_B['azir'])

        # a datapoint step within it: hkl follows the energy ratio from the
        # refined value, not the file's
        win._do_load_scan(os.path.join(tmp, '%d.dat' % SCAN_B['scannum']), 2, 0)
        _a2, _en2, hkl2, _psi2 = slider_values(win)
        ratio = SCAN_B['energies'][2] / SCAN_B['energies'][1]
        assert np.allclose(hkl2, hkl1 * ratio), (hkl1, hkl2)
    finally:
        win._chk_seed_dat.setChecked(True)


def test_session_restore_wins_over_the_dat():
    """A restored session carries its own geometry; the reload must not seed
    over it."""
    tmp = tempfile.mkdtemp()
    make_scans(tmp)
    sl  = slider_on(write_config(tmp))
    win = sl.win

    win._do_load_scan(os.path.join(tmp, '%d.dat' % SCAN_B['scannum']), 1, 0)
    win._sliders['a'].setValue(SCAN_B['lattice'][0] + 0.03)
    win._psi = SCAN_B['psi'] + 4.0
    data = win._session_dict()

    # move away, then restore
    win._do_load_scan(os.path.join(tmp, '%d.dat' % SCAN_A['scannum']), 0, 0)
    win._restore_from_dict(data)
    a, en, hkl, psi = slider_values(win)

    assert np.isclose(a, SCAN_B['lattice'][0] + 0.03), a
    assert np.allclose(hkl, data['hkl']), hkl
    assert np.isclose(psi, SCAN_B['psi'] + 4.0), psi


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('all passed')
