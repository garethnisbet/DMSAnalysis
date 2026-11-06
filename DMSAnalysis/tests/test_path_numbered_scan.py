"""Older i16 scans: images named by the ``path`` column, azimuth in ``psic``.

2009 SRS-format psi scans (e.g. ``/dls/i16/data/2009/mt0/run3/105903.dat``)
differ from the ``<scan>-pilatus2M-files/%05d.tif`` layout the apps were written
for in two ways that matter:

* each row's Pilatus 100K image is ``pilatus100k/test<path>.tif``, ``path``
  being a scanned column (1232192, 1232193, …) — not ``datapoint + 1`` — and the
  ``pilatus100k`` folder holds every scan of the visit;
* the scanned azimuth is the ``psic`` column.  The metadata ``psi`` is only the
  value at the start of the scan.

``dat2config`` detects the first and carries ``image_first`` (``image_numbers``
when not contiguous); every image reader goes through
``dat2config.image_number``.

Run standalone:
    python -m DMSAnalysis.tests.test_path_numbered_scan
or under pytest:
    pytest DMSAnalysis/tests/test_path_numbered_scan.py
"""

import os
import json
import tempfile

import numpy as np
import imageio.v2 as imageio

from .. import dat2config

SCANNUM = 105903
FIRST = 1232192
PSIC = [-100.0, -99.9, -99.8, -99.7]
LATTICE = [3.586, 3.586, 5.642, 90.0, 90.0, 120.0]
SHAPE = (195, 487)                      # Pilatus 100K


def write_srs_dat(path, path_numbers, psic=PSIC, meta_psi=89.999):
    """A 2009 SRS-format .dat, as GDA wrote psi scans with a Pilatus 100K:
    metadata energy in keV under ``Energy``, no ``energy2`` column, and the
    detector's ``maxx/maxy/maxval/sum`` columns repeated for its ROI."""
    meta = ['testdata=\'letagetabitarockin\'', '',
            'datestring=\'Fri Sep 11 12:49:24 2009\'',
            'psi=%.6f' % meta_psi, 'Energy=5.999995616377558',
            'azih=1.0', 'azik=0.0', 'azil=0.0',
            'a=%.6f' % LATTICE[0], 'b=%.6f' % LATTICE[1], 'c=%.6f' % LATTICE[2],
            'alpha1=90', 'alpha2=90', 'alpha3=120']
    cols = ['psic', 'h', 'k', 'l', 'path', 'maxx', 'maxy', 'maxval', 'sum',
            'ic1', 'maxx', 'maxy', 'maxval', 'sum']
    with open(path, 'w') as fh:
        fh.write(' &SRS\n SRSRUN=%d,SRSDAT=2009811,SRSTIM=04954,\n'
                 " SRSSTN='BASE',SRSPRJ='GDA_BASE',SRSEXP='Emulator',\n"
                 '<MetaDataAtStart>\n' % SCANNUM)
        fh.write('\n'.join(meta) + '\n</MetaDataAtStart>\n &END\n')
        fh.write('\t'.join(cols) + '\n')
        for p, n in zip(psic, path_numbers):
            row = [p, 0.0, 0.0, 4.1665, n, 257, 103, 9235.0, 4256522.0,
                   102.08, 10, 8, 9235.0, 125022.0]
            fh.write('\t'.join('%.10g' % v for v in row) + '\n')
    return path


def make_scan(tmp, path_numbers=None, other_scan=True):
    """The .dat plus ``pilatus100k/test<n>.tif`` frames whose every pixel holds
    ``n % 1000``, so a loaded image says which file it came from.  A second
    scan's frames share the folder, as at the beamline."""
    if path_numbers is None:
        path_numbers = [FIRST + i for i in range(len(PSIC))]
    write_srs_dat(os.path.join(tmp, '%d.dat' % SCANNUM), path_numbers)
    det = os.path.join(tmp, 'pilatus100k')
    os.makedirs(det, exist_ok=True)
    extra = [FIRST + 500, FIRST + 501] if other_scan else []
    for n in list(path_numbers) + extra:
        imageio.imwrite(os.path.join(det, 'test%d.tif' % n),
                        np.full(SHAPE, n % 1000, dtype=np.int32))
    return os.path.join(tmp, '%d.dat' % SCANNUM), path_numbers


# ── the reader ─────────────────────────────────────────────────────────────────

def test_metadata_of_a_path_numbered_psi_scan():
    with tempfile.TemporaryDirectory() as tmp:
        dat, nums = make_scan(tmp)
        exp = dat2config.extract_metadata(dat, 2, 0)
        assert exp['image_template'] == 'pilatus100k/test%d.tif', exp
        assert exp['image_first'] == FIRST
        assert 'image_numbers' not in exp            # contiguous
        # psi is the scanned psic at that point, not the metadata start value
        assert np.isclose(exp['psi'], PSIC[2]), exp['psi']
        assert np.isclose(exp['energy'], 5.999995616377558)
        assert np.allclose(exp['lattice'], LATTICE)
        assert np.allclose(exp['hkl'], [0, 0, 4.1665])
        assert dat2config.scan_length(dat) == len(PSIC)
        path = dat2config.image_path(tmp, exp, 2)
        assert path == os.path.join(tmp, 'pilatus100k', 'test%d.tif' % nums[2])
        assert os.path.exists(path)


def test_non_contiguous_path_numbers_are_listed():
    with tempfile.TemporaryDirectory() as tmp:
        nums = [FIRST, FIRST + 1, FIRST + 5, FIRST + 6]
        dat, _ = make_scan(tmp, nums)
        exp = dat2config.extract_metadata(dat, 0, 0)
        assert exp['image_numbers'] == nums
        assert [dat2config.image_number(exp, i) for i in range(4)] == nums


def test_default_numbering_is_unchanged():
    """No numbering keys: datapoint + 1, the <scan>-pilatus2M-files layout."""
    exp = {'image_template': '913232-pilatus2M-files/%05d.tif'}
    assert dat2config.image_number(exp, 0) == 1
    assert dat2config.image_number(exp, 41) == 42
    assert dat2config.image_number(None, 3) == 4


def test_files_folder_still_wins():
    """A <scan>-<detector>-files folder is the modern layout and is preferred
    even when the .dat also carries a path column."""
    with tempfile.TemporaryDirectory() as tmp:
        dat, _ = make_scan(tmp)
        os.makedirs(os.path.join(tmp, '%d-pilatus2M-files' % SCANNUM))
        exp = dat2config.extract_metadata(dat, 0, 0)
        assert exp['image_template'] == '%d-pilatus2M-files/%%05d.tif' % SCANNUM
        assert 'image_first' not in exp


# ── the slider ─────────────────────────────────────────────────────────────────

def write_config(tmpdir):
    cfg = {
        'scan': {'scannum': SCANNUM, 'scanpath': tmpdir + os.sep,
                 'datapoint': 0, 'datapoint0': 0},
        'geometry': {'hkl': [0.0, 0.0, 4.0], 'psi': 0.0,
                     'px_unscaled': 257, 'py_unscaled': 104, 'scatv': 0},
        'display': {'zoomval': 1, 'colourlim': [0, 1000], 'colmap': 'gray'},
        'roi': {'width_per_zoom': 45, 'comwidth_per_zoom': 5},
        'computation': {'numsteps': 60, 'simsigma_per_zoom': 4.5,
                        'thrange_delta': [-27, 10], 'bravais': 'hexagonal',
                        'curve_method': 'sweep'},
        'crystal': {'initial_guess_base': LATTICE + [0.0] * 18,
                    'reflist_hkl': [[0, 0, 4], [1, 0, 0]]},
        'flags': {'save': 0, 'fit': 0, 'firstplot': 0, 'detoptimize': 1,
                  'energyopt': 0},
    }
    path = os.path.join(tmpdir, 'cfg.json')
    with open(path, 'w') as fh:
        json.dump(cfg, fh)
    return path


def test_slider_loads_the_path_numbered_image():
    from .gui_harness import slider_on
    tmp = tempfile.mkdtemp()
    dat, nums = make_scan(tmp)
    sl = slider_on(write_config(tmp))
    win = sl.win

    win._do_load_scan(dat, 2, 0)
    # the frame for datapoint 2 is test<FIRST+2>.tif, not test3.tif
    assert np.allclose(win._imdata, nums[2] % 1000), np.unique(win._imdata)[:5]
    assert np.isclose(win._psi, PSIC[2]), win._psi
    assert win._imnumbering == {'image_first': FIRST}

    # the scrubber lists this scan's frames only and sits on datapoint 2
    files = win._scan_image_files()
    assert [n for n, _ in files] == nums, files
    win._update_img_scrub_range(2)
    assert win._img_files[win._img_scrub.value()][0] == nums[2]

    # an exported config reloads the same image
    cfg = win._build_workflow_config()
    assert cfg['experiment']['image_template'] == 'pilatus100k/test%d.tif'
    assert cfg['experiment']['image_first'] == FIRST
    assert dat2config.image_number(cfg['experiment'], 2) == nums[2]


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  %s' % name)
    print('all passed')
