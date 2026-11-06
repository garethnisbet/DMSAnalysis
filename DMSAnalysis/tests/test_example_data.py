"""The bundled example (scan 913232) and what makes it portable.

* a relative ``scan.scanpath`` resolves against the config file, so
  ``configs/example_913232.json`` finds ``example_data/913232/`` from any
  working directory;
* the example's ``.dat`` and its one frame load through the apps' own readers;
* the ROI kernel is uint8 and holds only the kept ROIs — the float64 stack of
  every candidate needed ~7 GB on this scan zoomed 2x and was OOM-killed.

Run standalone:
    python -m DMSAnalysis.tests.test_example_data
or under pytest:
    pytest DMSAnalysis/tests/test_example_data.py
"""

import json
import os

import numpy as np

from .. import dat2config
from .. import ts_quasi as ts

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(PKG, 'configs', 'example_913232.json')
DATA = os.path.join(PKG, 'example_data', '913232')


def test_relative_scanpath_resolves_against_the_config_file():
    got = dat2config.resolve_scanpath('../example_data/913232/', CONFIG)
    assert got == DATA + os.sep
    assert dat2config.resolve_scanpath('/abs/scan/dir', CONFIG) == '/abs/scan/dir' + os.sep
    here = dat2config.resolve_scanpath('rel', None)
    assert here == os.path.join(os.getcwd(), 'rel') + os.sep


def test_example_config_finds_its_scan_and_frame():
    import imageio.v2 as imageio
    cfg = json.load(open(CONFIG))
    sc = cfg['scan']
    path = dat2config.resolve_scanpath(sc['scanpath'], CONFIG)
    dat = os.path.join(path, '%d.dat' % sc['scannum'])
    exp = dat2config.extract_metadata(dat, sc['datapoint'], sc['datapoint0'])
    assert abs(exp['energy'] - 6.3) < 0.01
    im = imageio.imread(dat2config.image_path(path, exp, sc['datapoint']))
    assert im.shape == (1679, 1475) and im.dtype == np.int32
    assert im.max() > 1000 and (im < 0).any()      # counts, and module gaps
    # the example runs at the detector's own resolution: interpolation is the
    # ROI integrator's job now, not a 2x cubic zoom's
    assert cfg['display']['zoomval'] == 1


def test_example_config_differs_from_its_source_only_in_path_and_zoom():
    src = json.load(open(os.path.join(
        PKG, 'configs', 'fit_fivefold_axis_AlPdMn_Not_Annealed_2M_2ROIS_internal_hkl.json')))
    ex = json.load(open(CONFIG))
    src['scan']['scanpath'] = ex['scan']['scanpath']
    src['display']['zoomval'] = ex['display']['zoomval']
    assert src == ex


def test_kernel_is_uint8_and_only_the_kept_rois():
    from .test_dms_curves import cubic_kwargs
    kw = cubic_kwargs()
    reflist = np.asarray(kw['reflist'], float)
    lattice, hkl, energy = kw['lattice'], kw['hkl'], kw['energy']
    thb = ts.bragg(lattice, hkl, energy).th()[0]
    hkllist = ts.pilkhlrange(lattice, hkl, energy, thb - 27, thb + 10).hklscan(200)
    ig = np.zeros(24)
    ig[:6] = lattice
    ig[10] = 3000.0
    ig[14] = energy
    args = (reflist, hkllist, np.round(hkl), 1, [-360, 0], 0, hkl,
            np.matrix([[1, 0, 0], [0, 0, 1]]), (1200, 1200), 0.0, kw['azir'],
            -180.0, 600, 600, 0, ig[10], ig[11], ig[12], ig[13], ig[14], ig,
            np.zeros_like(reflist), [0.0] * 9, 'cubic')
    k = ts.roibuilder_ico_hkl(args)
    assert k.dtype == np.uint8
    assert 0 < k.shape[2] < 2 * len(reflist)       # the fixture misses some
    assert all(k[:, :, i].any() for i in range(k.shape[2]))


def test_path_stride_is_one_sample_per_detector_pixel():
    assert ts.roi_path_stride(0.25) == 1
    assert ts.roi_path_stride(0.5) == 2
    assert ts.roi_bin_from_config({}, 1) == ts.ROI_BIN_STEP
    assert ts.roi_bin_from_config({'bin_step_per_zoom': 0.25}, 2) == 0.5
    assert ts.roi_bin_from_config({'bin_step_per_zoom': None}, 2) is None


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok ', name)
