"""The parameter vector imcalc reports back must be the one it simulated.

``dmsfit_ico_hkl.imcalc`` publishes ``self.inputarray``, the full 24-element
guess describing the geometry it just simulated.  The slider reads it back into
its own guess after a fit (``_on_fit_done``), so any slot it fails to carry is
silently reset in the GUI — and the refined result then describes a geometry the
fit never evaluated.

Slots 7 and 8 are the chi and theta corrections (formerly hcor/kcor; every
branch of imcalc reads them from the guess and holds h/k/l_correction at zero).
They used to be written out as the h/k corrections, i.e. as zero, which is how a
fit that genuinely lowered the residual could hand the GUI a worse one.

Run standalone:
    python -m DMSAnalysis.tests.test_inputarray_roundtrip
or under pytest:
    pytest DMSAnalysis/tests/test_inputarray_roundtrip.py
"""

import numpy as np

from .. import ts_quasi as ts

# Slots of the 24-element guess (see CLAUDE.md) that imcalc must echo back.
PSICOR, CHICOR, THCOR = 6, 7, 8


def _engine(bravais, ig, detopt=True, energyopt=False):
    """A dmsfit_ico_hkl over a small synthetic image.

    Only imcalc is exercised, so the ROI kernel and the target centres are
    dummies — the peak extraction is never reached.
    """
    imdata = np.zeros((64, 64))
    kernel = np.zeros((64, 64, 1))
    kernel[30:34, 30:34, 0] = 1
    centres = np.array([[16.0]])
    reflist = np.array([[1.0, -1.0, 2.0]])
    reflist2 = np.zeros_like(reflist)
    hkl = np.array([0.0, 0.0, 3.0])
    dms = ts.dmsfit_ico_hkl(
        reflist, [-27.0, 10.0, 40], np.round(hkl), [-180.0, 180.0], 8.0,
        centres, kernel, hkl, np.matrix([[1, 0, 0], [0, 0, 1]]), imdata, 0.0,
        [1.0, 0.0, 0.0], 0.0, 32.0, 32.0, 0,
        bravais, bool(detopt), bool(energyopt),
        ig[10], ig[11], ig[12], ig[13], ig[14],
        reflist2, list(ig[15:24]), ig[0])
    dms.setLattice(list(ig[:6]))
    dms.setCalLattice(list(ig[:6]))
    dms.setIGFull(ig)
    return dms


def _guess():
    ig = np.zeros(24)
    ig[:6] = [4.02288, 4.02288, 4.02288, 89.85825, 89.85825, 89.85825]
    ig[PSICOR], ig[CHICOR], ig[THCOR] = 0.11737, 0.19480, 0.88490
    ig[10:15] = [5194.526, 0.228572, 0.667038, -3.4e-5, 8.354]
    return ig


def _roundtrip(bravais):
    """(guess, inputarray) after simulating a guess with non-zero corrections."""
    ig = _guess()
    dms = _engine(bravais, ig)
    slots = list(ts.reduced_param_indices(bravais, True, False)) \
        if bravais in ts.CONVENTIONAL_SYSTEMS else None
    if slots is None:
        # Icosahedral packing: a, psicor, chicor, thcor, lcor, detector block.
        reduced = np.concatenate([ig[[0, 6, 7, 8, 9, 10, 11, 12, 13]], ig[15:24]])
    else:
        reduced = ig[slots]
    dms.imcalc(reduced)
    return ig, np.asarray(dms.inputarray)


def test_conventional_chi_and_theta_survive_imcalc():
    ig, out = _roundtrip('rhombohedral')
    assert out[PSICOR] == ig[PSICOR]
    assert out[CHICOR] == ig[CHICOR], 'chi correction lost by inputarray'
    assert out[THCOR] == ig[THCOR], 'theta correction lost by inputarray'


def test_icosahedral_chi_and_theta_survive_imcalc():
    ig, out = _roundtrip('icosahedral')
    assert out[PSICOR] == ig[PSICOR]
    assert out[CHICOR] == ig[CHICOR], 'chi correction lost by inputarray'
    assert out[THCOR] == ig[THCOR], 'theta correction lost by inputarray'


def test_reported_lattice_is_the_constrained_one():
    """The echoed lattice is what was simulated, i.e. after the crystal-system
    constraint — not the raw slots of the guess."""
    ig = _guess()
    ig[1] = ig[2] = 9.99      # b, c are not free in rhombohedral
    ig[4] = ig[5] = 12.0      # nor are beta, gamma
    dms = _engine('rhombohedral', ig)
    dms.imcalc(ig[list(ts.reduced_param_indices('rhombohedral', True, False))])
    out = np.asarray(dms.inputarray)
    assert np.allclose(out[:3], ig[0])
    assert np.allclose(out[3:6], ig[3])


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  ', name)
    print('all passed')
