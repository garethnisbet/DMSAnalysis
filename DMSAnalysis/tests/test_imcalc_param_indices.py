"""The reduced parameter vector must be packed the way imcalc unpacks it.

``ts_quasi.imcalc_param_indices`` picks the slots of the 24-element guess that
the slider and fit.py hand to ``dmsfit_ico_hkl.imcalc``; imcalc reads them back
at fixed offsets per mode.  The tables used to be written out by hand and had
drifted: ``cubic_no_strain`` with the detector refined was a slot short, so
every overlay update raised IndexError inside the slider's worker thread and the
DMS lines silently stopped following the sliders; ``icosahedral_fixed_a`` read
the energy in as the detector z rotation.

The check is the round trip through ``inputarray`` (the full guess imcalc says it
simulated): the engine is built on one geometry and handed a vector from a
*different* one, so a slot only comes back with the vector's value if imcalc
really took it from the vector — and a slot the mode does not refine comes back
with the engine's own value.

Run standalone:
    python -m DMSAnalysis.tests.test_imcalc_param_indices
or under pytest:
    pytest DMSAnalysis/tests/test_imcalc_param_indices.py
"""

import itertools

import numpy as np

from .. import ts_quasi as ts
from .test_inputarray_roundtrip import _engine, _guess

DETECTOR_ENERGY = [10, 11, 12, 13, 14]
PHASON = list(range(15, 24))


def _shifted(base):
    """A guess differing from ``base`` in every slot a mode could refine."""
    ig = base.copy()
    ig[0:3] += 0.05
    ig[6:9] += [0.3, -0.2, 0.1]
    ig[10:15] += [3.0, 0.1, -0.1, 0.2, 0.004]
    ig[15:24] = np.arange(1, 10) * 1e-4
    return ig


def _check(mode, detopt, energyopt):
    base = _guess()
    ig = _shifted(base)
    idx = ts.imcalc_param_indices(mode, detopt, energyopt)
    dms = _engine(mode, base, detopt, energyopt)
    dms.imcalc(ig[idx])
    out = np.asarray(dms.inputarray)
    tag = '%s detopt=%d energyopt=%d' % (mode, detopt, energyopt)

    for s in [6, 7, 8] + DETECTOR_ENERGY:
        want = ig[s] if s in idx else base[s]
        assert np.isclose(out[s], want), '%s: slot %d is %r, expected %r' % (
            tag, s, out[s], want)
    if mode in ts.CONVENTIONAL_SYSTEMS or mode in ('icosahedral', 'cubic_no_strain'):
        assert np.isclose(out[0], ig[0]), '%s: lattice a not taken from the vector' % tag
    if mode in ('icosahedral', 'icosahedral_fixed_a'):
        assert np.allclose(out[PHASON], ig[PHASON]), '%s: phason block misread' % tag


def test_every_mode_round_trips():
    modes = list(ts.IMCALC_MODES) + ['cubic', 'rhombohedral', 'monoclinic']
    for mode, detopt, energyopt in itertools.product(modes, (True, False),
                                                     (True, False)):
        _check(mode, detopt, energyopt)


def test_cubic_no_strain_with_detector_evaluates():
    """The slider's failing case: the detector refined, energy not."""
    base = _guess()
    dms = _engine('cubic_no_strain', base, True, False)
    dms.imcalc(base[ts.imcalc_param_indices('cubic_no_strain', True, False)])
    assert np.isclose(dms.inputarray[13], base[13])


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok  ', name)
    print('all passed')
