#!/usr/bin/env python
"""
dat2config — extract scan metadata from a Diamond ``.dat`` file into a DMS
configuration dict.

This is the *only* place in the workflow that reads a ``.dat`` file.  Both the
slider (``slider_quasi_*.py``) and the fitter (``workflow.py``) consume the
resulting ``experiment`` config section instead of opening the ``.dat``
themselves, so a run is no longer coupled to the original beamline directory
tree.

The ``experiment`` block produced here holds everything the analysis scripts
used to pull from the ``.dat``::

    "experiment": {
        "lattice":        [a, b, c, alpha, beta, gamma],
        "energy":         <energy at datapoint>,
        "energy0":        <energy at datapoint0>,   # energy/energy0 rescales hkl
        "azir":           [azih, azik, azil],
        "hkl":            [h, k, l],                # primary reflection at datapoint
        "psi":            <azimuthal angle, deg>,
        "image_template": "913123-pilatus2M-files/%05d.tif"
    }

``hkl`` and ``psi`` are the as-measured primary reflection and azimuth; they are
omitted when the ``.dat`` does not carry them.

CLI::

    python -m calcms.dat2config <scan.dat> <out.json> \
        [--template TMPL.json] [--datapoint N] [--datapoint0 M]
"""

import os
import re
import glob
import json
import argparse

from . import loader as do


def _energy_at(d, idx):
    """Replicate the historical energy-extraction fallback chain, evaluated at
    scan index ``idx``.  Order: ``energy2`` → ``metadata.Energy`` →
    ``DCMenergy`` → ``metadata.en``."""
    try:
        return float(d.energy2[idx])
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        return float(d.metadata.Energy)
    except AttributeError:
        pass
    try:
        return float(d.DCMenergy[idx])
    except (AttributeError, IndexError, TypeError):
        pass
    return float(d.metadata.en)


def energy_at(dat_path, datapoint):
    """Return the scan energy (keV) at ``datapoint`` using the same fallback
    chain as :func:`extract_metadata`."""
    return _energy_at(do.load(dat_path), datapoint)


def _hkl_at(d, idx):
    """The primary reflection at scan index ``idx``.

    The scanned ``h``/``k``/``l`` columns are the as-measured indices at each
    point and are preferred.  Many scans (the fixed-angle energy scans this
    analysis lives on, for one) carry no such columns, only the single
    ``metadata`` hkl — the position the scan was set up at, at the metadata
    energy ``en``.  The diffractometer does not move during those scans, so the
    indices at ``idx`` are that position scaled by the Bragg energy ratio,
    ``hkl * E[idx] / en``: on a scan that has both, this reproduces the columns
    to six decimals.  Returns ``None`` when the file carries no hkl at all."""
    try:
        return [float(d.h[idx]), float(d.k[idx]), float(d.l[idx])]
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        m   = d.metadata
        hkl = [float(m['h']), float(m['k']), float(m['l'])]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    try:
        en_meta = float(m.en)
        if en_meta > 0:
            ratio = _energy_at(d, idx) / en_meta
            hkl   = [v * ratio for v in hkl]
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        pass
    return hkl


def _psi_of(d, idx):
    """The azimuthal angle (deg).  Scanned ``psi`` column first, then the
    metadata value; ``None`` when the file carries neither."""
    try:
        return float(d.psi[idx])
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        return float(d.metadata['psi'])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _detector_template(dat_path, scannum):
    """Return the ``%05d``-style image template for ``scannum``.

    Looks for a ``<scannum>-<detector>-files`` directory beside the ``.dat`` and
    uses its detector name; defaults to ``pilatus2M`` if none is found."""
    scandir = os.path.dirname(os.path.abspath(dat_path))
    matches = sorted(glob.glob(os.path.join(scandir, '%s-*-files' % scannum)))
    if matches:
        folder = os.path.basename(matches[0])
        return '%s/%%05d.tif' % folder
    return '%s-pilatus2M-files/%%05d.tif' % scannum


def scan_length(dat_path):
    """Return the number of scan points in ``dat_path`` (>= 1).

    Works for any scan type: ``energy2`` / ``DCMenergy`` are used when present,
    otherwise the length of any scanned data column is used (every column has one
    entry per scan point), so non-energy scans (psi, eta, hkl, …) report their
    real length instead of 1."""
    d = do.load(dat_path)
    for attr in ('energy2', 'DCMenergy'):
        arr = getattr(d, attr, None)
        if arr is not None and hasattr(arr, '__len__'):
            return len(arr)
    # Fall back to the length of the first scanned data column.
    try:
        for v in d.values():
            if hasattr(v, '__len__') and not isinstance(v, str):
                return len(v)
    except Exception:
        pass
    return 1


def extract_metadata(dat_path, datapoint, datapoint0):
    """Read ``dat_path`` and return the ``experiment`` config block.

    Parameters
    ----------
    dat_path : str
        Path to the ``.dat`` scan file.
    datapoint : int
        Scan index of the image being analysed.
    datapoint0 : int
        Reference scan index used for the hkl energy-rescale ratio.
    """
    d = do.load(dat_path)
    m = d.metadata

    scannum = re.sub(r'\.dat$', '', os.path.basename(dat_path))

    lattice = [float(m.a), float(m.b), float(m.c),
               float(m.alpha1), float(m.alpha2), float(m.alpha3)]
    azir = [float(m['azih']), float(m['azik']), float(m['azil'])]

    exp = {
        'lattice':        lattice,
        'energy':         _energy_at(d, datapoint),
        'energy0':        _energy_at(d, datapoint0),
        'azir':           azir,
        'image_template': _detector_template(dat_path, scannum),
    }
    # The primary reflection and the azimuth as measured.  Both are optional —
    # a scan that does not carry them simply leaves the key out, and the
    # consumer keeps whatever it was using.
    hkl = _hkl_at(d, datapoint)
    if hkl is not None:
        exp['hkl'] = hkl
    psi = _psi_of(d, datapoint)
    if psi is not None:
        exp['psi'] = psi
    return exp


def dat_to_config(dat_path, template_cfg_path=None, datapoint=None, datapoint0=None):
    """Build a full config dict from a ``.dat`` file and an optional template.

    The template supplies all fixed analysis parameters (flags, display, roi,
    computation, crystal, …).  The ``experiment`` and ``scan`` sections are
    (re)populated from the ``.dat``.  ``datapoint``/``datapoint0`` default to the
    template's ``scan`` values, or to ``0`` if absent.
    """
    cfg = {}
    if template_cfg_path and os.path.exists(template_cfg_path):
        with open(template_cfg_path) as fh:
            cfg = json.load(fh)

    scan = cfg.setdefault('scan', {})
    if datapoint is None:
        datapoint = int(scan.get('datapoint', 0))
    if datapoint0 is None:
        datapoint0 = int(scan.get('datapoint0', 0))

    scannum = re.sub(r'\.dat$', '', os.path.basename(dat_path))
    scanpath = os.path.dirname(os.path.abspath(dat_path))
    if not scanpath.endswith(os.sep):
        scanpath += os.sep

    scan.update({
        'scannum':    int(scannum) if scannum.isdigit() else scannum,
        'scanpath':   scanpath,
        'datapoint':  int(datapoint),
        'datapoint0': int(datapoint0),
    })
    cfg['experiment'] = extract_metadata(dat_path, datapoint, datapoint0)
    return cfg


def _main(argv=None):
    p = argparse.ArgumentParser(
        description='Extract scan metadata from a .dat file into a DMS config.')
    p.add_argument('dat', help='input .dat scan file')
    p.add_argument('out', help='output config JSON path')
    p.add_argument('--template', help='template config JSON to merge into')
    p.add_argument('--datapoint', type=int, default=None)
    p.add_argument('--datapoint0', type=int, default=None)
    args = p.parse_args(argv)

    cfg = dat_to_config(args.dat, args.template, args.datapoint, args.datapoint0)
    with open(args.out, 'w') as fh:
        json.dump(cfg, fh, indent=2)
    print('Wrote %s' % args.out)
    print('experiment:', json.dumps(cfg['experiment'], indent=2))


if __name__ == '__main__':
    _main()
