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
        "image_template": "913123-pilatus2M-files/%05d.tif"
    }

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
    """Return the number of scan points in ``dat_path`` (>= 1)."""
    d = do.load(dat_path)
    for attr in ('energy2', 'DCMenergy'):
        arr = getattr(d, attr, None)
        if arr is not None and hasattr(arr, '__len__'):
            return len(arr)
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

    return {
        'lattice':        lattice,
        'energy':         _energy_at(d, datapoint),
        'energy0':        _energy_at(d, datapoint0),
        'azir':           azir,
        'image_template': _detector_template(dat_path, scannum),
    }


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
