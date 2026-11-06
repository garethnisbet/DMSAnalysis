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
        "image_template": "913123-pilatus2M-files/%05d.tif",
        "image_first":    <image number of datapoint 0>      # optional
    }

``hkl`` and ``psi`` are the as-measured primary reflection and azimuth; they are
omitted when the ``.dat`` does not carry them.

Images are numbered ``datapoint + 1`` in the ``<scan>-<detector>-files``
folders.  Older scans (2009 SRS-format i16 files) name each image by the
``path`` column instead — ``pilatus100k/test%d.tif`` — and then carry
``image_first`` (and ``image_numbers`` if the numbers are not contiguous);
:func:`image_number` turns a datapoint into the file number either way.

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
    """The azimuthal angle (deg).  A scanned ``psi`` column first, then
    ``psic`` (what the older GDA psi scans — e.g. 2009 SRS files — name the
    scanned azimuth), then the metadata value; ``None`` when the file carries
    none of them.  The metadata ``psi`` is the azimuth at the start of the scan
    only, so on a psi scan it must never win over the column."""
    for col in ('psi', 'psic'):
        try:
            return float(getattr(d, col)[idx])
        except (AttributeError, IndexError, TypeError):
            pass
    try:
        return float(d.metadata['psi'])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


IMAGE_EXTS = ('.tif', '.tiff', '.cbf', '.png', '.edf')


def _path_numbered_images(d, scandir):
    """Image template + numbering for scans whose images are named by the
    ``path`` column (older GDA/SRS files, e.g. 2009 i16 Pilatus 100K data:
    ``pilatus100k/test1232192.tif`` for the row with ``path=1232192``).

    Returns ``(template, numbering)`` — ``template`` relative to ``scandir``,
    with ``%d`` where the number goes, and ``numbering`` the
    ``image_first``/``image_numbers`` keys :func:`image_number` reads — or
    ``None`` when the scan has no ``path`` column or no matching file is found
    in ``scandir`` or one directory below it."""
    try:
        nums = [int(round(float(v))) for v in d.path]
    except (AttributeError, TypeError, ValueError):
        return None
    if not nums:
        return None
    n0 = str(nums[0])
    cands = (glob.glob(os.path.join(scandir, '*' + n0 + '.*')) +
             glob.glob(os.path.join(scandir, '*', '*' + n0 + '.*')))
    for p in sorted(cands):
        base, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() not in IMAGE_EXTS or not base.endswith(n0):
            continue
        prefix = base[:-len(n0)]
        if prefix and prefix[-1].isdigit():
            continue                      # n0 is the tail of a longer number
        rel = os.path.relpath(os.path.dirname(p), scandir)
        tmpl = prefix + '%d' + ext
        tmpl = tmpl if rel == '.' else rel + '/' + tmpl
        contiguous = all(b - a == 1 for a, b in zip(nums, nums[1:]))
        numbering = ({'image_first': nums[0]} if contiguous
                     else {'image_first': nums[0], 'image_numbers': nums})
        return tmpl, numbering
    return None


def _detector_template(dat_path, scannum, d=None):
    """Return ``(template, numbering)`` for the scan's detector images.

    Looks for a ``<scannum>-<detector>-files`` directory beside the ``.dat``
    (``%05d`` files numbered from 1 — ``numbering`` empty).  Failing that, a
    scan with a ``path`` column is searched for path-numbered images
    (:func:`_path_numbered_images`).  Defaults to ``pilatus2M``."""
    scandir = os.path.dirname(os.path.abspath(dat_path))
    matches = sorted(glob.glob(os.path.join(scandir, '%s-*-files' % scannum)))
    if matches:
        folder = os.path.basename(matches[0])
        return '%s/%%05d.tif' % folder, {}
    if d is not None:
        found = _path_numbered_images(d, scandir)
        if found is not None:
            return found
    return '%s-pilatus2M-files/%%05d.tif' % scannum, {}


def image_number(exp, datapoint):
    """The number that goes into ``exp['image_template']`` for ``datapoint``.

    ``image_numbers`` (an explicit per-point list) wins, then
    ``image_first + datapoint``; with neither key it is ``datapoint + 1``, the
    numbering of the ``<scan>-pilatus2M-files/%05d.tif`` folders."""
    exp = exp or {}
    dp = int(datapoint)
    nums = exp.get('image_numbers')
    if nums is not None and 0 <= dp < len(nums):
        return int(nums[dp])
    return int(exp.get('image_first', 1)) + dp


def resolve_scanpath(scanpath, cfg_path=None):
    """A config's ``scan.scanpath`` as an absolute folder ending in a separator.

    A relative path is taken relative to the config file's folder (or the
    working directory when there is no file), so a config shipped next to its
    data — ``configs/example_913232.json`` with ``../example_data/913232/`` —
    runs from anywhere.  Absolute paths, the usual beamline case, are returned
    unchanged apart from the trailing separator the apps concatenate onto."""
    p = os.path.expanduser(str(scanpath))
    if not os.path.isabs(p):
        base = os.path.dirname(os.path.abspath(cfg_path)) if cfg_path else os.getcwd()
        p = os.path.normpath(os.path.join(base, p))
    return p if p.endswith(os.sep) else p + os.sep


def image_path(scanpath, exp, datapoint):
    """Full path of the detector image for ``datapoint``."""
    return os.path.join(scanpath, exp['image_template'] % image_number(exp, datapoint))


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

    tmpl, numbering = _detector_template(dat_path, scannum, d)
    exp = {
        'lattice':        lattice,
        'energy':         _energy_at(d, datapoint),
        'energy0':        _energy_at(d, datapoint0),
        'azir':           azir,
        'image_template': tmpl,
    }
    # Only for path-numbered images; absent means datapoint + 1.
    exp.update(numbering)
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
