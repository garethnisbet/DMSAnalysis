#!/usr/bin/env python
"""
slider_quasi_AlPdMn_Annealed_hkl_v3.py
Interactive DMS simulation viewer – PyQtGraph, dark theme, background threading.
"""
import sys, os, time, itertools, threading, json, re, subprocess, copy
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PyQt5')

PKGDIR  = os.path.abspath(os.path.dirname(__file__))
CONFIGS = os.path.join(PKGDIR, 'configs')

# Auto-saved session file (written on exit, offered for restore on next launch)
SESSION_FILE = os.path.join(os.path.expanduser('~'), '.dms_slider_session.json')

import numpy as np
from scipy import ndimage
import imageio.v2 as imageio

from . import ts_quasi as ts
from . import dat2config
from .config_table import ConfigTable

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder='row-major',
                    background='#1a1a1a', foreground='#cccccc')

# ── constants ──────────────────────────────────────────────────────────────────
tau = 55 / 34.

# ── scan / geometry ────────────────────────────────────────────────────────────
zoomval   = 1
numsteps  = 1000
numsteps_interactive = 300
colourlim = [0, 1000]
colmap    = 'gray'
simsigma  = 4.5 * zoomval
scatv     = 0

# ── config (optional path arg; .dat is read only as a fallback via the converter) ─
if len(sys.argv) > 1:
    cfg_path = os.path.abspath(sys.argv[1])
    with open(cfg_path) as _f:
        cfg = json.load(_f)
else:
    cfg_path = None
    cfg = {
        'scan': {
            'scannum':    913123,   # AlPdMn Annealed energy scan
            'scanpath':   '/home/ndf61257/MintSpace/i16extra/data/2021/mm29043-1/',
            'datapoint':  3,
            'datapoint0': 1,
        },
        'geometry': {
            'hkl':         [2.27931876, 3.70249186, 1.29579814],
            'psi':         -180,
            'px_unscaled': 1145,
            'py_unscaled': 817,
            'scatv':       scatv,
        },
        'display':     {'zoomval': zoomval, 'colourlim': colourlim, 'colmap': colmap},
        'computation': {'numsteps': numsteps,
                        'simsigma_per_zoom': simsigma / max(zoomval, 1),
                        'thrange_delta': [-27, 10]},
        'flags':       {'save': 0, 'fit': 0, 'firstplot': 0,
                        'detoptimize': 1, 'energyopt': 0, 'autoreflist': 0},
    }

scannum    = cfg['scan']['scannum']
scanpath   = cfg['scan']['scanpath']
datapoint  = cfg['scan']['datapoint']
datapoint0 = cfg['scan']['datapoint0']
imnum      = datapoint + 1

exp = cfg.get('experiment')
if exp is None:
    exp = dat2config.extract_metadata(
        os.path.join(scanpath, str(scannum) + '.dat'), datapoint, datapoint0)
    cfg['experiment'] = exp

lattice    = list(exp['lattice'])
energy     = float(exp['energy'])
energy0    = float(exp['energy0'])
azir       = list(exp['azir'])
imtemplate = exp['image_template']

psi = cfg['geometry']['psi']
hkl = np.array(cfg['geometry']['hkl'], dtype=float) * energy / energy0
hklint = np.round(hkl)

im      = imageio.imread(os.path.join(scanpath, imtemplate % imnum))
im      = ndimage.zoom(im, zoomval, order=3)
imdata  = np.copy(im)

px = cfg['geometry']['px_unscaled'] * zoomval
py = cfg['geometry']['py_unscaled'] * zoomval

thb      = ts.bragg(lattice, hkl, energy).th()[0]
thrange  = [thb - 27, thb + 10]
psirange = [psi - 180, psi + 180]
detvects = np.matrix([[1, 0, 0], [0, 0, 1]])
hkllist  = ts.pilkhlrange(lattice, hkl, energy, thrange[0], thrange[1]).hklscan(numsteps)
hkllistrange = [thrange[0], thrange[1], numsteps]

# ── fit / ROI-build settings (with defaults; honoured from config when present) ──
_roi         = cfg.get('roi', {})
width        = _roi.get('width_per_zoom', 45) * zoomval
comwidth     = _roi.get('comwidth_per_zoom', 5) * zoomval
_comp        = cfg.get('computation', {})
bravais      = _comp.get('bravais', 'icosahedral')
opt_method   = _comp.get('opt_method', 'COBYLA')
tolerance    = _comp.get('tolerance', 1e-6)
intensity    = _comp.get('intensity', 1)
threshold    = _comp.get('threshold', 0)
n_parallel_starts = _comp.get('n_parallel_starts', 4)
_flags       = cfg.get('flags', {})
detoptimize  = _flags.get('detoptimize', 1)
energyopt    = _flags.get('energyopt', 0)
strat        = ts.DE_Strategy['best1exp']

algo_display = ['COBYLA', 'Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC',
                'BH+Powell', 'BH+COBYLA', 'BH+NelderMead',
                'Diff. Evolution', 'Dual Annealing', 'Least-Sq (TRF)']
algo_methods = ['COBYLA', 'Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC',
                'BHPowell', 'BHCOBYLA', 'BHNelderMead',
                'GA', 'DualAnnealing', 'LSQ']

def _ref_pen(j, n, width=1.5):
    return pg.mkPen(pg.hsvColor(j / max(n, 1), 0.85, 0.95, 0.85), width=width)

# ── reflection list ────────────────────────────────────────────────────────────
ref_6d_manual = np.array([
    [-1, -1, -2, -1,  1,  1],
    [-1,  1, -1, -2, -1,  1],
    [ 1, -1, -1,  1,  2,  1],
    [ 1,  2,  1, -1, -1,  1],
    [ 2,  1,  1,  1,  1,  1],
    [-1,  0, -2, -2,  0,  1],
    [ 0,  2,  0, -2, -1,  1],
    [ 2,  0,  0,  1,  2,  1],
    [ 2,  2,  1,  0,  0,  1],
    [ 0, -1, -2,  0,  2,  1],
    [ 0,  1, -3, -3,  1,  4],
    [ 1,  3, -1, -3,  0,  4],
    [ 3,  3,  0, -1,  1,  4],
    [ 3,  1, -1,  0,  3,  4],
    [ 1,  0, -3, -1,  3,  4],
])

# ── initial guess (24-element, shared with workflow.py / dmsfit_ico_hkl) ─────────
#   [0-5]  a b c α β γ   [6-9]  psicor hcor kcor lcor   [10] detdist
#   [11-13] rotx roty rotz   [14] energy   [15-23] phason a11..a33
# hcor/kcor/lcor are reciprocal-index corrections (added to hkl by the _hkl engine);
# they default to 0 — manual alignment is done with psicor + the detector rotations.
initial_guess = np.array([
    6.461053, 6.461053, 6.461053, 90., 90., 90.,
    -2.171374, 0.0, 0.0, 0.0, 14480.587530 / 3 * zoomval,
     0.228572,  0.667038, -2.097034, energy + 0.00004667,
     0.001228,  0.000730,  0.000491,
     0.000507, -0.000951, -0.002741,
    -0.000441, -0.001405,  0.002354,
])

# ── slider definitions ─────────────────────────────────────────────────────────
# (label, ig_idx or 'h'/'k'/'l', half_range, fmt)
slider_defs = [
    ('a',       0,    0.2,   '%0.6f'),
    ('h',      'h',   1.2,   '%0.6f'),
    ('k',      'k',   1.2,   '%0.6f'),
    ('l',      'l',   1.5,   '%0.6f'),
    ('psicor',  6,    5.5,   '%0.6f'),
    ('hcor',    7,    1.0,   '%0.6f'),
    ('kcor',    8,    1.0,   '%0.6f'),
    ('lcor',    9,    1.0,   '%0.6f'),
    ('detdist',10,  300.0,   '%0.3f'),
    ('rotx',   11,    5.0,   '%0.6f'),
    ('roty',   12,    5.0,   '%0.6f'),
    ('rotz',   13,   10.0,   '%0.6f'),
    ('energy', 14,    0.5,   '%0.6f'),
    ('a11',    15,   0.05,   '%0.7f'),
    ('a12',    16,   0.05,   '%0.7f'),
    ('a13',    17,   0.05,   '%0.7f'),
    ('a21',    18,   0.05,   '%0.7f'),
    ('a22',    19,   0.05,   '%0.7f'),
    ('a23',    20,   0.05,   '%0.7f'),
    ('a31',    21,   0.05,   '%0.7f'),
    ('a32',    22,   0.05,   '%0.7f'),
    ('a33',    23,   0.05,   '%0.7f'),
]

# ── reflist helpers ────────────────────────────────────────────────────────────

def hklgen_ico_local(depth):
    rng = range(-depth, depth + 1)
    idx = np.array(list(itertools.product(rng, repeat=6)))
    return idx[np.any(idx != 0, axis=1)]

def build_reflist_from_6d(ref_6d_arr):
    p6d = ts.Projection6dArrayApproximant(ref_6d_arr, tau)
    r0  = p6d.reflection_6d()
    return np.array(r0[0]), np.array(r0[1])

def filter_6d_by_thresh(ref_6d_arr, thresh):
    if thresh <= 0:
        return ref_6d_arr
    return ref_6d_arr[np.any(np.abs(ref_6d_arr) >= thresh, axis=1)]

# ── shared _hkl engine (same as workflow.py / dmsfit_ico_hkl) ────────────────────
def extract_reduced(full_ig):
    """Reduced parameter vector consumed by dmsfit_ico_hkl.imcalc, keyed on the
    bravais / detoptimize / energyopt flags (identical to the workflow)."""
    if bravais == 'icosahedral':
        if detoptimize:
            idx = ([0,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23] if energyopt
                   else [0,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23])
        else:
            idx = ([0,6,7,8,9,14,15,16,17,18,19,20,21,22,23] if energyopt
                   else [0,6,7,8,9,15,16,17,18,19,20,21,22,23])
    elif bravais == 'icosahedral_fixed_a':
        if detoptimize:
            idx = ([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23] if energyopt
                   else [6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23])
        else:
            idx = ([6,7,8,13,14,15,16,17,18,19,20,21,22,23] if energyopt
                   else [6,7,8,14,15,16,17,18,19,20,21,22,23])
    elif bravais == 'cubic_no_strain':
        if detoptimize:
            idx = [0,6,7,8,9,10,11,12,13] if energyopt else [0,6,7,8,9,10,11,12]
        else:
            idx = [0,6,7,8,13] if energyopt else [0,6,7,8]
    else:
        raise ValueError('Unknown bravais: %s' % bravais)
    return np.asarray(full_ig, dtype=float)[idx]

def make_overlay_dms(reflist_, reflist2_, hkl_, imdata_, psirange_, thrange_,
                     azir_, psi_, px_, py_, ig):
    """Build a dmsfit_ico_hkl in calculator mode (dummy kernel/centres — only
    imcalc/dmsindex/dmslines are used for the live overlay).  This is the *same*
    engine the fit uses, so the overlay and the fit simulation match."""
    return ts.dmsfit_ico_hkl(
        np.matrix(reflist_), [thrange_[0], thrange_[1], numsteps],
        hklint, psirange_, width, np.zeros((1, 1)), np.zeros((1, 1, 1)),
        hkl_, detvects, imdata_, simsigma, azir_, psi_, px_, py_, scatv,
        bravais, bool(detoptimize), bool(energyopt),
        ig[10], ig[11], ig[12], ig[13], ig[14],
        np.matrix(reflist2_), list(ig[15:24]), ig[0])

# ── initial reflist ────────────────────────────────────────────────────────────
_rl, _rl2       = build_reflist_from_6d(ref_6d_manual)
full_reflist    = np.array(_rl)
full_reflist2   = np.array(_rl2)
full_reflist_6d = np.array(ref_6d_manual)

_ig0   = initial_guess.copy()

_dms_init = make_overlay_dms(
    full_reflist, full_reflist2, hkl, imdata, psirange, thrange,
    azir, psi, px, py, _ig0)

_dms_full_init = make_overlay_dms(
    full_reflist, full_reflist2, hkl, imdata, psirange, thrange,
    azir, psi, px, py, _ig0)


# ── FloatSlider (verbatim from workflow.py) ────────────────────────────────────

class FloatSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, label, val_init, val_min, val_max,
                 fmt='%0.6f', n_steps=10000, parent=None):
        super().__init__(parent)
        self._min = val_min
        self._max = val_max
        self._n   = n_steps
        self._fmt = fmt

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(4)

        lbl = QtWidgets.QLabel(label)
        lbl.setFixedWidth(52)
        lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self._sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._sl.setRange(0, n_steps)

        self._vl = QtWidgets.QLabel()
        self._vl.setFixedWidth(88)
        f = self._vl.font()
        f.setFamily('monospace')
        self._vl.setFont(f)

        row.addWidget(lbl)
        row.addWidget(self._sl, 1)
        row.addWidget(self._vl)

        self.setValue(val_init)
        self._sl.valueChanged.connect(self._emit)

    def _to_int(self, v):
        return int(round((v - self._min) / (self._max - self._min) * self._n))

    def _to_float(self, i):
        return self._min + i / self._n * (self._max - self._min)

    def setRange(self, val_min, val_max):
        cur = self.val
        self._min = val_min
        self._max = val_max
        self.setValue(min(max(cur, val_min), val_max))

    def setValue(self, v):
        self._sl.blockSignals(True)
        self._sl.setValue(max(0, min(self._n, self._to_int(v))))
        self._sl.blockSignals(False)
        self._vl.setText(self._fmt % v)

    @property
    def val(self):
        return self._to_float(self._sl.value())

    def _emit(self, i):
        v = self._to_float(i)
        self._vl.setText(self._fmt % v)
        self.valueChanged.emit(v)


# ── Background update worker ───────────────────────────────────────────────────

class UpdateWorker(QtCore.QThread):
    """Runs one vectorised dms.imcalc in a background thread; discards stale
    requests.  Emits ('discovery', (rows, cols)) for the full-reflist scatter or
    ('selected', dmslines) for the per-reflection selected curves."""
    done = QtCore.pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending  = None   # (ig, dms_ref, hkl, last_hkl_ref, mode)
        self._mutex    = QtCore.QMutex()
        self._cond     = QtCore.QWaitCondition()
        self._quit     = False
        self.idle      = threading.Event()
        self.idle.set()
        self.lattice   = list(lattice)
        self.thrange   = list(thrange)

    def submit(self, ig, dms_ref, hkl_arr, last_hkl_ref, mode):
        locker = QtCore.QMutexLocker(self._mutex)
        self._pending = (ig.copy(), dms_ref, hkl_arr.copy(), last_hkl_ref, mode)
        locker.unlock()
        self._cond.wakeOne()
        if not self.isRunning():
            self.start()

    def stop(self):
        locker = QtCore.QMutexLocker(self._mutex)
        self._quit = True
        locker.unlock()
        self._cond.wakeOne()
        self.wait()

    def run(self):
        while True:
            self._mutex.lock()
            while self._pending is None and not self._quit:
                self._cond.wait(self._mutex)
            if self._quit:
                self._mutex.unlock()
                return
            ig, dms_ref, hkl_arr, last_hkl_ref, mode = self._pending
            self._pending = None
            self._mutex.unlock()

            self.idle.clear()
            try:
                # The _hkl engine recomputes hkllist internally from self.hkl +
                # self.hkllistrange each imcalc, so we only push the current hkl.
                if not np.allclose(hkl_arr, last_hkl_ref):
                    dms_ref.hkl = hkl_arr.copy()
                    last_hkl_ref[:] = hkl_arr
                # When energy isn't a reduced (fit) parameter it comes from the
                # engine attribute, so keep the live energy slider in sync.
                if not energyopt:
                    dms_ref.energy = ig[14]

                dms_ref.imcalc(extract_reduced(ig))
                if mode == 'selected':
                    lines = [(np.copy(x), np.copy(y))
                             for x, y in (getattr(dms_ref, 'dmslines', None) or [])]
                    self.done.emit('selected', lines)
                else:
                    dmsindex = dms_ref.dmsindex
                    if len(dmsindex) == 2 and len(dmsindex[0]) > 0:
                        rows = np.asarray(dmsindex[0]).astype(float)
                        cols = np.asarray(dmsindex[1]).astype(float)
                    else:
                        rows = np.array([]); cols = np.array([])
                    self.done.emit('discovery', (rows, cols))
            except Exception as e:
                print('UpdateWorker error:', e)
            finally:
                self.idle.set()


# ── Fit worker (scipy optimiser in a background thread) ─────────────────────────

class FitWorker(QtCore.QThread):
    done    = QtCore.pyqtSignal(dict)
    error   = QtCore.pyqtSignal(str, float)
    stopped = QtCore.pyqtSignal(float)

    def __init__(self, dms, reduced, bounds, method, n_starts, parent=None):
        super().__init__(parent)
        self._dms        = dms
        self._reduced    = reduced.copy()
        self._bounds     = bounds
        self._method     = method
        self._n_starts   = n_starts
        self._t0         = time.time()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        dms     = self._dms
        reduced = self._reduced
        bounds  = self._bounds
        cur     = self._method
        ev      = self._stop_event

        def _fit_checked(x):
            if ev.is_set():
                raise StopIteration('stopped')
            return dms.fit(x)

        def _cb_check(*_a, **_k):
            return ev.is_set()

        try:
            from scipy.optimize import (minimize, differential_evolution,
                                        basinhopping, dual_annealing, least_squares)
            from joblib import Parallel, delayed
            if cur == 'GA':
                res = differential_evolution(_fit_checked, bounds, strategy=strat,
                                             polish=not ev.is_set(), workers=1,
                                             callback=_cb_check)
            elif cur == 'DualAnnealing':
                # Generalized simulated annealing — global, bounded, derivative-free.
                # _fit_checked raises StopIteration on stop; the callback is a
                # secondary stop hook (returns True to abort).
                def _da_cb(x, f, context):
                    return ev.is_set()
                res = dual_annealing(_fit_checked, bounds, callback=_da_cb)
            elif cur == 'LSQ':
                # Exploit the least-squares structure: optimise the per-ROI centre
                # residual vector directly with Trust-Region-Reflective + a robust
                # loss (downweights failed-ROI fallback rows).
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                def _resid(x):
                    if ev.is_set():
                        raise StopIteration('stopped')
                    return dms.residuals(x)
                res = least_squares(_resid, reduced, bounds=(lo, hi),
                                    method='trf', loss='soft_l1',
                                    xtol=tolerance, ftol=tolerance)
            elif cur in ('L-BFGS-B', 'TNC'):
                # Bounded finite-difference-gradient locals, multi-started.
                n = self._n_starts
                rng = np.random.default_rng(42)
                starts = [reduced] + [
                    reduced + rng.uniform(-0.5, 0.5, reduced.shape)
                    for _ in range(n - 1)]
                def _run_one_b(s):
                    if ev.is_set():
                        raise StopIteration('stopped')
                    _d = copy.deepcopy(dms)
                    return minimize(_d.fit, s, method=cur, bounds=bounds,
                                    tol=tolerance)
                results = Parallel(n_jobs=n, prefer='threads')(
                    delayed(_run_one_b)(s) for s in starts)
                res = min(results, key=lambda r: r.fun)
            elif cur in ('BHPowell', 'BHCOBYLA', 'BHNelderMead'):
                bh_map = {'BHPowell':     ('Powell',      150),
                          'BHCOBYLA':     ('COBYLA',      400),
                          'BHNelderMead': ('Nelder-Mead', 400)}
                method, niter = bh_map[cur]
                res = basinhopping(_fit_checked, reduced,
                                   minimizer_kwargs={"method": method},
                                   niter=niter, callback=_cb_check)
            else:
                n = self._n_starts
                rng = np.random.default_rng(42)
                starts = [reduced] + [
                    reduced + rng.uniform(-0.5, 0.5, reduced.shape)
                    for _ in range(n - 1)]
                def _run_one(s):
                    if ev.is_set():
                        raise StopIteration('stopped')
                    _d = copy.deepcopy(dms)
                    return minimize(_d.fit, s, method=cur, tol=tolerance,
                                    options={'xtol': tolerance, 'ftol': tolerance})
                results = Parallel(n_jobs=n, prefer='threads')(
                    delayed(_run_one)(s) for s in starts)
                res = min(results, key=lambda r: r.fun)

            elapsed = time.time() - self._t0
            dms.hkllistrange[2] = numsteps
            opt, simim, dmsindex, dataim, inputarray = dms.full(res.x)
            dmslines = [(np.copy(x), np.copy(y)) for x, y in dms.dmslines] \
                if hasattr(dms, 'dmslines') else []
            self.done.emit({
                'opt': opt, 'simim': simim, 'dmslines': dmslines,
                'res_x': np.array(res.x),
                'dmsindex': dmsindex, 'dataim': np.array(dataim),
                'inputarray': np.array(inputarray),
                'elapsed': elapsed, 'method': cur})
        except StopIteration:
            self.stopped.emit(time.time() - self._t0)
        except Exception as e:
            self.error.emit(str(e), time.time() - self._t0)
            import traceback; traceback.print_exc()


# ── Main window ────────────────────────────────────────────────────────────────

class DMSSlider(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'DMS Slider v3 — scan {scannum}  dp={datapoint}  '
                            f'E={energy:.4f} keV')

        self.ig            = initial_guess.copy()
        self._hkl          = hkl.copy()
        self._last_hkl     = hkl.copy()
        self.full_reflist    = full_reflist.copy()
        self.full_reflist2   = full_reflist2.copy()
        self.full_reflist_6d = full_reflist_6d.copy()
        self._dms      = _dms_init
        self._dms_full = _dms_full_init
        # vectorised engine over the currently-selected reflections (built on demand)
        self._sel_dms      = None
        self._sel_order    = []      # arc items in reflist-row order
        self._sel_last_hkl = np.full(3, np.inf)

        # scan-specific state (updated when a different scan is loaded)
        self._lattice        = list(lattice)
        self._thrange        = list(thrange)
        self._px             = px
        self._py             = py
        self._psi            = psi
        self._azir           = list(azir)
        self._imdata         = imdata.copy()
        self._hkl_ref        = hkl.copy()
        self._hklint         = hklint.copy()
        self._hkllist        = hkllist
        self._psirange       = list(psirange)
        self._scanpath       = scanpath
        self._scannum        = scannum
        self._datapoint      = datapoint
        self._datapoint0     = datapoint0
        self._imtemplate     = imtemplate
        self._pending_scan_path = scanpath + str(scannum) + '.dat'
        self._initial_guess  = initial_guess.copy()
        self._en_scan        = energy        # raw scan energy (no user offset)
        self._cfg            = cfg           # live config (shown in the Config table)
        # default workflow template = the example config shipped with the package
        _default_tmpl = os.path.join(
            CONFIGS, 'fit_fivefold_axis_AlPdMn_Not_Annealed_2M_2ROIS_internal_hkl.json')
        self._workflow_template = _default_tmpl if os.path.exists(_default_tmpl) else ''

        # pick state
        self._geo_mode        = False
        self._psi_tol         = 3.0
        self._use_auto        = False
        self._pending_picks   = []
        self._pending_markers = []
        self._pick_items      = []
        self._arc_to_6d        = {}
        self._arc_to_list_item = {}   # id(arc) → QListWidgetItem
        self._suppress         = False

        # fit / ROI-build state (populated on demand by "Build curves")
        self._fitting       = False
        self._fit_worker    = None
        self._fit_dms       = None
        self._kernel        = None
        self._centres       = None
        self._linedatax     = None
        self._linedatay     = None
        self._imcoeffs      = None
        self._reflist_fit   = None
        self._reflist2_fit  = None
        self._ref_6d_fit    = None
        self._exp_curves    = []
        self._sim_curves    = []
        self._roi_plots     = []
        self._exp_centre_lines = []
        self._sim_centre_lines = []
        self._centre_override_rois = set()
        # centre overrides restored from a session file but not yet applied
        # (applied the next time "Build curves" rebuilds the ROI centres)
        self._pending_centre_overrides = {}
        # last optimiser result, kept so it can be captured in the session
        self._last_res_x   = None
        self._last_fit_info = None
        # full output of the last completed fit (dms.full), kept so it can be
        # written to Processing/ on request ("Save fit → Processing")
        self._last_fit_output = None
        self._selected_roi  = None
        self._active_method = opt_method if opt_method in algo_methods else algo_methods[0]

        self._worker = UpdateWorker()
        self._worker.done.connect(self._on_update_done,
                                  QtCore.Qt.QueuedConnection)

        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(200)
        self._update_timer.timeout.connect(self._do_update)

        self._build_ui()
        self._do_update()
        # Offer to restore the previous session once the event loop is running.
        QtCore.QTimer.singleShot(0, self._maybe_restore_session)

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root_layout.addWidget(splitter)

        # ── Image panel (left) ─────────────────────────────────────────────────
        gw = pg.GraphicsLayoutWidget()
        gw.setMinimumWidth(500)
        self._vb = gw.addViewBox()
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)
        self._vb.setMenuEnabled(False)

        self._img_item = pg.ImageItem()
        self._vb.addItem(self._img_item)
        try:
            cmap = pg.colormap.get(colmap, source='matplotlib')
        except Exception:
            cmap = pg.colormap.get('grey')
        self._img_item.setColorMap(cmap)
        self._img_item.setImage(imdata, autoLevels=False)
        self._img_item.setLevels(colourlim)

        self._dms_scatter = pg.ScatterPlotItem(
            size=3, pen=None, brush=pg.mkBrush(255, 60, 60, 200))
        self._vb.addItem(self._dms_scatter)

        self._coord_lbl = QtWidgets.QLabel('row —   col —   I=—')
        self._coord_lbl.setAlignment(QtCore.Qt.AlignCenter)
        f = self._coord_lbl.font()
        f.setFamily('monospace')
        f.setPointSize(8)
        self._coord_lbl.setFont(f)

        self._mouse_proxy = pg.SignalProxy(
            gw.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)

        img_col = QtWidgets.QVBoxLayout()
        img_col.addWidget(gw, 1)
        img_col.addWidget(self._coord_lbl)
        img_w = QtWidgets.QWidget()
        img_w.setLayout(img_col)
        splitter.addWidget(img_w)

        # Click handler on scene
        gw.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self._gw = gw

        # ── Control panel (right) ──────────────────────────────────────────────
        ctrl_col = QtWidgets.QVBoxLayout()
        ctrl_col.setSpacing(4)
        ctrl_w = QtWidgets.QWidget()
        ctrl_w.setLayout(ctrl_col)
        ctrl_w.setMinimumWidth(320)

        # ── Scan loader ────────────────────────────────────────────────────────
        scan_box = QtWidgets.QGroupBox('Scan')
        sbl = QtWidgets.QGridLayout(scan_box)
        sbl.setSpacing(4)
        sbl.setContentsMargins(4, 4, 4, 4)

        self._lbl_scan_path = QtWidgets.QLabel(
            os.path.basename(self._pending_scan_path))
        self._lbl_scan_path.setWordWrap(True)
        f_sp = self._lbl_scan_path.font()
        f_sp.setFamily('monospace')
        f_sp.setPointSize(7)
        self._lbl_scan_path.setFont(f_sp)
        sbl.addWidget(self._lbl_scan_path, 0, 0, 1, 4)

        btn_browse = QtWidgets.QPushButton('Browse…')
        btn_browse.clicked.connect(self._on_browse_scan)
        sbl.addWidget(btn_browse, 1, 0)

        sbl.addWidget(QtWidgets.QLabel('dp0'), 1, 2)
        self._sb_dp0 = QtWidgets.QSpinBox()
        self._sb_dp0.setRange(0, 9999)
        self._sb_dp0.setValue(self._datapoint0)
        sbl.addWidget(self._sb_dp0, 1, 3)

        btn_load_scan = QtWidgets.QPushButton('Load Scan')
        btn_load_scan.setStyleSheet('background: #102020; color: #aaffff')
        btn_load_scan.setToolTip('Load the selected .dat scan and detector image')
        btn_load_scan.clicked.connect(self._on_load_scan)
        sbl.addWidget(btn_load_scan, 2, 0)

        sbl.addWidget(QtWidgets.QLabel('dp'), 2, 2)
        self._sb_dp = QtWidgets.QSpinBox()
        self._sb_dp.setRange(0, 9999)
        self._sb_dp.setValue(self._datapoint)
        sbl.addWidget(self._sb_dp, 2, 3)

        self._lbl_scan_info = QtWidgets.QLabel(
            'E=%.4f keV' % energy)
        f_si = self._lbl_scan_info.font()
        f_si.setFamily('monospace')
        f_si.setPointSize(7)
        self._lbl_scan_info.setFont(f_si)
        sbl.addWidget(self._lbl_scan_info, 3, 0, 1, 4)

        ctrl_col.addWidget(scan_box)

        # ── Fit (build integrated curves for the checked reflections, then fit) ──
        fit_box = QtWidgets.QGroupBox('Fit')
        fitl = QtWidgets.QGridLayout(fit_box)
        fitl.setSpacing(4)
        fitl.setContentsMargins(4, 4, 4, 4)

        btn_build = QtWidgets.QPushButton('Build curves')
        btn_build.setStyleSheet('background: #102030; color: #aaccff')
        btn_build.clicked.connect(self._on_build_curves)
        fitl.addWidget(btn_build, 0, 0, 1, 2)

        # Number of points along the integrated curves (hkl scan resolution).
        # Drives Build curves and the final fit (numsteps global).
        _pts_lbl = QtWidgets.QLabel('Points')
        _pts_lbl.setToolTip('Number of points sampled along each integrated curve '
                            '(hkl scan resolution used by Build curves and the fit)')
        self._sb_numsteps = QtWidgets.QSpinBox()
        self._sb_numsteps.setRange(50, 20000)
        self._sb_numsteps.setSingleStep(50)
        self._sb_numsteps.setValue(int(numsteps))
        self._sb_numsteps.setToolTip(_pts_lbl.toolTip())
        self._sb_numsteps.valueChanged.connect(self._on_numsteps_changed)
        fitl.addWidget(_pts_lbl, 1, 0)
        fitl.addWidget(self._sb_numsteps, 1, 1)

        # ROI integration half-width in pixels (width global).
        _w_lbl = QtWidgets.QLabel('Width (px)')
        _w_lbl.setToolTip('ROI integration width in pixels (rebuild curves to apply)')
        self._sb_width = QtWidgets.QSpinBox()
        self._sb_width.setRange(3, 500)
        self._sb_width.setValue(int(width))
        self._sb_width.setToolTip(_w_lbl.toolTip())
        self._sb_width.valueChanged.connect(self._on_width_changed)
        fitl.addWidget(_w_lbl, 2, 0)
        fitl.addWidget(self._sb_width, 2, 1)

        # Simulation Gaussian blur sigma applied to the simulated DMS image
        # (simsigma global; the engine applies it live each imcalc).
        _sig_lbl = QtWidgets.QLabel('Sigma')
        _sig_lbl.setToolTip('Gaussian blur sigma applied to the simulated DMS '
                            'overlay/curves (updates the overlay live)')
        self._sb_simsigma = QtWidgets.QDoubleSpinBox()
        self._sb_simsigma.setRange(0.0, 50.0)
        self._sb_simsigma.setSingleStep(0.5)
        self._sb_simsigma.setDecimals(2)
        self._sb_simsigma.setValue(float(simsigma))
        self._sb_simsigma.setToolTip(_sig_lbl.toolTip())
        self._sb_simsigma.valueChanged.connect(self._on_simsigma_changed)
        fitl.addWidget(_sig_lbl, 3, 0)
        fitl.addWidget(self._sb_simsigma, 3, 1)

        self._algo_combo = QtWidgets.QComboBox()
        for disp in algo_display:
            self._algo_combo.addItem(disp)
        self._algo_combo.setCurrentIndex(algo_methods.index(self._active_method))
        self._algo_combo.currentIndexChanged.connect(
            lambda i: self._on_algo(algo_methods[i]))
        fitl.addWidget(self._algo_combo, 4, 0, 1, 2)

        self._btn_fit = QtWidgets.QPushButton('Fit')
        self._btn_fit.setStyleSheet('background: #1a5c1a; color: #ccffcc; font-weight: bold')
        self._btn_fit.clicked.connect(self._do_fit)
        fitl.addWidget(self._btn_fit, 5, 0)
        self._btn_stop = QtWidgets.QPushButton('Stop')
        self._btn_stop.setStyleSheet('background: #5c1a1a; color: #ffcccc; font-weight: bold')
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop_fit)
        fitl.addWidget(self._btn_stop, 5, 1)

        btn_wf_export = QtWidgets.QPushButton('Export Fit Config')
        btn_wf_export.setStyleSheet('background: #103018; color: #bfe6c8')
        btn_wf_export.setToolTip('Export a fit.py-compatible workflow config JSON '
                                 'for batch (non-interactive) fitting')
        btn_wf_export.clicked.connect(self._on_export_workflow_json)
        fitl.addWidget(btn_wf_export, 6, 0, 1, 2)

        self._btn_save_fit = QtWidgets.QPushButton('Save fit → Processing')
        self._btn_save_fit.setStyleSheet('background: #2a2a10; color: #e6e0bf')
        self._btn_save_fit.setToolTip('Write a timestamped Processing/ snapshot of '
                                      'the last completed fit (overlay PNG, ROI plot, '
                                      'res.x.txt, Result.txt, config + code snapshots)')
        self._btn_save_fit.setEnabled(False)
        self._btn_save_fit.clicked.connect(self._on_save_fit_processing)
        fitl.addWidget(self._btn_save_fit, 7, 0, 1, 2)

        ctrl_col.addWidget(fit_box)

        # ── Editable config table (metadata + key scalars) ───────────────────────
        cfg_box = QtWidgets.QGroupBox('Config')
        cbl = QtWidgets.QVBoxLayout(cfg_box)
        cbl.setContentsMargins(4, 4, 4, 4)
        self._cfgtable = ConfigTable()
        self._cfgtable.set_config(self._cfg)
        self._cfgtable.set_save_path(
            cfg_path or os.path.join(os.getcwd(), 'config_%s.json' % self._scannum))
        self._cfgtable.configChanged.connect(self._on_cfg_table_changed)
        self._cfgtable.setMaximumHeight(200)
        cbl.addWidget(self._cfgtable)
        ctrl_col.addWidget(cfg_box)

        # Sliders in scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        vbox  = QtWidgets.QVBoxLayout(inner)
        vbox.setSpacing(1)
        vbox.setContentsMargins(0, 0, 0, 0)

        self._sliders = {}
        for label, idx, half, fmt in slider_defs:
            if idx == 'h':
                centre = float(self._hkl[0])
            elif idx == 'k':
                centre = float(self._hkl[1])
            elif idx == 'l':
                centre = float(self._hkl[2])
            else:
                centre = float(self.ig[idx])
            fs = FloatSlider(label, centre, centre - half, centre + half, fmt)
            fs.valueChanged.connect(self._on_slider_changed)
            vbox.addWidget(fs)
            self._sliders[label] = fs

        vbox.addStretch()
        scroll.setWidget(inner)
        ctrl_col.addWidget(scroll)

        # Selected arcs list
        arc_box = QtWidgets.QGroupBox('Selected reflections')
        arc_box_l = QtWidgets.QVBoxLayout(arc_box)
        arc_box_l.setSpacing(2)
        arc_box_l.setContentsMargins(4, 4, 4, 4)
        hint = QtWidgets.QLabel('Left-click arc to add  ·  right-click arc or item to remove')
        hint.setWordWrap(True)
        f_hint = hint.font()
        f_hint.setPointSize(7)
        hint.setFont(f_hint)
        arc_box_l.addWidget(hint)
        self._arc_list = QtWidgets.QListWidget()
        self._arc_list.setMinimumHeight(280)
        self._arc_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._arc_list.customContextMenuRequested.connect(self._on_list_context_menu)
        self._arc_list.itemChanged.connect(self._on_list_item_changed)
        f_list = self._arc_list.font()
        f_list.setFamily('monospace')
        f_list.setPointSize(8)
        self._arc_list.setFont(f_list)
        arc_box_l.addWidget(self._arc_list)

        refl_btn_row = QtWidgets.QHBoxLayout()
        btn_save_refl = QtWidgets.QPushButton('Save reflections')
        btn_save_refl.setStyleSheet('background: #102030; color: #cce0ff')
        btn_save_refl.setToolTip('Save just the selected reflections (and checked '
                                 'state) to a reusable JSON file')
        btn_save_refl.clicked.connect(self._on_save_reflections)
        refl_btn_row.addWidget(btn_save_refl)
        btn_load_refl = QtWidgets.QPushButton('Load reflections')
        btn_load_refl.setStyleSheet('background: #201030; color: #ddccff')
        btn_load_refl.setToolTip('Load a reflection list into the selection '
                                 '(leaves scan and geometry unchanged)')
        btn_load_refl.clicked.connect(self._on_load_reflections)
        refl_btn_row.addWidget(btn_load_refl)
        arc_box_l.addLayout(refl_btn_row)

        # Live Curve: the overlay lines always update live; when this is checked
        # the ROI integrated curves are also recomputed on every slider move
        # (heavier, but lets you watch the fit quality as you refine).
        self._chk_live_curve = QtWidgets.QCheckBox('Live Curve (also update ROI curves)')
        self._chk_live_curve.setChecked(False)
        f_la = self._chk_live_curve.font(); f_la.setPointSize(7)
        self._chk_live_curve.setFont(f_la)
        self._chk_live_curve.toggled.connect(self._on_live_curve_toggled)
        arc_box_l.addWidget(self._chk_live_curve)
        ctrl_col.addWidget(arc_box)

        # Reflist group
        rg = QtWidgets.QGroupBox('Reflist')
        rgl = QtWidgets.QGridLayout(rg)
        rgl.setSpacing(4)

        self._chk_auto = QtWidgets.QCheckBox('Auto reflist')
        rgl.addWidget(self._chk_auto, 0, 0, 1, 2)

        rgl.addWidget(QtWidgets.QLabel('Depth'), 1, 0)
        self._sb_depth = QtWidgets.QSpinBox()
        self._sb_depth.setRange(1, 20)
        self._sb_depth.setValue(1)
        rgl.addWidget(self._sb_depth, 1, 1)

        rgl.addWidget(QtWidgets.QLabel('Max N'), 1, 2)
        self._sb_max_n = QtWidgets.QSpinBox()
        self._sb_max_n.setRange(1, 50000)
        self._sb_max_n.setValue(30)
        rgl.addWidget(self._sb_max_n, 1, 3)

        rgl.addWidget(QtWidgets.QLabel('Thresh'), 2, 0)
        self._sb_thresh = QtWidgets.QSpinBox()
        self._sb_thresh.setRange(0, 20)
        self._sb_thresh.setValue(0)
        rgl.addWidget(self._sb_thresh, 2, 1)

        rgl.addWidget(QtWidgets.QLabel('psi_tol'), 2, 2)
        self._sb_psi_tol = QtWidgets.QDoubleSpinBox()
        self._sb_psi_tol.setRange(0.0, 30.0)
        self._sb_psi_tol.setSingleStep(0.5)
        self._sb_psi_tol.setDecimals(1)
        self._sb_psi_tol.setValue(self._psi_tol)
        rgl.addWidget(self._sb_psi_tol, 2, 3)

        n_total = self.full_reflist.shape[0]
        init_n  = min(30, n_total)

        rgl.addWidget(QtWidgets.QLabel('N refs'), 3, 0)
        self._sl_n_refs = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._sl_n_refs.setRange(1, max(1, init_n))
        self._sl_n_refs.setValue(init_n)
        rgl.addWidget(self._sl_n_refs, 3, 1, 1, 3)

        rgl.addWidget(QtWidgets.QLabel('Offset'), 4, 0)
        self._sl_offset = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._sl_offset.setRange(0, max(0, n_total - 1))
        self._sl_offset.setValue(0)
        rgl.addWidget(self._sl_offset, 4, 1, 1, 3)

        self._lbl_nrefs = QtWidgets.QLabel('N=%d reflections' % n_total)
        rgl.addWidget(self._lbl_nrefs, 5, 0, 1, 4)
        ctrl_col.addWidget(rg)

        # Pick / Identify group
        pg_box = QtWidgets.QGroupBox('Pick / Identify')
        pgl = QtWidgets.QGridLayout(pg_box)
        pgl.setSpacing(4)
        self._btn_clear = QtWidgets.QPushButton('Clear picks')
        self._btn_clear.setStyleSheet('background: #3a1a1a; color: #ffcccc')
        pgl.addWidget(self._btn_clear, 0, 0)
        self._chk_geo = QtWidgets.QCheckBox('Geo 3-click')
        pgl.addWidget(self._chk_geo, 0, 1)
        self._lbl_pick = QtWidgets.QLabel('')
        self._lbl_pick.setWordWrap(True)
        self._lbl_pick.setMinimumHeight(32)
        f2 = self._lbl_pick.font()
        f2.setFamily('monospace')
        self._lbl_pick.setFont(f2)
        pgl.addWidget(self._lbl_pick, 1, 0, 1, 2)
        ctrl_col.addWidget(pg_box)

        # Reset / Print / Session row
        btn_row = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton('Reset')
        btn_reset.setStyleSheet('background: #4a4a10; color: #ffffcc')
        btn_reset.clicked.connect(self._on_reset)
        btn_print = QtWidgets.QPushButton('Print ig')
        btn_print.setStyleSheet('background: #103050; color: #cce0ff')
        btn_print.clicked.connect(self._on_print)
        btn_save = QtWidgets.QPushButton('Save Session')
        btn_save.setStyleSheet('background: #103010; color: #ccffcc')
        btn_save.setToolTip('Save the whole workflow (scan, geometry, selected '
                            'reflections, centre overrides and fit) to a JSON file')
        btn_save.clicked.connect(self._on_save_json)
        btn_load = QtWidgets.QPushButton('Load Session')
        btn_load.setStyleSheet('background: #201030; color: #ddccff')
        btn_load.setToolTip('Restore a previously saved workflow session from a JSON file')
        btn_load.clicked.connect(self._on_load_json)
        btn_clear = QtWidgets.QPushButton('Clear Session')
        btn_clear.setStyleSheet('background: #401010; color: #ffcccc')
        btn_clear.setToolTip('Reset the whole workflow: geometry, selected '
                             'reflections, built curves, centre overrides and fit')
        btn_clear.clicked.connect(self._on_clear_session)
        btn_row.addWidget(btn_reset)
        btn_row.addWidget(btn_print)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_clear)
        ctrl_col.addLayout(btn_row)

        # Status label
        self._status = QtWidgets.QLabel('Ready')
        self._status.setWordWrap(True)
        f3 = self._status.font()
        f3.setFamily('monospace')
        f3.setPointSize(8)
        self._status.setFont(f3)
        ctrl_col.addWidget(self._status)
        ctrl_col.addStretch(1)

        splitter.addWidget(ctrl_w)

        # ── ROI integrated-curve grid (right pane; populated by "Build curves") ──
        roi_w = QtWidgets.QWidget()
        roi_col = QtWidgets.QVBoxLayout(roi_w)
        roi_col.setContentsMargins(2, 2, 2, 2)
        self._roi_grid = pg.GraphicsLayoutWidget()
        self._roi_grid.scene().sigMouseClicked.connect(self._on_roi_grid_clicked)
        roi_col.addWidget(self._roi_grid, 1)
        self._roi_coord_lbl = QtWidgets.QLabel('build curves to integrate ROIs')
        self._roi_coord_lbl.setAlignment(QtCore.Qt.AlignCenter)
        f4 = self._roi_coord_lbl.font()
        f4.setFamily('monospace'); f4.setPointSize(8)
        self._roi_coord_lbl.setFont(f4)
        roi_col.addWidget(self._roi_coord_lbl)
        self._roi_mouse_proxy = pg.SignalProxy(
            self._roi_grid.scene().sigMouseMoved, rateLimit=60,
            slot=self._on_roi_mouse_moved)
        splitter.addWidget(roi_w)

        splitter.setSizes([900, 340, 460])
        self.resize(1700, 860)

        # Connect controls
        self._chk_auto.stateChanged.connect(
            lambda s: (setattr(self, '_use_auto', s == QtCore.Qt.Checked),
                       self._regenerate_reflist()))
        self._sb_depth.valueChanged.connect(lambda _: self._regenerate_reflist())
        self._sb_max_n.valueChanged.connect(lambda _: self._regenerate_reflist())
        self._sb_thresh.valueChanged.connect(lambda _: self._regenerate_reflist())
        self._sb_psi_tol.valueChanged.connect(
            lambda v: setattr(self, '_psi_tol', float(v)))
        self._sl_n_refs.valueChanged.connect(self._on_slice_changed)
        self._sl_offset.valueChanged.connect(self._on_slice_changed)
        self._btn_clear.clicked.connect(self._on_clear_picks)
        self._chk_geo.stateChanged.connect(
            lambda s: setattr(self, '_geo_mode', s == QtCore.Qt.Checked))

    # ── Update pipeline ────────────────────────────────────────────────────────

    def _on_slider_changed(self, _=None):
        if not self._suppress:
            self._update_timer.start()

    def _sync_ig(self):
        for label, idx, *_ in slider_defs:
            fs = self._sliders[label]
            if idx == 'h':
                self._hkl[0] = fs.val
            elif idx == 'k':
                self._hkl[1] = fs.val
            elif idx == 'l':
                self._hkl[2] = fs.val
            else:
                self.ig[idx] = fs.val
        self.ig[1] = self.ig[2] = self.ig[0]
        self.ig[3] = self.ig[4] = self.ig[5] = 90.0

    def _do_update(self):
        self._sync_ig()
        self._worker.lattice = self._lattice
        self._worker.thrange = self._thrange
        if self._sel_dms is not None and self._sel_order:
            # Selected reflections: one vectorised imcalc → per-reflection lines.
            # Always live (responsive); the ROI curves are updated separately.
            self._worker.submit(self.ig, self._sel_dms, self._hkl,
                                self._sel_last_hkl, 'selected')
        else:
            # Discovery: full-reflist slice scatter (only when nothing selected).
            self._worker.submit(self.ig, self._dms, self._hkl,
                                self._last_hkl, 'discovery')

    def _on_update_done(self, mode, payload):
        if mode == 'selected':
            self._dms_scatter.setData(x=[], y=[])
            lines = payload or []
            for k, arc in enumerate(self._sel_order):
                if k < len(lines):
                    x = np.asarray(lines[k][0], dtype=float)
                    y = np.asarray(lines[k][1], dtype=float)
                    m = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[m], y[m]
                    arc.setData(x=x, y=y)
                    arc._x_data, arc._y_data = x, y
                else:
                    arc.setData(x=[], y=[])
            self._maybe_update_live_curves()
        else:
            rows, cols = payload
            self._dms_scatter.setData(x=cols, y=rows)
        self._status.setText('Ready')

    def _maybe_update_live_curves(self):
        """When 'Live Curve' is on, recompute the ROI integrated curves at the
        current geometry (heavier — runs only if curves have been built)."""
        if (getattr(self, '_chk_live_curve', None) is None
                or not self._chk_live_curve.isChecked()
                or self._fit_dms is None):
            return
        try:
            self._fit_dms.imcalc(extract_reduced(self.ig))
            self._try_draw_sim_lines()
        except Exception:
            pass

    def _selected_arcs(self):
        """Checked arcs in list order, with their 6D indices."""
        arcs, sel6d = [], []
        for i in range(self._arc_list.count()):
            item = self._arc_list.item(i)
            if item.checkState() != QtCore.Qt.Checked:
                continue
            arc = item.data(QtCore.Qt.UserRole)
            h6d = self._arc_to_6d.get(id(arc)) if arc is not None else None
            if arc is not None and h6d is not None:
                arcs.append(arc)
                sel6d.append([int(v) for v in h6d])
        return arcs, np.array(sel6d)

    def _rebuild_selected_engine(self):
        """(Re)build the single vectorised engine over the checked reflections so
        the overlay draws only those, in one imcalc."""
        arcs, sel6d = self._selected_arcs()
        self._sel_order = arcs
        # Clear list arcs that are currently unchecked (candidate previews that
        # were never added to the list keep their static preview).
        for list_item in self._arc_to_list_item.values():
            if list_item.checkState() != QtCore.Qt.Checked:
                arc = list_item.data(QtCore.Qt.UserRole)
                if arc is not None:
                    arc.setData(x=[], y=[])
        if len(arcs) == 0:
            self._sel_dms = None
            return
        rl, rl2 = build_reflist_from_6d(sel6d)
        self._sel_dms = make_overlay_dms(
            rl, rl2, self._hkl, self._imdata, self._psirange, self._thrange,
            self._azir, self._psi, self._px, self._py, self.ig)
        self._sel_last_hkl = np.full(3, np.inf)

    def _on_selection_changed(self):
        self._rebuild_selected_engine()
        self._do_update()

    def _on_live_curve_toggled(self, checked):
        if checked:
            self._maybe_update_live_curves()
        self._status.setText('Live Curve on' if checked else 'Live Curve off')

    def _prep_arc_engine(self):
        """Point the full-reflist overlay engine at the current hkl/theta range
        (fine numsteps) before tracing single-reflection arcs."""
        self._dms_full.hkl = self._hkl.copy()
        self._dms_full.hkllistrange = [self._thrange[0], self._thrange[1], numsteps]

    def _arc_xy(self):
        """Return the (x=cols, y=rows) locus of the single reflection currently
        loaded in self._dms_full, from its dmslines (NaN separators stripped)."""
        lines = getattr(self._dms_full, 'dmslines', None)
        if not lines:
            return np.array([]), np.array([])
        x = np.asarray(lines[0][0], dtype=float)
        y = np.asarray(lines[0][1], dtype=float)
        m = ~(np.isnan(x) | np.isnan(y))
        return x[m], y[m]


    # ── Reflist management ─────────────────────────────────────────────────────

    def _regenerate_reflist(self):
        depth  = self._sb_depth.value()
        thresh = self._sb_thresh.value()
        max_n  = self._sb_max_n.value()
        if self._use_auto:
            src = hklgen_ico_local(depth) if not hasattr(ts, 'hklgen_ico') \
                  else np.array(ts.hklgen_ico(depth).v())
        else:
            src = np.array(ref_6d_manual)
        src = filter_6d_by_thresh(src, thresh)
        if src.shape[0] == 0:
            self._status.setText('Threshold removed all reflections — lower Thresh')
            return
        rl, rl2 = build_reflist_from_6d(src)
        self.full_reflist    = rl
        self.full_reflist2   = rl2
        self.full_reflist_6d = src
        n_total = rl.shape[0]
        init_n  = min(max_n, n_total)
        self._sl_n_refs.blockSignals(True)
        self._sl_offset.blockSignals(True)
        self._sl_n_refs.setRange(1, max(1, init_n))
        self._sl_n_refs.setValue(init_n)
        self._sl_offset.setRange(0, max(0, n_total - 1))
        self._sl_offset.setValue(0)
        self._sl_n_refs.blockSignals(False)
        self._sl_offset.blockSignals(False)
        self._lbl_nrefs.setText('N=%d  (thresh=%d)' % (n_total, thresh))
        self._rebuild_dms_slice()
        self._do_update()

    def _rebuild_dms_slice(self):
        offset  = self._sl_offset.value()
        n       = self._sl_n_refs.value()
        n_total = self.full_reflist.shape[0]
        end     = min(offset + n, n_total)
        if offset >= end:
            return
        rl  = self.full_reflist[offset:end]
        rl2 = self.full_reflist2[offset:end]
        self._dms = make_overlay_dms(
            rl, rl2, self._hkl, self._imdata, self._psirange, self._thrange,
            self._azir, self._psi, self._px, self._py, self.ig)
        self._last_hkl = np.full(3, np.inf)  # force hkl push on next update

    def _on_slice_changed(self, _=None):
        self._rebuild_dms_slice()
        self._update_timer.start()

    # ── Mouse / image coordinate tracking ─────────────────────────────────────

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self._vb.sceneBoundingRect().contains(pos):
            pt  = self._vb.mapSceneToView(pos)
            row = int(pt.y())
            col = int(pt.x())
            if 0 <= row < self._imdata.shape[0] and 0 <= col < self._imdata.shape[1]:
                val = self._imdata[row, col]
                self._coord_lbl.setText(
                    'row %4d   col %4d   I=%.1f' % (row, col, val))
            else:
                self._coord_lbl.setText(
                    'row %4d   col %4d   I=—' % (row, col))
        else:
            self._coord_lbl.setText('row —   col —   I=—')

    # ── Click / pick handling ──────────────────────────────────────────────────

    def _on_scene_clicked(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            arc = self._nearest_arc_at(event.scenePos())
            if arc is not None:
                hkl_6d = self._arc_to_6d[id(arc)]
                self._add_arc_to_list(hkl_6d, arc)
            return
        if event.button() == QtCore.Qt.RightButton:
            arc = self._nearest_arc_at(event.scenePos())
            if arc is not None:
                self._remove_arc_from_list(arc)
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        pos    = event.scenePos()
        vb_pos = self._vb.mapSceneToView(pos)
        col, row = vb_pos.x(), vb_pos.y()
        h, w = imdata.shape[0], imdata.shape[1]
        if not (0 <= col < w and 0 <= row < h):
            return

        row_i, col_i = int(round(row)), int(round(col))
        self._pending_picks.append((row_i, col_i))

        cross = pg.ScatterPlotItem(
            x=[float(col_i)], y=[float(row_i)],
            symbol='+', size=16, pen=pg.mkPen('#4488ff', width=2), brush=None)
        self._vb.addItem(cross)
        self._pending_markers.append(cross)
        self._lbl_pick.setText('Point %d / 3' % len(self._pending_picks))

        if len(self._pending_picks) < 3:
            return

        pts = self._pending_picks.copy()
        for m in self._pending_markers:
            self._vb.removeItem(m)
        self._pending_picks.clear()
        self._pending_markers.clear()

        if self._geo_mode:
            self._run_geo_search(pts)
        else:
            self._run_nearest_ref(pts)

    def _add_arc_to_list(self, hkl_6d, arc_item):
        """Add arc to the selected-reflections list (ignores duplicates)."""
        vec_str = '[%s]' % ' '.join('%d' % v for v in hkl_6d)
        # Check for duplicate by text
        for i in range(self._arc_list.count()):
            if self._arc_list.item(i).text() == vec_str:
                return
        list_item = QtWidgets.QListWidgetItem(vec_str)
        list_item.setFlags(list_item.flags() | QtCore.Qt.ItemIsUserCheckable)
        list_item.setCheckState(QtCore.Qt.Checked)
        list_item.setData(QtCore.Qt.UserRole, arc_item)
        colour = getattr(arc_item, '_colour', None)
        if colour is not None:
            list_item.setForeground(QtGui.QBrush(colour))
        self._arc_list.blockSignals(True)
        self._arc_list.addItem(list_item)
        self._arc_list.blockSignals(False)
        self._arc_to_list_item[id(arc_item)] = list_item
        if not getattr(self, '_bulk_select', False):
            self._on_selection_changed()

    def _remove_arc_from_list(self, arc_item):
        """Remove arc from list and from the scene."""
        list_item = self._arc_to_list_item.pop(id(arc_item), None)
        if list_item is not None:
            row = self._arc_list.row(list_item)
            if row >= 0:
                self._arc_list.takeItem(row)
        if arc_item in self._pick_items:
            self._vb.removeItem(arc_item)
            self._pick_items.remove(arc_item)
        self._arc_to_6d.pop(id(arc_item), None)
        if not getattr(self, '_bulk_select', False):
            self._on_selection_changed()

    def _on_list_item_changed(self, list_item):
        """Checkbox toggle → rebuild the selected-reflection overlay."""
        arc_item = list_item.data(QtCore.Qt.UserRole)
        if arc_item is not None:
            arc_item.setVisible(list_item.checkState() == QtCore.Qt.Checked)
        if not getattr(self, '_bulk_select', False):
            self._on_selection_changed()

    def _on_list_context_menu(self, pos):
        list_item = self._arc_list.itemAt(pos)
        if list_item is None:
            return
        menu = QtWidgets.QMenu(self)
        remove_action = menu.addAction('Remove')
        action = menu.exec_(self._arc_list.mapToGlobal(pos))
        if action == remove_action:
            arc_item = list_item.data(QtCore.Qt.UserRole)
            if arc_item is not None:
                self._remove_arc_from_list(arc_item)

    def _nearest_arc_at(self, scene_pos, threshold=8.0):
        """Return the arc ScatterPlotItem closest to scene_pos (within threshold view pixels)."""
        vb_pos = self._vb.mapSceneToView(scene_pos)
        col, row = vb_pos.x(), vb_pos.y()
        best_arc, best_dist = None, threshold
        for arc_item in list(self._pick_items):
            if id(arc_item) not in self._arc_to_6d:
                continue
            if not hasattr(arc_item, '_x_data'):
                continue
            d = float(np.sqrt((arc_item._x_data - col)**2 +
                               (arc_item._y_data - row)**2).min())
            if d < best_dist:
                best_dist, best_arc = d, arc_item
        return best_arc

    def _on_clear_picks(self):
        for item in self._pick_items:
            self._vb.removeItem(item)
        self._pick_items.clear()
        self._arc_to_6d.clear()
        self._arc_to_list_item.clear()
        self._arc_list.clear()
        for m in self._pending_markers:
            self._vb.removeItem(m)
        self._pending_markers.clear()
        self._pending_picks.clear()
        self._lbl_pick.setText('')
        self._sel_dms = None
        self._sel_order = []
        if not getattr(self, '_bulk_select', False):
            self._do_update()

    # ── Physics helpers ────────────────────────────────────────────────────────

    def _pixel_to_direction(self, row, col):
        a       = self.ig[0]
        thb_cur = ts.bragg([a, a, a, 90, 90, 90], self._hkl, self.ig[14]).th()[0]
        irmat   = np.array(
            ts.rotxyz([1, 0, 0], self.ig[11] + thb_cur).rmat() *
            ts.rotxyz([0, 1, 0], self.ig[12]).rmat() *
            ts.rotxyz([0, 0, 1], self.ig[13]).rmat()
        )
        pxvec    = np.array([row - self._dms.px, 0.0, self._dms.py - col])
        prepxvec = pxvec @ np.linalg.inv(irmat)
        centralv = -np.array(ts.psith2v(0.0, float(thb_cur))).flatten() * self.ig[10]
        diff     = prepxvec - centralv
        n        = np.linalg.norm(diff)
        if n < 1e-10:
            return None
        return diff / n

    def _ewald_scores(self, dirs):
        a  = self.ig[0]
        ko = self.ig[14] / 12.398
        bm = np.array(ts.bmatrix([a, a, a, 90, 90, 90]).bm())
        hkl002 = ts.PhasonDistoArray(
            np.array(self.full_reflist),
            np.array(self.full_reflist2),
            list(self.ig[15:24])
        ).qe1()
        hkl002_cart = np.array(hkl002) @ bm.T
        N = hkl002_cart.shape[0]

        G_primary  = np.array(self._hkl).flatten()  @ bm.T
        azir_cart  = np.array(self._dms.azir).flatten() @ bm.T
        sample_psi = self._dms.psi

        z_cart  = np.array([0., 0., 1.])
        rotvect = np.cross(z_cart, G_primary)
        rv_norm = np.linalg.norm(rotvect)
        zref    = z_cart @ bm.T
        cos_a   = np.clip(
            np.dot(G_primary, zref) /
            (np.linalg.norm(G_primary) * np.linalg.norm(zref)), -1., 1.)
        align_rad = np.arccos(cos_a)
        if rv_norm < 1e-6:
            R = np.eye(3)
        else:
            u = rotvect / rv_norm
            c, s = np.cos(align_rad), np.sin(align_rad)
            R = np.array([
                [c+u[0]*u[0]*(1-c),       u[0]*u[1]*(1-c)-u[2]*s,  u[0]*u[2]*(1-c)+u[1]*s],
                [u[1]*u[0]*(1-c)+u[2]*s,  c+u[1]*u[1]*(1-c),       u[1]*u[2]*(1-c)-u[0]*s],
                [u[2]*u[0]*(1-c)-u[1]*s,  u[2]*u[1]*(1-c)+u[0]*s,  c+u[2]*u[2]*(1-c)     ],
            ])

        g_z       = hkl002_cart @ R
        azir_z    = azir_cart   @ R
        azirangle = np.degrees(np.arctan2(azir_z[0], azir_z[1]))
        rhk       = np.sqrt(g_z[:, 0]**2 + g_z[:, 1]**2)
        rhkangle  = np.degrees(np.arctan2(g_z[:, 0], g_z[:, 1]))

        scores = np.zeros(N)
        for d in dirs:
            brag1   = np.degrees(np.arcsin(np.clip(d[2], -1., 1.)))
            psi_abs = np.degrees(np.arctan2(-d[0], d[1]))
            psi_req = sample_psi - psi_abs - self.ig[6]
            orighk  = ko * np.cos(np.radians(brag1))
            raw_sin = (ko * np.sin(np.radians(-brag1)) + g_z[:, 2]) / ko
            valid   = np.abs(raw_sin) <= 1.
            rewl    = ko * np.cos(np.arcsin(np.clip(raw_sin, -1., 1.)))
            numer   = orighk**2 - rhk**2 + rewl**2
            half_n  = numer / (2. * orighk)
            disc    = rewl**2 - half_n**2
            valid  &= disc >= 0.
            xint    = np.sqrt(np.maximum(disc, 0.))
            ia1     = np.degrees(np.arctan2( xint, half_n - orighk))
            ia2     = np.degrees(np.arctan2(-xint, half_n - orighk))
            psi1    = (ia1 + azirangle - rhkangle + 180.) % 360. - 180.
            psi2    = (ia2 + azirangle - rhkangle + 180.) % 360. - 180.
            diff1   = np.abs(((psi1 - psi_req + 180.) % 360.) - 180.)
            diff2   = np.abs(((psi2 - psi_req + 180.) % 360.) - 180.)
            scores += np.where(valid, np.minimum(diff1, diff2), 1e6)
        scores /= max(len(dirs), 1)
        return scores

    def _add_red_crosses(self, pts):
        for r, c in pts:
            cross = pg.ScatterPlotItem(
                x=[float(c)], y=[float(r)],
                symbol='+', size=16, pen=pg.mkPen('#ff4444', width=2), brush=None)
            self._vb.addItem(cross)
            self._pick_items.append(cross)

    def _plot_arc(self, hkl_6d, colour, draw=True):
        """Create an arc ScatterPlotItem for a single reflection.  With draw=True
        it is traced immediately (a one-reflection imcalc) — used for candidate
        previews.  With draw=False the item is created empty and left for the
        vectorised selected-engine pass to populate (fast bulk add / load)."""
        x_arr = y_arr = np.array([])
        if draw:
            rl1, rl2 = build_reflist_from_6d(hkl_6d.reshape(1, -1))
            self._prep_arc_engine()
            self._dms_full.reflist  = np.matrix(rl1)
            self._dms_full.reflist2 = np.matrix(rl2)
            try:
                self._dms_full.imcalc(extract_reduced(self.ig))
                x_arr, y_arr = self._arc_xy()
                if x_arr.size == 0:
                    return
            except Exception as e:
                print('Arc error [%s]: %s' % (' '.join('%d' % v for v in hkl_6d), e))
                return
        arc = pg.ScatterPlotItem(
            x=x_arr, y=y_arr, size=3, pen=None, brush=pg.mkBrush(colour))
        arc._x_data = x_arr   # cached for hit-testing
        arc._y_data = y_arr
        arc._colour = pg.mkColor(colour)
        self._vb.addItem(arc)
        self._pick_items.append(arc)
        self._arc_to_6d[id(arc)] = hkl_6d.copy()

    def _run_geo_search(self, pts):
        self._sync_ig()
        self._status.setText('Searching (geo)...')
        QtWidgets.QApplication.processEvents()
        dirs = [self._pixel_to_direction(r, c) for r, c in pts]
        dirs = [d for d in dirs if d is not None]
        if not dirs:
            self._lbl_pick.setText('No valid directions')
            self._status.setText('Ready')
            return

        scores   = self._ewald_scores(dirs)
        mask     = scores < self._psi_tol
        cand_idx = np.where(mask)[0]
        print('Geo search: %d/%d pass (psi_tol=%.1f°)' % (
            len(cand_idx), len(scores), self._psi_tol))

        if len(cand_idx) == 0:
            self._lbl_pick.setText('No match found')
            self._status.setText('Ready')
            return

        order  = np.argsort(scores[cand_idx])
        cands  = self.full_reflist_6d[cand_idx[order]]
        s_vals = scores[cand_idx[order]]

        self._add_red_crosses(pts)
        palette = [pg.intColor(i, hues=10) for i in range(10)]
        for k, (hkl_6d, score) in enumerate(zip(cands[:10], s_vals[:10])):
            print('  [%s]  psi_err=%.2f°' % (
                ' '.join('%d' % v for v in hkl_6d), score))
            self._plot_arc(hkl_6d, palette[k % 10])

        best_str = '[%s]' % ' '.join('%d' % v for v in cands[0])
        self._lbl_pick.setText('%s  +%d more' % (best_str, max(0, len(cands) - 1)))
        self._status.setText('Ready')

    def _run_nearest_ref(self, pts):
        self._sync_ig()
        self._status.setText('Searching (nearest-ref)...')
        QtWidgets.QApplication.processEvents()
        dirs = [self._pixel_to_direction(r, c) for r, c in pts]
        dirs = [d for d in dirs if d is not None]
        if not dirs:
            self._lbl_pick.setText('')
            self._status.setText('Ready')
            return

        scores  = self._ewald_scores(dirs)
        order   = np.argsort(scores)
        print('Nearest-ref top-5: %s' % ', '.join(
            '[%s]=%.2f' % (' '.join('%d' % v for v in self.full_reflist_6d[i]), scores[i])
            for i in order[:5]))

        best_idx = int(order[0])
        if scores[best_idx] > 10.0:
            print('No match (best=%.2f°)' % scores[best_idx])
            self._lbl_pick.setText('No match found')
            self._status.setText('Ready')
            return

        hkl_6d  = self.full_reflist_6d[best_idx].copy()
        vec_str = ' '.join('%d' % v for v in hkl_6d)
        print('Nearest-ref: [%s]  psi_err=%.2f°' % (vec_str, scores[best_idx]))
        self._lbl_pick.setText('[%s]  %.2f°' % (vec_str, scores[best_idx]))

        self._add_red_crosses(pts)
        self._plot_arc(hkl_6d, pg.mkColor('#00cccc'))
        self._status.setText('Ready')

    # ── Reset / Print ──────────────────────────────────────────────────────────

    def _on_reset(self):
        ig_reset  = self._initial_guess.copy()
        hkl_reset = self._hkl_ref.copy()
        self._suppress = True
        for label, idx, *_ in slider_defs:
            fs = self._sliders[label]
            if idx == 'h':
                fs.setValue(hkl_reset[0])
            elif idx == 'k':
                fs.setValue(hkl_reset[1])
            elif idx == 'l':
                fs.setValue(hkl_reset[2])
            else:
                fs.setValue(ig_reset[idx])
        self._suppress = False
        self.ig[:] = ig_reset
        self._hkl[:] = hkl_reset
        self._do_update()
        self._status.setText('Reset to initial guess')

    def _on_print(self):
        self._sync_ig()
        print('\n' + '=' * 72)
        print('hkl = %s' % self._hkl)
        print('initial_guess = np.array([%s])' %
              ', '.join('%.7f' % v for v in self.ig))
        print('=' * 72)

    def _on_cfg_table_changed(self, new_cfg):
        """Apply live edits from the Config table.  Geometry edits (psi / px /
        py) take effect immediately; metadata/scan edits are stored and applied
        on the next scan Load."""
        self._cfg = new_cfg
        geo = new_cfg.get('geometry', {})
        if 'psi' in geo:
            self._psi = float(geo['psi'])
            self._psirange = [self._psi - 180, self._psi + 180]
        if 'px_unscaled' in geo:
            self._px = float(geo['px_unscaled']) * zoomval
        if 'py_unscaled' in geo:
            self._py = float(geo['py_unscaled']) * zoomval
        self._rebuild_dms_slice()
        self._rebuild_selected_engine()   # psi/px/py are baked into the engine
        self._do_update()
        self._status.setText('Config updated')

    # ── Session capture / restore ────────────────────────────────────────────────

    def _collect_reflections(self):
        """Return (ref_6d, ref_6d_checked) for the currently selected
        reflections, in list order."""
        ref_6d, ref_6d_checked = [], []
        for i in range(self._arc_list.count()):
            item = self._arc_list.item(i)
            arc_item = item.data(QtCore.Qt.UserRole)
            hkl_6d = self._arc_to_6d.get(id(arc_item)) if arc_item is not None else None
            if hkl_6d is not None:
                ref_6d.append([int(v) for v in hkl_6d])
                ref_6d_checked.append(item.checkState() == QtCore.Qt.Checked)
        return ref_6d, ref_6d_checked

    def _apply_reflections(self, ref_6d_list, checked_list=None):
        """Plot the given 6D reflections and add them to the selection list,
        honouring their checked state.  Assumes the list has been cleared."""
        if checked_list is None:
            checked_list = [True] * len(ref_6d_list)
        # Bulk add without per-item rebuilds; one vectorised pass at the end.
        self._bulk_select = True
        if ref_6d_list:
            palette = [pg.intColor(i, hues=10) for i in range(10)]
            for k, (hkl_6d_raw, checked) in enumerate(zip(ref_6d_list, checked_list)):
                hkl_6d   = np.array(hkl_6d_raw, dtype=int)
                n_before = len(self._pick_items)
                self._plot_arc(hkl_6d, palette[k % 10], draw=False)
                if len(self._pick_items) <= n_before:
                    continue
                arc_item = self._pick_items[-1]
                self._add_arc_to_list(hkl_6d, arc_item)
                if not checked:
                    list_item = self._arc_to_list_item.get(id(arc_item))
                    if list_item is not None:
                        self._arc_list.blockSignals(True)
                        list_item.setCheckState(QtCore.Qt.Unchecked)
                        self._arc_list.blockSignals(False)
                        arc_item.setVisible(False)
        self._bulk_select = False
        self._on_selection_changed()

    def _session_dict(self):
        """Capture the full workflow state so a session can be resumed: the
        loaded scan, refined geometry, selected reflections, manual ROI-centre
        overrides and the last fit result."""
        self._sync_ig()
        ref_6d, ref_6d_checked = self._collect_reflections()

        # Manual ROI-centre overrides (only meaningful once curves are built)
        manual_centres = {}
        if self._centres is not None:
            for ridx in sorted(self._centre_override_rois):
                if ridx < self._centres.shape[0]:
                    manual_centres[str(int(ridx))] = float(self._centres[ridx, 0])
        # Carry forward any overrides restored but not yet re-applied to a build
        for ridx, xval in self._pending_centre_overrides.items():
            manual_centres.setdefault(str(int(ridx)), float(xval))

        fit_result = None
        if self._last_res_x is not None:
            fit_result = {'res_x': [float(v) for v in self._last_res_x]}
            if self._last_fit_info:
                fit_result.update(self._last_fit_info)

        return {
            'version':        2,
            'scan': {
                'scanpath':   self._scanpath,
                'scannum':    int(self._scannum),
                'datapoint':  int(self._datapoint),
                'datapoint0': int(self._datapoint0),
            },
            # top-level scannum/datapoint kept for backward compatibility
            'scannum':        int(self._scannum),
            'datapoint':      int(self._datapoint),
            'hkl':            self._hkl.tolist(),
            'initial_guess':  self.ig.tolist(),
            'ref_6d':         ref_6d,
            'ref_6d_checked': ref_6d_checked,
            'manual_centres': manual_centres,
            'fit_result':     fit_result,
        }

    def _write_session(self, path, data):
        text = json.dumps(data, indent=2)
        # Collapse inner integer arrays (ref_6d rows) onto a single line
        text = re.sub(
            r'\[\n\s+((?:-?\d+,\n\s+)*-?\d+)\n\s+\]',
            lambda m: '[' + ', '.join(
                x.strip() for x in re.split(r',\n\s*', m.group(1))) + ']',
            text)
        with open(path, 'w') as fh:
            fh.write(text + '\n')

    def _on_save_json(self):
        data = self._session_dict()
        default_path = os.path.join(
            os.getcwd(),
            'slider_state_%d_dp%d.json' % (self._scannum, self._datapoint))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save state as JSON', default_path, 'JSON files (*.json)')
        if not path:
            return
        self._write_session(path, data)
        self._status.setText('Saved → %s' % os.path.basename(path))

    def _on_save_reflections(self):
        """Save just the selected reflections (and their checked state) to a
        reusable JSON file, in the same format as the shipped reflection lists."""
        self._sync_ig()
        ref_6d, ref_6d_checked = self._collect_reflections()
        if not ref_6d:
            self._status.setText('No reflections selected to save')
            return
        data = {
            'scannum':        int(self._scannum),
            'datapoint':      int(self._datapoint),
            'hkl':            self._hkl.tolist(),
            'initial_guess':  self.ig.tolist(),
            'ref_6d':         ref_6d,
            'ref_6d_checked': ref_6d_checked,
        }
        default_path = os.path.join(
            os.getcwd(),
            'reflections_%d_dp%d.json' % (self._scannum, self._datapoint))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save reflections as JSON', default_path, 'JSON files (*.json)')
        if not path:
            return
        self._write_session(path, data)
        self._status.setText('Saved %d reflections → %s' % (
            len(ref_6d), os.path.basename(path)))

    def _on_load_reflections(self):
        """Load a reflection list into the selection, leaving the loaded scan
        and geometry untouched.  Accepts any file with a 'ref_6d' field
        (reflection lists and full sessions alike)."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load reflections from JSON', os.getcwd(), 'JSON files (*.json)')
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as e:
            self._status.setText('Load failed: %s' % e)
            return
        ref_6d_list = data.get('ref_6d')
        if not ref_6d_list:
            self._status.setText('No reflections found in %s' % os.path.basename(path))
            return
        self._on_clear_picks()
        self._apply_reflections(ref_6d_list, data.get('ref_6d_checked'))
        self._status.setText('Loaded %d reflections ← %s' % (
            len(ref_6d_list), os.path.basename(path)))

    def _on_load_json(self):
        default_dir = os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load state from JSON', default_dir, 'JSON files (*.json)')
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as e:
            self._status.setText('Load failed: %s' % e)
            return
        self._restore_from_dict(data)
        self._status.setText('Loaded → %s' % os.path.basename(path))

    def _restore_from_dict(self, data):
        """Restore the full workflow state captured by _session_dict()."""
        # 1. Reload the scan/image, if the session records one.  Do this first so
        #    the saved geometry below overwrites the scan-derived defaults.
        scan = data.get('scan')
        if scan and scan.get('scanpath') and scan.get('scannum') is not None:
            full = '%s%s.dat' % (scan['scanpath'], scan['scannum'])
            dp   = int(scan['datapoint'])
            dp0  = int(scan.get('datapoint0', scan['datapoint']))
            try:
                self._do_load_scan(full, dp, dp0)
                self._pending_scan_path = full
                self._lbl_scan_path.setText(os.path.basename(full))
                try:
                    n = dat2config.scan_length(full)
                    self._sb_dp0.setRange(0, max(0, n - 1))
                    self._sb_dp.setRange(0, max(0, n - 1))
                except Exception:
                    pass
                self._sb_dp0.setValue(dp0)
                self._sb_dp.setValue(dp)
            except Exception as e:
                self._status.setText('Scan reload failed: %s' % str(e)[:60])

        # 2. Restore sliders / ig / hkl.  The slider is now 24-element (psi/h/k/l);
        #    migrate a legacy 23-element state (psi/theta/chi, no kcor) by inserting
        #    kcor=0 at index 8 and zeroing the old theta/chi values (no equivalent
        #    in the index model).
        ig_loaded  = np.array(data['initial_guess'], dtype=float)
        if ig_loaded.size == 23:
            ig_loaded = np.insert(ig_loaded, 8, 0.0)   # insert kcor
            ig_loaded[7] = 0.0                           # old thetacorrection → hcor=0
            ig_loaded[9] = 0.0                           # old chicorrection   → lcor=0
        hkl_loaded = np.array(data['hkl'], dtype=float)
        self._suppress = True
        for label, idx, *_ in slider_defs:
            fs = self._sliders[label]
            if idx == 'h':
                fs.setValue(hkl_loaded[0])
            elif idx == 'k':
                fs.setValue(hkl_loaded[1])
            elif idx == 'l':
                fs.setValue(hkl_loaded[2])
            else:
                fs.setValue(ig_loaded[idx])
        self._suppress = False
        self.ig[:]   = ig_loaded
        self._hkl[:] = hkl_loaded

        # 3. Clear existing arcs / picks, then re-plot the saved reflections
        self._on_clear_picks()
        self._apply_reflections(data.get('ref_6d', []),
                                data.get('ref_6d_checked'))

        # 5. Stash manual centre overrides and fit result.  Centre overrides are
        #    applied the next time "Build curves" rebuilds the ROI centres.
        self._pending_centre_overrides = {
            int(k): float(v) for k, v in data.get('manual_centres', {}).items()}
        fit_result = data.get('fit_result')
        if fit_result and fit_result.get('res_x') is not None:
            self._last_res_x = np.array(fit_result['res_x'], dtype=float)
            self._last_fit_info = {k: fit_result[k]
                                   for k in ('opt', 'elapsed', 'method')
                                   if k in fit_result}
        else:
            self._last_res_x = None
            self._last_fit_info = None

    def _maybe_restore_session(self):
        """On launch, offer to restore the auto-saved previous session."""
        if not os.path.exists(SESSION_FILE):
            return
        try:
            with open(SESSION_FILE) as fh:
                data = json.load(fh)
        except Exception:
            return
        scan = data.get('scan', {})
        descr = 'scan %s  dp %s' % (scan.get('scannum', '?'),
                                    scan.get('datapoint', '?'))
        reply = QtWidgets.QMessageBox.question(
            self, 'Restore previous session',
            'Resume your last session (%s)?' % descr,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes)
        if reply == QtWidgets.QMessageBox.Yes:
            self._restore_from_dict(data)
            self._status.setText('Restored previous session (%s)' % descr)

    def _on_clear_session(self):
        """Reset the entire workflow to a clean slate."""
        reply = QtWidgets.QMessageBox.question(
            self, 'Clear workflow',
            'Clear the whole workflow (geometry, selected reflections, built '
            'curves, centre overrides and fit result)?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        # Drop selected reflections and built ROI/fit state
        self._on_clear_picks()
        self._fit_dms      = None
        self._kernel       = None
        self._centres      = None
        self._linedatax    = None
        self._linedatay    = None
        self._imcoeffs     = None
        self._reflist_fit  = None
        self._reflist2_fit = None
        self._ref_6d_fit   = None
        self._centre_override_rois = set()
        self._pending_centre_overrides = {}
        self._last_res_x   = None
        self._last_fit_info = None
        self._last_fit_output = None
        self._btn_save_fit.setEnabled(False)
        self._init_line_plot()
        # Reset geometry sliders to the initial guess for the current scan
        self._on_reset()
        self._status.setText('Workflow cleared')

    # ── Scan loading ───────────────────────────────────────────────────────────

    def _on_browse_scan(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open scan file', self._scanpath,
            'DAT files (*.dat);;All files (*)')
        if not path:
            return
        self._pending_scan_path = path
        self._lbl_scan_path.setText(os.path.basename(path))
        # Probe the file to count datapoints and update spinbox ranges
        try:
            n = dat2config.scan_length(path)
            self._sb_dp0.setRange(0, max(0, n - 1))
            self._sb_dp.setRange(0, max(0, n - 1))
            self._lbl_scan_info.setText('N=%d scan points' % n)
        except Exception as e:
            self._lbl_scan_info.setText('Read error: %s' % str(e)[:50])

    def _on_load_scan(self):
        path = self._pending_scan_path
        dp   = self._sb_dp.value()
        dp0  = self._sb_dp0.value()
        self._status.setText('Loading…')
        QtWidgets.QApplication.processEvents()
        try:
            self._do_load_scan(path, dp, dp0)
        except Exception as e:
            self._status.setText('Load failed: %s' % str(e)[:70])
            import traceback; traceback.print_exc()

    def _do_load_scan(self, path, dp, dp0):
        # The converter is the only sanctioned .dat reader.
        exp = dat2config.extract_metadata(path, dp, dp0)
        lat = list(exp['lattice'])

        hkl_ref    = np.array([2.27931876, 3.70249186, 1.29579814])
        hkl_ref    = hkl_ref * exp['energy'] / exp['energy0']
        hklint_ref = np.round(hkl_ref)

        en_new    = float(exp['energy'])
        azir_new  = list(exp['azir'])
        scan_dir  = os.path.dirname(os.path.abspath(path)) + os.sep
        basename  = os.path.basename(path)
        m         = re.match(r'^(\d+)\.dat$', basename)
        snum_new  = int(m.group(1)) if m else self._scannum
        imtmpl    = exp['image_template']
        imnum_new = dp + 1

        try:
            im_new = imageio.imread(scan_dir + imtmpl % imnum_new)
            im_new = ndimage.zoom(im_new, zoomval, order=3)
        except Exception as e:
            raise RuntimeError('Cannot load image %s: %s' % (imtmpl % imnum_new, e))

        imdata_new = np.copy(im_new)

        # Update energy slider range to centre on the new scan energy, but
        # preserve the user's current slider value if it falls within the new
        # range.  Do NOT call setValue here – slider-controlled parameters
        # (energy, h, k, l, psi) must stay exactly as the user left them so
        # that reloading the same file/dp produces no shift.
        self._suppress = True
        self._sliders['energy'].setRange(en_new - 0.5, en_new + 0.5)
        self._suppress = False

        # Read the current slider state into self.ig / self._hkl so the DMS
        # objects are built with the exact same parameters that were in use
        # before the load (guarantees same file/dp → no shift).
        self._sync_ig()

        # Derive geometry from current slider state, not raw scan values
        cur_energy   = self.ig[14]
        cur_hkl      = self._hkl.copy()
        thb_cur      = ts.bragg(lat, cur_hkl, cur_energy).th()[0]
        thrange_cur  = [thb_cur - 27, thb_cur + 10]
        psirange_cur = [self._psi - 180, self._psi + 180]
        hkllist_cur  = ts.pilkhlrange(
            lat, cur_hkl, cur_energy, thrange_cur[0], thrange_cur[1]
        ).hklscan(numsteps)

        # Commit scan-level state (image, lattice, azir, reference hkl)
        self._lattice    = lat
        self._azir       = azir_new
        self._imdata     = imdata_new
        self._hkl_ref    = hkl_ref.copy()
        self._hklint     = hklint_ref.copy()
        self._thrange    = thrange_cur
        self._psirange   = psirange_cur
        self._hkllist    = hkllist_cur
        self._scanpath   = scan_dir
        self._scannum    = snum_new
        self._datapoint  = dp
        self._datapoint0 = dp0
        self._imtemplate = imtmpl
        self._initial_guess     = self.ig.copy()
        self._initial_guess[14] = en_new
        self._en_scan           = en_new
        self._last_hkl[:] = np.inf

        # Sync worker
        self._worker.lattice = self._lattice
        self._worker.thrange = self._thrange

        # Rebuild DMS objects using current slider state (psi/hkl/energy unchanged)
        ig0 = self.ig.copy()
        self._dms = make_overlay_dms(
            self.full_reflist, self.full_reflist2, self._hkl, self._imdata,
            self._psirange, self._thrange, self._azir, self._psi,
            self._px, self._py, ig0)
        self._dms_full = make_overlay_dms(
            self.full_reflist, self.full_reflist2, self._hkl, self._imdata,
            self._psirange, self._thrange, self._azir, self._psi,
            self._px, self._py, ig0)

        # Update image display
        self._img_item.setImage(imdata_new, autoLevels=False)
        self._img_item.setLevels(colourlim)

        # Refresh the live config + table with the newly imported metadata
        self._cfg.setdefault('scan', {}).update({
            'scannum': snum_new, 'scanpath': scan_dir,
            'datapoint': dp, 'datapoint0': dp0,
        })
        self._cfg['experiment'] = {
            'lattice':        list(lat),
            'energy':         float(en_new),
            'energy0':        float(exp['energy0']),
            'azir':           list(azir_new),
            'image_template': imtmpl,
        }
        self._cfgtable.set_config(self._cfg)

        # Update UI labels
        self._lbl_scan_path.setText(basename)
        self._lbl_scan_info.setText('E=%.4f keV  dp=%d' % (en_new, dp))
        self.setWindowTitle('DMS Slider v3 — scan %d  dp=%d  E=%.4f keV' %
                            (snum_new, dp, en_new))
        self._rebuild_selected_engine()   # new image/psi baked into the engine
        self._do_update()
        self._status.setText('Loaded scan %d dp=%d  E=%.4f keV' % (snum_new, dp, en_new))

    # ── Workflow export / launch ───────────────────────────────────────────────

    def _workflow_ig24(self):
        """The slider and workflow now share the 24-element layout and the same
        engine, so export is the slider ig with two unit conversions only:
        detector distance → full/un-zoomed px, and energy → offset from the raw
        scan energy (workflow adds the scan energy back on load)."""
        ig24 = self.ig.copy()
        ig24[10] = self.ig[10] * 2.0 / zoomval    # detdist → full, un-zoomed px
        ig24[14] = self.ig[14] - self._en_scan    # energy → offset from scan energy
        return ig24

    def _build_workflow_config(self):
        """Return a workflow-compatible config dict populated from the current
        slider state.  The template JSON (if set) supplies all the fixed
        experiment parameters; the scan, experiment, geometry, and crystal
        sections are overridden with live slider values."""
        self._sync_ig()

        # Load template
        if self._workflow_template and os.path.exists(self._workflow_template):
            with open(self._workflow_template) as fh:
                cfg = json.load(fh)
        else:
            cfg = {
                'flags': {
                    'save': 0, 'fit': 0, 'firstplot': 0,
                    'detoptimize': 1, 'energyopt': 0, 'autoreflist': 0,
                    'show_centres': 1, 'show_numbers': 1, 'axis_off': 0,
                },
                'display': {
                    'zoomval': zoomval, 'colourlim': list(colourlim),
                    'colmap': colmap, 'subcellsx': 7, 'subcellsy': 4,
                },
                'roi': {'width_per_zoom': 45, 'comwidth_per_zoom': 5},
                'geometry': {'scatv': scatv},
                'computation': {
                    'numsteps': numsteps,
                    'simsigma_per_zoom': simsigma / max(zoomval, 1),
                    'thrange_delta': [-27, 10],
                    'bravais': 'icosahedral',
                    'opt_method': 'COBYLA',
                    'tolerance': 1e-6,
                    'intensity': 1, 'threshold': 0, 'n_parallel_starts': 1,
                },
                'crystal': {
                    'lattice2': [float(self.ig[0])] * 3 + [90., 90., 90.],
                },
                'manual_centres': {},
                'paths': {'cif_file': ''},
            }

        # Collect checked reflections (matches what the fit uses); fall back to
        # all plotted arcs, then the manual list.
        ref_6d = self._checked_ref_6d().tolist()
        if not ref_6d:
            ref_6d = [[int(v) for v in h] for h in self._arc_to_6d.values()]
        if not ref_6d:
            ref_6d = ref_6d_manual.tolist()

        ig24 = self._workflow_ig24()

        # ── Override with live slider state ───────────────────────────────────
        # datapoint0 = datapoint → workflow energy-rescaling factor = 1.0,
        # so the exported hkl is used as-is.
        cfg['scan'] = {
            'scannum':    int(self._scannum),
            'scanpath':   self._scanpath,
            'datapoint':  int(self._datapoint),
            'datapoint0': int(self._datapoint),
        }
        # Decoupled metadata: workflow reads this instead of opening the .dat.
        # dp0 == dp ⇒ energy/energy0 ratio = 1, so hkl is used exactly as exported.
        cfg['experiment'] = {
            'lattice':        list(self._lattice),
            'energy':         float(self._en_scan),
            'energy0':        float(self._en_scan),
            'azir':           list(self._azir),
            'image_template': self._imtemplate,
        }
        cfg['geometry'].update({
            'hkl':         self._hkl.tolist(),
            'psi':         float(self._psi),
            'px_unscaled': float(self._px / zoomval),
            'py_unscaled': float(self._py / zoomval),
            'scatv':       scatv,
        })
        cfg['crystal']['ref_6d']             = ref_6d
        cfg['crystal']['initial_guess_base'] = ig24.tolist()
        cfg['crystal']['lattice2']           = [float(self.ig[0])] * 3 + [90., 90., 90.]
        cfg['crystal']['tau_approx']         = float(tau)   # pass rational approx to workflow
        cfg['display']['zoomval']            = zoomval
        cfg['display']['colourlim']          = list(colourlim)
        cfg['computation']['numsteps']       = numsteps
        cfg['computation']['simsigma_per_zoom'] = float(simsigma / max(zoomval, 1))
        cfg.setdefault('roi', {})['width_per_zoom'] = float(width / max(zoomval, 1))
        # Template manual_centres reference ROI indices from a different ref_6d;
        # always clear them so workflow.py doesn't crash with an IndexError.
        cfg['manual_centres'] = {}

        return cfg

    def _on_export_workflow_json(self):
        cfg = self._build_workflow_config()
        default = os.path.join(
            os.getcwd(), 'workflow_%d_dp%d.json' % (self._scannum, self._datapoint))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save workflow config', default, 'JSON files (*.json)')
        if not path:
            return None
        with open(path, 'w') as fh:
            json.dump(cfg, fh, indent=2)
        self._status.setText('Workflow config saved → %s' % os.path.basename(path))
        return path

    def _on_save_fit_processing(self):
        """Write a timestamped Processing/ snapshot of the last completed fit,
        mirroring the artifacts produced by the batch fit.py (save=1)."""
        out = self._last_fit_output
        if out is None:
            self._status.setText('Run a fit before saving to Processing')
            return

        import shutil
        from time import strftime
        try:
            scan    = int(self._scannum)
            imnum   = int(self._datapoint) + 1
            fittype = out['method']
            datestr = strftime('%Y%m%d%H%M')
            outpath = os.path.join(
                os.getcwd(), 'Processing',
                '%s_%d_%d_fivefold_2ROIS_AlPdMn_Not_Annealed_%s'
                % (datestr, imnum, scan, fittype)) + os.sep
            os.makedirs(outpath, exist_ok=True)

            # ── code + config snapshots ──────────────────────────────────────
            for fname in ('slider.py', 'ts_quasi.py'):
                src = os.path.join(PKGDIR, fname)
                if os.path.exists(src):
                    shutil.copy(src, outpath)
            cfg_snapshot = self._build_workflow_config()
            with open(os.path.join(outpath, 'config_%d.json' % scan), 'w') as fh:
                json.dump(cfg_snapshot, fh, indent=2)

            # ── DMS overlay image (IM_<scan>.png), built like fit.py ─────────
            im3 = np.copy(self._imdata).astype(float)
            holder = np.zeros((im3.shape[0], im3.shape[1], 3))
            imr = np.zeros((im3.shape[0], im3.shape[1]))
            dmsindex = out.get('dmsindex')
            if (dmsindex is not None and len(dmsindex) == 2
                    and len(np.asarray(dmsindex[0])) > 0):
                imr[dmsindex] = 255
            holder[:, :, 0] = imr
            holder[:, :, 1] = imr
            clip = colourlim[1]
            im3[im3 > clip] = clip
            mx = im3.max() or 1.0
            holder[:, :, 2] = (255. / mx) * im3
            imageio.imsave(os.path.join(outpath, 'IM_%05d.png' % scan),
                           holder.astype(np.uint8))

            # ── ROI integrated-curve grid (slider's analog of fit.py's plot) ─
            try:
                self._roi_grid.grab().save(
                    os.path.join(outpath, '_PLOT_%05d.png' % scan))
            except Exception:
                pass

            # ── result vectors ───────────────────────────────────────────────
            np.savetxt(os.path.join(outpath, 'res.x.txt'), out['res_x'])
            inputs = out['inputarray']
            with open(os.path.join(outpath, 'Result.txt'), 'w') as f:
                f.write('initial_guess = np.array([')
                for v in inputs:
                    f.write('%f,' % v)
                f.write('])\n')
                f.write('opt = %s\n' % out['opt'])
                f.write('method = %s\n' % fittype)

            self._status.setText('Fit saved → %s'
                                 % os.path.join('Processing', os.path.basename(
                                     os.path.normpath(outpath))))
            print('Fit results written to ' + outpath)
        except Exception as e:
            self._status.setText('Save failed: %s' % str(e)[:80])
            import traceback; traceback.print_exc()

    # ── Build integrated curves (on request, from the checked arcs) ──────────────

    def _checked_ref_6d(self):
        """6D indices of the arcs that are currently checked in the arc list."""
        out = []
        for i in range(self._arc_list.count()):
            item = self._arc_list.item(i)
            if item.checkState() != QtCore.Qt.Checked:
                continue
            arc_item = item.data(QtCore.Qt.UserRole)
            hkl_6d = self._arc_to_6d.get(id(arc_item)) if arc_item is not None else None
            if hkl_6d is not None:
                out.append([int(v) for v in hkl_6d])
        return np.array(out)

    def _on_build_curves(self):
        if self._fitting:
            self._status.setText('Stop the fit before rebuilding curves')
            return
        self._sync_ig()
        sel6d = self._checked_ref_6d()
        if sel6d.shape[0] == 0:
            self._status.setText('Check at least one reflection (click arcs first)')
            return
        self._status.setText('Building integrated curves...')
        QtWidgets.QApplication.processEvents()
        self._worker.idle.wait(timeout=5.0)

        try:
            ig = self.ig
            rl, rl2 = build_reflist_from_6d(sel6d)
            self._reflist_fit  = np.array(rl)
            self._reflist2_fit = np.array(rl2)
            self._ref_6d_fit   = sel6d

            hkllist_cur = ts.pilkhlrange(
                self._lattice, self._hkl, ig[14],
                self._thrange[0], self._thrange[1]).hklscan(numsteps)
            self._hkllistrange_fit = [self._thrange[0], self._thrange[1], numsteps]

            builderargs = (
                self._reflist_fit, hkllist_cur, hklint, intensity,
                self._psirange, threshold, self._hkl, detvects, self._imdata.shape,
                simsigma, self._azir, self._psi, self._px, self._py, scatv,
                ig[10], ig[11], ig[12], ig[13], ig[14],
                ig, self._reflist2_fit, list(ig[15:24])
            )
            self._kernel = ts.roibuilder_ico_hkl(builderargs)
            self._imcoeffs, self._linedatax, self._linedatay, _, _, _ = \
                ts.multiroifit2(self._imdata, self._kernel, width, 0.02, 10.0)
            self._centres = np.array([self._imcoeffs[:, 2]]).T
            self._centre_override_rois = set()
            # Re-apply manual centre overrides restored from a session file
            if self._pending_centre_overrides:
                for ridx, xval in self._pending_centre_overrides.items():
                    if 0 <= ridx < self._centres.shape[0]:
                        self._centres[ridx, 0] = xval
                        self._centre_override_rois.add(ridx)
                self._pending_centre_overrides = {}

            self._fit_dms = ts.dmsfit_ico_hkl(
                self._reflist_fit, list(self._hkllistrange_fit), hklint,
                self._psirange, width, self._centres, self._kernel,
                self._hkl, detvects, self._imdata, simsigma, self._azir,
                self._psi, self._px, self._py, scatv,
                bravais, bool(detoptimize), bool(energyopt),
                ig[10], ig[11], ig[12], ig[13], ig[14],
                self._reflist2_fit, list(ig[15:24]), ig[0])
            self._fit_dms.setCalLattice(ig[:6].tolist())
            self._fit_dms.setLattice(ig[:6].tolist())
            self._fit_dms.hkllistrange[2] = numsteps_interactive
            try:
                self._fit_dms.imcalc(extract_reduced(ig))
            except Exception:
                pass

            self._init_line_plot()
            self._status.setText('%d reflections, %d ROIs — ready to fit' % (
                sel6d.shape[0], self._kernel.shape[2]))
        except Exception as e:
            self._status.setText('Build failed: %s' % str(e)[:80])
            import traceback; traceback.print_exc()

    # ── ROI integrated-curve grid ────────────────────────────────────────────────

    def _init_line_plot(self):
        self._roi_grid.clear()
        self._exp_curves, self._sim_curves = [], []
        self._exp_centre_lines, self._sim_centre_lines = [], []
        self._roi_plots = []
        self._selected_roi = None
        if self._kernel is None:
            return
        n        = self._kernel.shape[2]
        ncols    = self._cfg.get('display', {}).get('subcellsy', 4)
        nref     = len(self._reflist_fit)
        show_num = self._cfg.get('flags', {}).get('show_numbers', 1)
        # Colour each reflection's sim curve to match its on-image DMS line/arc.
        # Checked arcs are in the same order as self._ref_6d_fit, so refnum indexes
        # both.  Fall back to the HSV ramp if an arc has no cached colour.
        sel_arcs, _ = self._selected_arcs()
        def _ref_colour(j):
            if j < len(sel_arcs) and getattr(sel_arcs[j], '_colour', None) is not None:
                return sel_arcs[j]._colour
            return pg.hsvColor(j / max(nref, 1), 0.85, 0.95, 0.85)
        refnum, roicount = 0, 0
        for i in range(n):
            r, c = divmod(i, ncols)
            pl = self._roi_grid.addPlot(row=r, col=c)
            pl.setMenuEnabled(False); pl.hideButtons()
            pl.hideAxis('left'); pl.hideAxis('bottom')
            pl.setDefaultPadding(0.05)
            if self._ref_6d_fit is not None and refnum < len(self._ref_6d_fit):
                lbl = ('%d: %s' % (i, list(self._ref_6d_fit[refnum])) if show_num
                       else str(list(self._ref_6d_fit[refnum])))
            else:
                lbl = str(i)
            pl.setTitle(lbl, size='7pt')
            cur_refnum = refnum
            if roicount == 1:
                refnum += 1; roicount = -1
            roicount += 1
            ref_col = _ref_colour(cur_refnum)
            self._exp_curves.append(pl.plot(pen=pg.mkPen('#4488ff', width=1)))
            self._sim_curves.append(pl.plot(pen=pg.mkPen(ref_col, width=1)))
            exp_cl = pg.InfiniteLine(angle=90, movable=False,
                pen=pg.mkPen('#4488ff', width=1, style=QtCore.Qt.DashLine))
            sim_cl = pg.InfiniteLine(angle=90, movable=False,
                pen=pg.mkPen(ref_col, width=1, style=QtCore.Qt.DashLine))
            pl.addItem(exp_cl); pl.addItem(sim_cl)
            self._exp_centre_lines.append(exp_cl)
            self._sim_centre_lines.append(sim_cl)
            self._roi_plots.append(pl)
        self._draw_exp_lines()
        self._try_draw_sim_lines()

    def _draw_exp_lines(self):
        for i, curve in enumerate(self._exp_curves):
            curve.setData(self._linedatax[i], self._linedatay[i])
        for i, cl in enumerate(self._exp_centre_lines):
            overridden = i in self._centre_override_rois
            if overridden and self._centres is not None and i < self._centres.shape[0]:
                cl.setValue(float(self._centres[i, 0]))
            elif not overridden and i < len(self._imcoeffs):
                cl.setValue(float(self._imcoeffs[i, 2]))
            cl.setPen(pg.mkPen('#ffaa00', width=1.5) if overridden
                      else pg.mkPen('#4488ff', width=1, style=QtCore.Qt.DashLine))

    def _draw_sim_lines(self, ldscoeffs, ldsx, ldsy):
        for i, curve in enumerate(self._sim_curves):
            if i >= len(ldsy):
                break
            y_exp = self._linedatay[i]; y_sim = ldsy[i]
            denom = y_sim.max() - y_sim.min()
            if abs(denom) < 1e-10:
                denom = 1.0
            yscale  = (y_exp.max() - y_exp.min()) / denom
            yoffset = y_exp.min() - (y_sim * yscale).min()
            curve.setData(ldsx[i], y_sim * yscale + yoffset)
        for i, cl in enumerate(self._sim_centre_lines):
            if i < len(ldscoeffs):
                cl.setValue(float(ldscoeffs[i, 2]))

    def _try_draw_sim_lines(self):
        if (self._fit_dms is not None and self._fit_dms.imsim is not None
                and self._sim_curves):
            try:
                coefs, ldsx, ldsy, _, _, _ = ts.multiroifit(
                    self._fit_dms.imsim, self._kernel, width, 10)
                self._draw_sim_lines(coefs, ldsx, ldsy)
            except Exception:
                pass

    def _on_roi_grid_clicked(self, event):
        if not self._roi_plots:
            return
        pos = event.scenePos()
        for i, pl in enumerate(self._roi_plots):
            if pl.vb.sceneBoundingRect().contains(pos):
                if event.button() == QtCore.Qt.RightButton:
                    pt = pl.vb.mapSceneToView(pos)
                    self._set_centre_override(i, pt.x())
                    event.accept(); return
                if self._selected_roi == i:
                    self._selected_roi = None
                    pl.vb.setBackgroundColor(None)
                    self._arc_list.clearSelection()
                else:
                    if self._selected_roi is not None:
                        self._roi_plots[self._selected_roi].vb.setBackgroundColor(None)
                    self._selected_roi = i
                    pl.vb.setBackgroundColor((60, 40, 0, 80))
                    self._select_refl_in_list(i)
                break

    def _select_refl_in_list(self, roi_idx):
        """Select, in the arc/reflection list, the reflection that ROI roi_idx
        belongs to (each reflection contributes two consecutive ROIs)."""
        if self._ref_6d_fit is None:
            return
        refidx = roi_idx // 2
        if refidx >= len(self._ref_6d_fit):
            return
        vec_str = '[%s]' % ' '.join('%d' % v for v in self._ref_6d_fit[refidx])
        for j in range(self._arc_list.count()):
            item = self._arc_list.item(j)
            if item.text() == vec_str:
                self._arc_list.setCurrentItem(item)
                self._arc_list.scrollToItem(item)
                break

    def _set_centre_override(self, roi_idx, x):
        self._centres[roi_idx, 0] = x
        self._fit_dms.centres[roi_idx, 0] = x
        self._centre_override_rois.add(roi_idx)
        if roi_idx < len(self._exp_centre_lines):
            self._exp_centre_lines[roi_idx].setValue(x)
            self._exp_centre_lines[roi_idx].setPen(pg.mkPen('#ffaa00', width=1.5))
        self._status.setText('Centre override ROI %d: x=%.1f' % (roi_idx, x))

    def _on_roi_mouse_moved(self, evt):
        pos = evt[0]
        for pl in self._roi_plots:
            if pl.vb.sceneBoundingRect().contains(pos):
                pt = pl.vb.mapSceneToView(pos)
                self._roi_coord_lbl.setText('x=%.1f  y=%.4g' % (pt.x(), pt.y()))
                return

    # ── Fit ──────────────────────────────────────────────────────────────────────

    def _on_algo(self, method):
        self._active_method = method

    def _on_numsteps_changed(self, value):
        """Update the point count (hkl scan resolution) used for the live image
        overlay and the fit.  The live engines bake the resolution into their
        hkllistrange, so push the new value in and redraw immediately."""
        global numsteps
        numsteps = int(value)
        for eng in (self._dms, self._dms_full, self._sel_dms, self._fit_dms):
            if eng is not None:
                eng.hkllistrange[2] = numsteps
        self._do_update()
        self._status.setText('Points = %d' % numsteps)

    def _on_width_changed(self, value):
        """Update the ROI integration width (pixels)."""
        global width
        width = int(value)
        if self._fit_dms is not None:
            self._status.setText('Width = %d px — rebuild curves to apply' % width)

    def _on_simsigma_changed(self, value):
        """Update the simulation Gaussian blur sigma.  The engine applies it each
        imcalc, so push the new value into the live engines and redraw."""
        global simsigma
        simsigma = float(value)
        for eng in (self._dms, self._dms_full, self._sel_dms, self._fit_dms):
            if eng is not None:
                eng.simsigma = simsigma
        self._do_update()
        self._status.setText('Sigma = %.2f' % simsigma)

    def _do_fit(self):
        if self._fitting:
            return
        if self._fit_dms is None:
            self._status.setText('Build curves before fitting')
            return
        self._sync_ig()
        ig = self.ig
        self._fitting = True
        self._status.setText('Fitting...')

        reduced = extract_reduced(ig)
        dms = self._fit_dms
        dms.hkllistrange[2] = numsteps
        dms.detdistancepx = ig[10]; dms.detxrot = ig[11]
        dms.detyrot = ig[12];       dms.detzrot = ig[13]
        dms.energy = ig[14];        dms.a = ig[0]
        dms.setLattice([ig[0], ig[0], ig[0], 90, 90, 90])

        bounds = list(zip(reduced - 1.5, reduced + 1.5))
        self._worker.idle.wait(timeout=5.0)

        self._fit_worker = FitWorker(
            dms, reduced, bounds, self._active_method, n_parallel_starts)
        self._fit_worker.done.connect(self._on_fit_done)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.stopped.connect(self._on_fit_stopped)
        self._btn_stop.setEnabled(True)
        self._fit_worker.start()

    def _on_fit_done(self, result):
        self._fitting = False
        self._btn_stop.setEnabled(False)
        inputarray = result['inputarray']
        self._suppress = True
        for label, idx, *_ in slider_defs:
            if isinstance(idx, int) and idx < len(inputarray):
                self._sliders[label].setValue(inputarray[idx])
        self.ig[:] = inputarray
        self._suppress = False
        # Keep the refined result so it can be captured in the session
        self._last_res_x = np.array(result.get('res_x', inputarray), dtype=float)
        self._last_fit_info = {'opt': float(result['opt']),
                               'elapsed': float(result['elapsed']),
                               'method': result['method']}
        # Keep the full fit output so it can be written to Processing/ on request
        self._last_fit_output = {
            'opt':        float(result['opt']),
            'method':     result['method'],
            'res_x':      np.array(result.get('res_x', inputarray), dtype=float),
            'inputarray': np.array(inputarray, dtype=float),
            'dmsindex':   result.get('dmsindex'),
        }
        self._btn_save_fit.setEnabled(True)
        self._status.setText('Fit complete.  χ²=%.4f  t=%.1fs  [%s]' % (
            result['opt'], result['elapsed'], result['method']))
        print('initial_guess = np.array([' +
              ','.join('%.6f' % v for v in inputarray) + '])')
        self._do_update()
        self._draw_exp_lines()
        self._try_draw_sim_lines()

    def _on_fit_error(self, msg, elapsed):
        self._fitting = False
        self._btn_stop.setEnabled(False)
        self._status.setText('Fit failed: %s' % msg[:60])

    def _on_stop_fit(self):
        if self._fit_worker and self._fit_worker.isRunning():
            self._btn_stop.setEnabled(False)
            self._status.setText('Stopping fit...')
            self._fit_worker.stop()

    def _on_fit_stopped(self, elapsed):
        self._fitting = False
        self._btn_stop.setEnabled(False)
        self._status.setText('Fit stopped after %.1fs' % elapsed)

    def closeEvent(self, event):
        # Auto-save the session so it can be offered for restore next launch.
        try:
            self._write_session(SESSION_FILE, self._session_dict())
        except Exception:
            pass
        self._worker.stop()
        if self._fit_worker and self._fit_worker.isRunning():
            self._fit_worker.wait()
        super().closeEvent(event)


# ── Launch ─────────────────────────────────────────────────────────────────────

app = QtWidgets.QApplication(sys.argv)
app.setStyle('Fusion')

_p     = QtGui.QPalette()
_dark  = QtGui.QColor(26,  26,  26)
_mid   = QtGui.QColor(42,  42,  42)
_light = QtGui.QColor(58,  58,  58)
_text  = QtGui.QColor(210, 210, 210)
_hilit = QtGui.QColor(42,  130, 218)
_p.setColor(QtGui.QPalette.Window,          _dark)
_p.setColor(QtGui.QPalette.WindowText,      _text)
_p.setColor(QtGui.QPalette.Base,            _mid)
_p.setColor(QtGui.QPalette.AlternateBase,   _light)
_p.setColor(QtGui.QPalette.Text,            _text)
_p.setColor(QtGui.QPalette.Button,          _light)
_p.setColor(QtGui.QPalette.ButtonText,      _text)
_p.setColor(QtGui.QPalette.ToolTipBase,     _mid)
_p.setColor(QtGui.QPalette.ToolTipText,     _text)
_p.setColor(QtGui.QPalette.Highlight,       _hilit)
_p.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
_p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text,       QtGui.QColor(100, 100, 100))
_p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(100, 100, 100))
app.setPalette(_p)

win = DMSSlider()
win.show()
sys.exit(app.exec_())
