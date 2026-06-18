#!/usr/bin/env python
"""
Unified DMS workflow: interactive slider refinement → automated fitting.

Usage:
    python workflow.py [config.json]

Opens an interactive slider window for manual refinement of the initial guess
overlaid on the experimental detector image. Click on an algorithm radio button
to launch the optimizer using the current slider values. Click "Print" to output
the current parameter vector. Click "Reset" to restore the initial guess.
"""

import os, sys, json, time, copy, threading
if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
    os.environ.setdefault('QT_QPA_PLATFORM', 'wayland')

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize, differential_evolution, basinhopping
from joblib import Parallel, delayed
import imageio.v2 as imageio
from time import strftime

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder='row-major', background='#1a1a1a', foreground='#cccccc')

PKGDIR  = os.path.abspath(os.path.dirname(__file__))
CONFIGS = os.path.join(PKGDIR, 'configs')
from . import ts_quasi as ts
from . import dat2config
from .config_table import ConfigTable

# ── Load config ───────────────────────────────────────────────────────────────

if len(sys.argv) > 1:
    cfg_path = os.path.abspath(sys.argv[1])
else:
    cfg_path = os.path.join(
        CONFIGS,
        'fit_fivefold_axis_AlPdMn_Not_Annealed_2M_2ROIS_internal_hkl.json'
    )

with open(cfg_path) as f:
    cfg = json.load(f)

print(f'Loaded config: {cfg_path}')


def resolve_experiment(cfg):
    """Return cfg['experiment'], extracting it from the .dat once (and caching it
    back into cfg) if the config predates the decoupled schema."""
    exp = cfg.get("experiment")
    if exp is None:
        s = cfg["scan"]
        dat_path = os.path.join(s["scanpath"], str(s["scannum"]) + '.dat')
        exp = dat2config.extract_metadata(dat_path, s["datapoint"], s["datapoint0"])
        cfg["experiment"] = exp
        print(f'[workflow] experiment section missing — extracted from {dat_path}')
    return exp

# ── Extract config values ─────────────────────────────────────────────────────

zoomval      = cfg["display"]["zoomval"]
width        = cfg["roi"]["width_per_zoom"] * zoomval
comwidth     = cfg["roi"]["comwidth_per_zoom"] * zoomval
scan = scannum = cfg["scan"]["scannum"]
datapoint0   = cfg["scan"]["datapoint0"]
datapoint    = cfg["scan"]["datapoint"]
scanpath     = cfg["scan"]["scanpath"]
imnum        = datapoint + 1
tolerance    = cfg["computation"]["tolerance"]
scatv        = cfg["geometry"]["scatv"]
detoptimize  = cfg["flags"]["detoptimize"]
energyopt    = cfg["flags"]["energyopt"]
colourlim    = cfg["display"]["colourlim"]
colmap       = cfg["display"]["colmap"]
bravais      = cfg["computation"]["bravais"]
autoreflist  = cfg["flags"]["autoreflist"]
OptMethod    = cfg["computation"]["opt_method"]
strat        = ts.DE_Strategy['best1exp']
intensity    = cfg["computation"]["intensity"]
threshold    = cfg["computation"]["threshold"]
numsteps     = cfg["computation"]["numsteps"]
numsteps_interactive = min(numsteps, 300)
simsigma     = cfg["computation"]["simsigma_per_zoom"] * zoomval
lattice2     = cfg["crystal"]["lattice2"]
cif_file     = cfg["paths"]["cif_file"]
show_centres = cfg["flags"].get("show_centres", 1)
show_numbers = cfg["flags"].get("show_numbers", 1)
axis_off     = cfg["flags"].get("axis_off", 0)
datestr      = strftime("%Y%m%d%H%M")

# ── Load experimental data ────────────────────────────────────────────────────

exp = resolve_experiment(cfg)
lattice = list(exp["lattice"])
energy = float(exp["energy"])
energy0 = float(exp["energy0"])
azir = list(exp["azir"])
imtemplate = exp["image_template"]
psi = cfg["geometry"]["psi"]
hkl = np.array(cfg["geometry"]["hkl"])
hkl = hkl * energy / energy0
hklint = np.round(hkl)

# ── Load and filter image ─────────────────────────────────────────────────────

im_raw = imageio.imread(os.path.join(scanpath, imtemplate % imnum))
im = ndimage.zoom(im_raw, zoomval, order=3)
imdata = np.copy(im)

px = cfg["geometry"]["px_unscaled"] * zoomval
py = cfg["geometry"]["py_unscaled"] * zoomval

thb = ts.bragg(lattice, hkl, energy).th()[0]
_td = cfg["computation"]["thrange_delta"]
thrange = [thb + _td[0], thb + _td[1]]
psirange = [psi - 360, psi + 360]
detvects = np.matrix([[1, 0, 0], [0, 0, 1]])
hkllist = ts.pilkhlrange(lattice, hkl, energy, thrange[0], thrange[1]).hklscan(numsteps)
hkllistrange = [thrange[0], thrange[1], numsteps]

# ── Build reflection list ─────────────────────────────────────────────────────

if autoreflist:
    mslist = [[np.NAN] * 7]
    hkllistcorse = ts.pilkhlrange(lattice, hkl, energy, thrange[0], thrange[1]).hklscan(30)
    SF, reflist, lattice2, structure, sfc = ts.loadcif(cif_file, energy)
    for hklval in range(len(hkllistcorse[:, 0])):
        ms = ts.calcms(lattice, hkllistcorse[hklval, :], hklint, reflist, energy, azir)
        mslist = np.concatenate((mslist, ms.full()), 0)
    mslist = ts.reducebypsirange(mslist, psirange)
    reflist = np.matrix(ts.uniquearray(mslist[:, 0:3]))
    reflist2 = 0
    ref_6d = None
else:
    ref_6d = np.array(cfg["crystal"]["ref_6d"])
    _tau = cfg.get("crystal", {}).get("tau_approx", 55.0 / 34.0)
    p6d = ts.Projection6dArrayApproximant(ref_6d, _tau)
    reflist0 = p6d.reflection_6d()
    reflist  = np.array(reflist0[0])
    reflist2 = np.array(reflist0[1])

# ── Build initial parameter vector ───────────────────────────────────────────
# 24-element vector:
#   [a, b, c, alpha, beta, gamma, psicor, hcor, kcor, lcor,
#    detdist, dxrot, dyrot, dzrot, energy, a11..a33]

ig_base = np.array(cfg["crystal"]["initial_guess_base"], dtype=float)
ig_base[10] = ig_base[10] / 2 * zoomval
_energy_offset = cfg["crystal"]["initial_guess_base"][14]
ig_base[14] = energy + ig_base[14]
initial_guess = ig_base.copy()

detdistancepx = initial_guess[10]
rotx  = initial_guess[11]
roty  = initial_guess[12]
rotz  = initial_guess[13]
mtrx2 = list(initial_guess[15:24])

# ── Build ROI kernels & extract centres ──────────────────────────────────────

print('Building ROI kernels...')
builderargs = (
    reflist, hkllist, hklint, intensity, psirange, threshold, hkl,
    detvects, imdata.shape, simsigma, azir, psi, px, py, scatv,
    detdistancepx, rotx, roty, rotz, energy,
    initial_guess, reflist2, mtrx2
)
kernel = ts.roibuilder_ico_hkl(builderargs)

print('Extracting ROI centres...')
imcoeffs, linedatax, linedatay, fitpoints, rois, pcov = ts.multiroifit2(
    imdata, kernel, width, 0.02, 10.0
)
centres = np.array([imcoeffs[:, 2]]).T

for _idx, _val in cfg["manual_centres"].items():
    centres[int(_idx)] = _val / 2 * zoomval

# ── Helper: extract reduced parameter vector ─────────────────────────────────

def extract_reduced(full_ig):
    if bravais == 'icosahedral':
        if detoptimize:
            if energyopt:
                idx = [0,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
            else:
                idx = [0,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23]
        else:
            if energyopt:
                idx = [0,6,7,8,9,14,15,16,17,18,19,20,21,22,23]
            else:
                idx = [0,6,7,8,9,15,16,17,18,19,20,21,22,23]
    elif bravais == 'icosahedral_fixed_a':
        if detoptimize:
            if energyopt:
                idx = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
            else:
                idx = [6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23]
        else:
            if energyopt:
                idx = [6,7,8,13,14,15,16,17,18,19,20,21,22,23]
            else:
                idx = [6,7,8,14,15,16,17,18,19,20,21,22,23]
    elif bravais == 'cubic_no_strain':
        if detoptimize:
            if energyopt:
                idx = [0,6,7,8,9,10,11,12,13]
            else:
                idx = [0,6,7,8,9,10,11,12]
        else:
            if energyopt:
                idx = [0,6,7,8,13]
            else:
                idx = [0,6,7,8]
    elif bravais == 'calibrate':
        if detoptimize:
            if energyopt:
                idx = [6,7,8,9,10,11,12,13]
            else:
                idx = [6,7,8,9,10,11,12]
        else:
            if energyopt:
                idx = [6,7,8,13]
            else:
                idx = [6,7,8]
    else:
        raise ValueError(f'Unknown bravais: {bravais}')
    return full_ig[idx]

# ── Create fitting object ────────────────────────────────────────────────────

dms = ts.dmsfit_ico_hkl(
    reflist, list(hkllistrange), hklint, psirange, width, centres, kernel,
    hkl, detvects, imdata, simsigma, azir, psi, px, py, scatv,
    bravais, detoptimize, energyopt,
    detdistancepx, rotx, roty, rotz, energy,
    reflist2, mtrx2, initial_guess[0]
)
dms.setCalLattice(initial_guess[:6].tolist())
dms.setLattice(initial_guess[:6].tolist())

# ── Initial computation ──────────────────────────────────────────────────────

ig = initial_guess.copy()
ig_reduced = extract_reduced(ig)
imdata_max = imdata.max()

dms.hkllistrange[2] = numsteps_interactive
try:
    dms.imcalc(ig_reduced)
except Exception:
    pass
imoverlay = np.copy(imdata)

print(f'Ready. {kernel.shape[2]} ROIs, {len(centres)} centres.')

# ── Slider definitions ────────────────────────────────────────────────────────

slider_defs = [
    ('a',      0,   0.2,   '%0.6f'),
    ('psicor', 6,   5.0,   '%0.6f'),
    ('hcor',   7,   2.0,   '%0.6f'),
    ('kcor',   8,   2.0,   '%0.6f'),
    ('lcor',   9,   2.0,   '%0.6f'),
    ('detdist',10,  300.0, '%0.3f'),
    ('dxrot',  11,  5.0,   '%0.6f'),
    ('dyrot',  12,  5.0,   '%0.6f'),
    ('dzrot',  13,  10.0,  '%0.6f'),
    ('energy', 14,  0.5,   '%0.6f'),
    ('a11',    15,  0.05,  '%0.6f'),
    ('a12',    16,  0.05,  '%0.6f'),
    ('a13',    17,  0.05,  '%0.6f'),
    ('a21',    18,  0.05,  '%0.6f'),
    ('a22',    19,  0.05,  '%0.6f'),
    ('a23',    20,  0.05,  '%0.6f'),
    ('a31',    21,  0.05,  '%0.6f'),
    ('a32',    22,  0.05,  '%0.6f'),
    ('a33',    23,  0.05,  '%0.6f'),
]

algo_display = ['COBYLA', 'Nelder-Mead', 'Powell', 'BH+Powell',
                'BH+COBYLA', 'BH+N-Mead', 'GA']
algo_methods = ['COBYLA', 'Nelder-Mead', 'Powell', 'BHPowell',
                'BHCOBYLA', 'BHNelderMead', 'GA']

# ── Float slider widget ───────────────────────────────────────────────────────

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
        self._vl.setFixedWidth(82)
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


# ── ROI comparison window ─────────────────────────────────────────────────────

class ROIWindow(QtWidgets.QWidget):
    def __init__(self, linedatax, linedatay, fitpoints,
                 linedatasimx, linedatasimy, fitpointssim,
                 centres, ref_6d, kernel,
                 show_centres, show_numbers, axis_off,
                 subcellsx, subcellsy, parent=None):
        super().__init__(parent)
        self.setWindowTitle('ROI Comparison — Data vs Fit')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        gw = pg.GraphicsLayoutWidget()
        layout.addWidget(gw)

        n_rois   = kernel.shape[2]
        refnum   = 0
        roicount = 0

        for i in range(min(n_rois, subcellsx * subcellsy)):
            r, c = divmod(i, subcellsy)
            pl = gw.addPlot(row=r, col=c)
            pl.setMenuEnabled(False)
            pl.hideButtons()
            pl.setDefaultPadding(0.02)

            pl.plot(linedatax[i], linedatay[i],
                    pen=pg.mkPen('g', width=0.5))
            pl.plot(linedatax[i], fitpoints[i],
                    pen=None, symbol='o', symbolSize=2,
                    symbolBrush='r', symbolPen=None)

            if show_centres:
                pl.addLine(x=centres[i][0], pen=pg.mkPen('g', width=0.5))

            denom = linedatasimy[i].max() - linedatasimy[i].min()
            if abs(denom) < 1e-10:
                denom = 1.0
            yscale  = (linedatay[i].max() - linedatay[i].min()) / denom
            yoffset = linedatay[i].min() - (linedatasimy[i] * yscale).min()

            pl.plot(linedatasimx[i], linedatasimy[i] * yscale + yoffset,
                    pen=pg.mkPen('b', width=0.5,
                                 style=QtCore.Qt.DashDotLine))
            pl.plot(linedatasimx[i], fitpointssim[i] * yscale + yoffset,
                    pen=None, symbol='o', symbolSize=2,
                    symbolBrush='g', symbolPen=None)

            if ref_6d is not None:
                title = (f'{i} {ref_6d[refnum,:]}' if show_numbers
                         else str(ref_6d[refnum, :]))
            else:
                title = str(i)
            pl.setTitle(title, size='7pt')

            if axis_off:
                pl.hideAxis('left')
                pl.hideAxis('bottom')

            if roicount == 1:
                refnum  += 1
                roicount = -1
            roicount += 1

        self.resize(700, 900)


def _ref_pen(j, n, width=1.5):
    """Consistent HSV colour for reflection index j out of n."""
    return pg.mkPen(pg.hsvColor(j / max(n, 1), 0.85, 0.95, 0.8), width=width)


# ── Worker threads ────────────────────────────────────────────────────────────

class UpdateWorker(QtCore.QThread):
    """Runs imcalc + multiroifit in background; discards stale requests."""
    done = QtCore.pyqtSignal(np.ndarray, object, object)  # img, sim_data, dmslines

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reduced = None
        self._mutex   = QtCore.QMutex()
        self._cond    = QtCore.QWaitCondition()
        self._quit    = False
        self.idle     = threading.Event()
        self.idle.set()

    def submit(self, reduced):
        locker = QtCore.QMutexLocker(self._mutex)
        self._reduced = reduced.copy()
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
            while self._reduced is None and not self._quit:
                self._cond.wait(self._mutex)
            if self._quit:
                self._mutex.unlock()
                return
            reduced       = self._reduced
            self._reduced = None
            self._mutex.unlock()

            self.idle.clear()
            try:
                dms.hkllistrange[2] = numsteps_interactive
                dms.imcalc(reduced)
                sim_data = None
                if dms.imsim is not None:
                    coefs, ldsx, ldsy, _, _, _ = ts.multiroifit(
                        dms.imsim, kernel, width, 10)
                    sim_data = (coefs, ldsx, ldsy)
                dmslines = [(np.copy(x), np.copy(y)) for x, y in dms.dmslines] \
                    if hasattr(dms, 'dmslines') else []
                self.done.emit(np.copy(imdata), sim_data, dmslines)
            except Exception:
                pass
            finally:
                self.idle.set()


class FitWorker(QtCore.QThread):
    """Runs the scipy optimizer in background."""
    done    = QtCore.pyqtSignal(dict)
    error   = QtCore.pyqtSignal(str, float)
    stopped = QtCore.pyqtSignal(float)

    def __init__(self, reduced, bounds, method, n_starts, parent=None):
        super().__init__(parent)
        self._reduced    = reduced.copy()
        self._bounds     = bounds
        self._method     = method
        self._n_starts   = n_starts
        self._t0         = time.time()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        reduced = self._reduced
        bounds  = self._bounds
        cur     = self._method
        ev      = self._stop_event

        def _fit_checked(x):
            if ev.is_set():
                raise StopIteration('stopped')
            return dms.fit(x)

        def _cb_check(*_args, **_kwargs):
            return ev.is_set()

        try:
            if cur == 'GA':
                print('Using Differential Evolution with strategy ' + strat)
                res = differential_evolution(_fit_checked, bounds, strategy=strat,
                                             polish=not ev.is_set(), workers=1,
                                             callback=_cb_check)
            elif cur in ('BHPowell', 'BHCOBYLA', 'BHNelderMead'):
                bh_map = {'BHPowell':     ('Powell',     150),
                          'BHCOBYLA':     ('COBYLA',     400),
                          'BHNelderMead': ('Nelder-Mead', 400)}
                method, niter = bh_map[cur]
                print(f'Using Basinhopping ({method})')
                res = basinhopping(_fit_checked, reduced,
                                   minimizer_kwargs={"method": method},
                                   niter=niter, callback=_cb_check)
            else:
                n = self._n_starts
                print(f'Using {cur} with {n}-start parallel search')
                rng = np.random.default_rng(42)
                starts = [reduced] + [
                    reduced + rng.uniform(-0.5, 0.5, reduced.shape)
                    for _ in range(n - 1)
                ]
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
                'opt': opt, 'simim': simim, 'dmsindex': dmsindex,
                'dmslines': dmslines,
                'inputarray': np.array(inputarray),
                'elapsed': elapsed, 'method': cur,
            })
        except StopIteration:
            self.stopped.emit(time.time() - self._t0)
        except Exception as e:
            self.error.emit(str(e), time.time() - self._t0)
            import traceback; traceback.print_exc()


# ── Main window ───────────────────────────────────────────────────────────────

class DMSWorkflow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('DMS Workflow — Slider → Fit')

        self._suppress      = False
        self._roi_win       = None
        self._fitting       = False
        self._fit_worker    = None
        self._active_method = (OptMethod if OptMethod in algo_methods
                               else algo_methods[0])

        self._update_worker = UpdateWorker()
        self._update_worker.done.connect(self._on_update_done,
                                         QtCore.Qt.QueuedConnection)

        # Debounce timer: fires 200 ms after the last slider move
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(200)
        self._update_timer.timeout.connect(self._do_update)

        self._build_ui()
        self._refresh_image(imoverlay)
        self._init_line_plot()
        self._update_sim_lines(
            [(np.copy(x), np.copy(y)) for x, y in dms.dmslines]
            if hasattr(dms, 'dmslines') else []
        )

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(0)
        root = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root_layout.addWidget(root)

        # Image panel (left)
        gw = pg.GraphicsLayoutWidget()
        gw.setMinimumWidth(550)
        vb = gw.addViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)
        vb.setMenuEnabled(False)

        self._img_item = pg.ImageItem()
        vb.addItem(self._img_item)

        try:
            cmap = pg.colormap.get(colmap, source='matplotlib')
        except Exception:
            cmap = pg.colormap.get('grey')
        self._img_item.setColorMap(cmap)
        self._img_item.setLevels(colourlim)

        self._img_vb = vb
        self._sim_lines = []   # pg.PlotCurveItem per reflection, populated lazily
        self._highlight_curve = pg.PlotDataItem(
            pen=pg.mkPen('#ffff00', width=2))
        self._highlight_curve.setZValue(10)
        vb.addItem(self._highlight_curve)

        self._title_lbl = QtWidgets.QLabel(
            f'Scan {scannum}  dp={datapoint}  E={energy:.4f} keV')
        self._title_lbl.setAlignment(QtCore.Qt.AlignCenter)

        self._coord_lbl = QtWidgets.QLabel('row —  col —  I=—')
        self._coord_lbl.setAlignment(QtCore.Qt.AlignCenter)
        f = self._coord_lbl.font()
        f.setFamily('monospace')
        f.setPointSize(8)
        self._coord_lbl.setFont(f)

        self._mouse_proxy = pg.SignalProxy(
            gw.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)

        img_col = QtWidgets.QVBoxLayout()
        img_col.addWidget(self._title_lbl)
        img_col.addWidget(gw, 1)
        img_col.addWidget(self._coord_lbl)
        img_w = QtWidgets.QWidget()
        img_w.setLayout(img_col)
        img_w.setMinimumWidth(400)
        root.addWidget(img_w)

        # ── Middle column: fitting controls (narrow) ──────────────────────────
        fit_col = QtWidgets.QVBoxLayout()
        fit_col.setSpacing(4)
        fit_w = QtWidgets.QWidget()
        fit_w.setLayout(fit_col)
        fit_w.setMinimumWidth(180)

        # Scan / DP / Load row
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel('Scan #'))
        self._tb_scan = QtWidgets.QLineEdit(str(scannum))
        self._tb_scan.setFixedWidth(60)
        top_row.addWidget(self._tb_scan)
        top_row.addWidget(QtWidgets.QLabel('DP'))
        self._tb_dp = QtWidgets.QLineEdit(str(datapoint))
        self._tb_dp.setFixedWidth(40)
        top_row.addWidget(self._tb_dp)
        btn_load = QtWidgets.QPushButton('Load')
        btn_load.setStyleSheet('background: #6b2020; color: #ffcccc')
        btn_load.clicked.connect(self._on_load)
        top_row.addWidget(btn_load)
        top_row.addStretch()
        fit_col.addLayout(top_row)

        # Editable config table (metadata + key scalars; applied on next Load)
        cfg_box = QtWidgets.QGroupBox('Config')
        cfg_box_layout = QtWidgets.QVBoxLayout(cfg_box)
        cfg_box_layout.setContentsMargins(2, 2, 2, 2)
        self._cfgtable = ConfigTable()
        self._cfgtable.set_config(cfg)
        self._cfgtable.set_save_path(cfg_path)
        self._cfgtable.configChanged.connect(self._on_cfg_table_changed)
        self._cfgtable.setMaximumHeight(220)
        cfg_box_layout.addWidget(self._cfgtable)
        fit_col.addWidget(cfg_box)

        # Sliders in a scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        vbox  = QtWidgets.QVBoxLayout(inner)
        vbox.setSpacing(1)
        vbox.setContentsMargins(0, 0, 0, 0)

        self._sliders = {}
        for label, idx, half, fmt in slider_defs:
            val = ig[idx]
            fs  = FloatSlider(label, val, val - half, val + half, fmt)
            fs.valueChanged.connect(self._on_slider_changed)
            vbox.addWidget(fs)
            self._sliders[idx] = fs

        vbox.addStretch()
        scroll.setWidget(inner)
        fit_col.addWidget(scroll, 1)

        # Algorithm selector
        algo_box = QtWidgets.QGroupBox('Fit algorithm')
        algo_layout = QtWidgets.QVBoxLayout(algo_box)
        algo_layout.setSpacing(2)
        self._algo_btns = {}
        for disp, meth in zip(algo_display, algo_methods):
            rb = QtWidgets.QRadioButton(disp)
            if meth == self._active_method:
                rb.setChecked(True)
            rb.clicked.connect(lambda checked, m=meth: self._on_algo(m))
            algo_layout.addWidget(rb)
            self._algo_btns[meth] = rb
        fit_col.addWidget(algo_box)

        # Fit / Stop / Reset / Print
        fit_row = QtWidgets.QHBoxLayout()
        btn_fit = QtWidgets.QPushButton('Fit')
        btn_fit.setStyleSheet('background: #1a5c1a; color: #ccffcc; font-weight: bold')
        btn_fit.setMinimumHeight(32)
        btn_fit.clicked.connect(self._do_fit)
        self._btn_stop = QtWidgets.QPushButton('Stop')
        self._btn_stop.setStyleSheet('background: #5c1a1a; color: #ffcccc; font-weight: bold')
        self._btn_stop.setMinimumHeight(32)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop_fit)
        fit_row.addWidget(btn_fit, 2)
        fit_row.addWidget(self._btn_stop, 1)
        fit_col.addLayout(fit_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton('Reset')
        btn_reset.setStyleSheet('background: #4a4a10; color: #ffffcc')
        btn_reset.clicked.connect(self._on_reset)
        btn_print = QtWidgets.QPushButton('Print')
        btn_print.setStyleSheet('background: #103050; color: #cce0ff')
        btn_print.clicked.connect(self._on_print)
        btn_row.addWidget(btn_reset)
        btn_row.addWidget(btn_print)
        fit_col.addLayout(btn_row)

        # Status label
        self._status = QtWidgets.QLabel(
            'Ready — adjust sliders then select algorithm to fit')
        self._status.setWordWrap(True)
        f = self._status.font()
        f.setFamily('monospace')
        f.setPointSize(8)
        self._status.setFont(f)
        fit_col.addWidget(self._status)
        root.addWidget(fit_w)

        # ── Right column: ROI line plots ──────────────────────────────────────
        roi_col = QtWidgets.QVBoxLayout()
        roi_w = QtWidgets.QWidget()
        roi_w.setLayout(roi_col)
        roi_w.setMinimumWidth(200)

        self._roi_grid = pg.GraphicsLayoutWidget()
        self._exp_curves = []
        self._sim_curves = []
        self._roi_plots   = []
        self._selected_roi = None
        self._roi_grid.scene().sigMouseClicked.connect(self._on_roi_grid_clicked)
        roi_col.addWidget(self._roi_grid, 1)

        self._roi_coord_lbl = QtWidgets.QLabel('x=—  y=—')
        self._roi_coord_lbl.setAlignment(QtCore.Qt.AlignCenter)
        f2 = self._roi_coord_lbl.font()
        f2.setFamily('monospace')
        f2.setPointSize(8)
        self._roi_coord_lbl.setFont(f2)
        roi_col.addWidget(self._roi_coord_lbl)

        self._roi_mouse_proxy = pg.SignalProxy(
            self._roi_grid.scene().sigMouseMoved,
            rateLimit=60, slot=self._on_roi_mouse_moved)
        root.addWidget(roi_w)

        # Initial column proportions: image 3 : fit 1 : roi 2
        root.setSizes([800, 267, 533])
        root.setStretchFactor(0, 3)
        root.setStretchFactor(1, 1)
        root.setStretchFactor(2, 2)

        self.resize(1600, 850)

    # ── Line profile plot ─────────────────────────────────────────────────────

    def _init_line_plot(self):
        self._roi_grid.clear()
        self._exp_curves        = []
        self._sim_curves        = []
        self._exp_centre_lines  = []
        self._sim_centre_lines  = []
        self._roi_plots         = []
        self._centre_override_rois = set()
        self._selected_roi = None
        self._highlight_curve.setData([], [])
        n        = kernel.shape[2]
        ncols    = cfg["display"].get("subcellsy", 4)
        refnum   = 0
        roicount = 0
        for i in range(n):
            r, c = divmod(i, ncols)
            pl = self._roi_grid.addPlot(row=r, col=c)
            pl.setMenuEnabled(False)
            pl.hideButtons()
            pl.hideAxis('left')
            pl.hideAxis('bottom')
            pl.setDefaultPadding(0.05)

            if ref_6d is not None:
                lbl = (f'{i}: {list(ref_6d[refnum])}' if show_numbers
                       else str(list(ref_6d[refnum])))
            else:
                lbl = str(i)
            pl.setTitle(lbl, size='7pt')

            cur_refnum = refnum
            if roicount == 1:
                refnum  += 1
                roicount = -1
            roicount += 1

            self._exp_curves.append(pl.plot(pen=pg.mkPen('#4488ff', width=1)))
            self._sim_curves.append(pl.plot(pen=_ref_pen(cur_refnum, len(reflist), width=1)))

            exp_cl = pg.InfiniteLine(angle=90, movable=False,
                pen=pg.mkPen('#4488ff', width=1, style=QtCore.Qt.DashLine))
            sim_cl = pg.InfiniteLine(angle=90, movable=False,
                pen=pg.mkPen(pg.hsvColor(cur_refnum / max(len(reflist), 1),
                                         0.85, 0.95, 0.8),
                             width=1, style=QtCore.Qt.DashLine))
            pl.addItem(exp_cl)
            pl.addItem(sim_cl)
            self._exp_centre_lines.append(exp_cl)
            self._sim_centre_lines.append(sim_cl)

            self._roi_plots.append(pl)
        self._draw_exp_lines()
        self._try_draw_sim_lines()

    def _try_draw_sim_lines(self):
        if dms.imsim is not None and self._sim_curves:
            try:
                coefs, ldsx, ldsy, _, _, _ = ts.multiroifit(dms.imsim, kernel, width, 10)
                self._draw_sim_lines(coefs, ldsx, ldsy)
            except Exception:
                pass

    def _draw_exp_lines(self):
        for i, curve in enumerate(self._exp_curves):
            curve.setData(linedatax[i], linedatay[i])
        for i, cl in enumerate(self._exp_centre_lines):
            overridden = i in self._centre_override_rois
            if not overridden and i < len(imcoeffs):
                cl.setValue(float(imcoeffs[i, 2]))
            cl.setPen(
                pg.mkPen('#ffaa00', width=1.5)
                if overridden
                else pg.mkPen('#4488ff', width=1, style=QtCore.Qt.DashLine)
            )

    def _draw_sim_lines(self, ldscoeffs, ldsx, ldsy):
        for i, curve in enumerate(self._sim_curves):
            if i >= len(ldsy):
                break
            y_exp = linedatay[i]
            y_sim = ldsy[i]
            denom = y_sim.max() - y_sim.min()
            if abs(denom) < 1e-10:
                denom = 1.0
            yscale  = (y_exp.max() - y_exp.min()) / denom
            yoffset = y_exp.min() - (y_sim * yscale).min()
            curve.setData(ldsx[i], y_sim * yscale + yoffset)
        for i, cl in enumerate(self._sim_centre_lines):
            if i < len(ldscoeffs):
                cl.setValue(float(ldscoeffs[i, 2]))

    # ── ROI click / highlight ─────────────────────────────────────────────────

    def _on_roi_grid_clicked(self, event):
        pos = event.scenePos()
        for i, pl in enumerate(self._roi_plots):
            if pl.vb.sceneBoundingRect().contains(pos):
                if event.button() == QtCore.Qt.RightButton:
                    pt = pl.vb.mapSceneToView(pos)
                    self._set_centre_override(i, pt.x())
                    event.accept()
                    return
                # Left click: toggle ROI highlight
                if self._selected_roi == i:
                    self._selected_roi = None
                    self._highlight_curve.setData([], [])
                    pl.vb.setBackgroundColor(None)
                else:
                    if self._selected_roi is not None:
                        self._roi_plots[self._selected_roi].vb.setBackgroundColor(None)
                    self._selected_roi = i
                    pl.vb.setBackgroundColor((60, 40, 0, 80))
                    self._highlight_roi(i)
                break

    def _set_centre_override(self, roi_idx, x):
        centres[roi_idx, 0] = x
        dms.centres[roi_idx, 0] = x
        self._centre_override_rois.add(roi_idx)
        if roi_idx < len(self._exp_centre_lines):
            self._exp_centre_lines[roi_idx].setValue(x)
            self._exp_centre_lines[roi_idx].setPen(pg.mkPen('#ffaa00', width=1.5))
        self._status.setText(
            f'Centre override ROI {roi_idx}: x={x:.1f}  '
            f'(right-click again to reposition, Load to clear all overrides)')
        self._do_update()

    def _highlight_roi(self, roi_idx):
        mask = kernel[:, :, roi_idx] > 0
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            self._highlight_curve.setData([], [])
            return
        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])
        self._highlight_curve.setData(
            [c0, c1, c1, c0, c0],
            [r0, r0, r1, r1, r0])

    # ── Image refresh ─────────────────────────────────────────────────────────

    def _refresh_image(self, arr):
        self._img_item.setImage(arr, autoLevels=False)
        self._img_item.setLevels(colourlim)

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self._img_vb.sceneBoundingRect().contains(pos):
            pt  = self._img_vb.mapSceneToView(pos)
            row = int(pt.y())
            col = int(pt.x())
            if 0 <= row < imdata.shape[0] and 0 <= col < imdata.shape[1]:
                val = imdata[row, col]
                self._coord_lbl.setText(f'row {row:4d}  col {col:4d}  I={val:.1f}')
            else:
                self._coord_lbl.setText(f'row {row:4d}  col {col:4d}  I=—')
        else:
            self._coord_lbl.setText('row —  col —  I=—')

    def _on_roi_mouse_moved(self, evt):
        pos = evt[0]
        for pl in self._roi_plots:
            if pl.vb.sceneBoundingRect().contains(pos):
                pt = pl.vb.mapSceneToView(pos)
                self._roi_coord_lbl.setText(f'x={pt.x():.1f}  y={pt.y():.4g}')
                return
        self._roi_coord_lbl.setText('x=—  y=—')

    # ── Slider handling ───────────────────────────────────────────────────────

    def _on_slider_changed(self, _val):
        if not self._suppress:
            self._update_timer.start()

    def _sync_ig(self):
        for idx, fs in self._sliders.items():
            ig[idx] = fs.val
        ig[1] = ig[2] = ig[0]
        ig[3] = ig[4] = ig[5] = 90.0

    def _do_update(self):
        if self._fitting:
            return
        self._sync_ig()
        self._update_worker.submit(extract_reduced(ig))

    def _on_update_done(self, img, sim_data, dmslines):
        self._refresh_image(img)
        self._update_sim_lines(dmslines)
        if sim_data is not None and self._sim_curves:
            self._draw_sim_lines(*sim_data)

    def _update_sim_lines(self, dmslines):
        n = len(dmslines)
        if len(self._sim_lines) != n:
            for item in self._sim_lines:
                self._img_vb.removeItem(item)
            self._sim_lines = []
            for j in range(n):
                item = pg.PlotCurveItem(pen=_ref_pen(j, n), connect='finite')
                item.setZValue(5)
                self._img_vb.addItem(item)
                self._sim_lines.append(item)
        for j, (x, y) in enumerate(dmslines):
            self._sim_lines[j].setData(x=x, y=y)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def _on_algo(self, method):
        self._active_method = method

    def _do_fit(self):
        if self._fitting:
            return
        self._fitting = True
        self._sync_ig()
        self._status.setText('Fitting...')

        reduced = extract_reduced(ig)
        dms.hkllistrange[2] = numsteps
        dms.detdistancepx = ig[10]
        dms.detxrot       = ig[11]
        dms.detyrot       = ig[12]
        dms.detzrot       = ig[13]
        dms.energy        = ig[14]
        dms.a             = ig[0]
        dms.setLattice([ig[0], ig[0], ig[0], 90, 90, 90])

        iglow  = reduced - 1.5
        ighigh = reduced + 1.5
        bounds = list(zip(iglow, ighigh))
        n_starts = cfg["computation"].get("n_parallel_starts", 4)

        # Wait for any in-progress imcalc to finish before touching dms
        self._update_worker.idle.wait(timeout=5.0)

        self._fit_worker = FitWorker(
            reduced, bounds, self._active_method, n_starts)
        self._fit_worker.done.connect(self._on_fit_done)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.stopped.connect(self._on_fit_stopped)
        self._btn_stop.setEnabled(True)
        self._fit_worker.start()

    def _on_fit_done(self, result):
        self._fitting = False
        self._btn_stop.setEnabled(False)
        opt        = result['opt']
        simim      = result['simim']
        inputarray = result['inputarray']
        elapsed    = result['elapsed']
        method     = result['method']

        self._suppress = True
        for idx, fs in self._sliders.items():
            if idx < len(inputarray):
                fs.setValue(inputarray[idx])
        ig[:] = inputarray
        self._suppress = False

        self._refresh_image(np.copy(imdata))
        self._update_sim_lines(result.get('dmslines', []))
        self._status.setText(
            f'Fit complete.  χ²={opt:.4f}  t={elapsed:.1f}s  [{method}]')
        print(f'\nFit complete in {elapsed:.1f}s')
        print(f'χ² = {opt:.6f}')
        print('initial_guess = np.array(['
              + ','.join(f'{v:.6f}' for v in inputarray) + '])')
        self._try_draw_sim_lines()
        self._show_roi_comparison(simim)

    def _on_fit_error(self, msg, elapsed):
        self._fitting = False
        self._btn_stop.setEnabled(False)
        self._status.setText(f'Fit failed: {msg}')
        print(f'\nFit failed after {elapsed:.1f}s: {msg}')

    def _on_stop_fit(self):
        if self._fit_worker and self._fit_worker.isRunning():
            self._btn_stop.setEnabled(False)
            self._status.setText('Stopping fit...')
            self._fit_worker.stop()

    def _on_fit_stopped(self, elapsed):
        self._fitting = False
        self._btn_stop.setEnabled(False)
        self._status.setText(f'Fit stopped after {elapsed:.1f}s')
        print(f'\nFit stopped after {elapsed:.1f}s')

    def closeEvent(self, event):
        self._update_worker.stop()
        if self._fit_worker and self._fit_worker.isRunning():
            self._fit_worker.wait()
        super().closeEvent(event)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _on_reset(self):
        ig[:] = initial_guess
        self._suppress = True
        for idx, fs in self._sliders.items():
            fs.setValue(ig[idx])
        self._suppress = False
        self._do_update()
        self._status.setText('Reset to initial guess from config')

    # ── Print ─────────────────────────────────────────────────────────────────

    def _on_print(self):
        self._sync_ig()
        print('\n' + '=' * 72)
        print('Current 24-element parameter vector:')
        print('initial_guess = np.array(['
              + ','.join(f'{v:.6f}' for v in ig) + '])')
        print()
        reduced = extract_reduced(ig)
        print(f'Reduced ({bravais}, detopt={detoptimize}, eopt={energyopt}):')
        print('ig = np.array(['
              + ','.join(f'{v:.6f}' for v in reduced) + '])')
        print()
        labels  = ['a', 'psicor', 'hcor', 'kcor', 'lcor', 'detdist',
                   'dxrot', 'dyrot', 'dzrot', 'energy',
                   'a11', 'a12', 'a13', 'a21', 'a22', 'a23',
                   'a31', 'a32', 'a33']
        indices = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                   15, 16, 17, 18, 19, 20, 21, 22, 23]
        for lbl, idx in zip(labels, indices):
            print(f'  {lbl:8s} = {ig[idx]:.6f}')
        print('=' * 72)

    # ── Config table ────────────────────────────────────────────────────────────

    def _on_cfg_table_changed(self, new_cfg):
        """Merge live table edits into the module config and keep the scan/dp
        line edits in sync.  Geometry/metadata edits take effect on next Load."""
        cfg.update(new_cfg)
        self._suppress = True
        self._tb_scan.setText(str(cfg["scan"].get("scannum", scannum)))
        self._tb_dp.setText(str(cfg["scan"].get("datapoint", datapoint)))
        self._suppress = False

    # ── Load ──────────────────────────────────────────────────────────────────

    def _on_load(self):
        global scannum, scan, datapoint, datapoint0, imnum
        global lattice, energy, energy0, azir, psi, px, py
        global hkl, hklint, im_raw, im, imdata, imdata_max
        global thb, thrange, hkllist, hkllistrange
        global kernel, imcoeffs, linedatax, linedatay, fitpoints, rois, pcov
        global centres, dms, imoverlay

        try:
            new_scan = int(self._tb_scan.text())
            new_dp   = int(self._tb_dp.text())
        except ValueError:
            self._status.setText('Invalid scan number or data point — enter integers')
            return

        self._status.setText(f'Loading scan {new_scan} dp={new_dp}...')
        QtWidgets.QApplication.processEvents()

        try:
            # Pull any GUI table edits (psi/px/py/datapoint0/…) into the live cfg
            cfg.update(self._cfgtable.to_config())

            scannum = scan = new_scan
            datapoint = new_dp
            datapoint0 = int(cfg["scan"].get("datapoint0", datapoint0))
            imnum = datapoint + 1
            cfg["scan"]["scannum"]   = scannum
            cfg["scan"]["datapoint"] = datapoint

            # A new scan/dp means fresh metadata; re-extract via the converter
            # (the only sanctioned .dat reader) and refresh the experiment block.
            dat_path = os.path.join(scanpath, str(scannum) + '.dat')
            exp = dat2config.extract_metadata(dat_path, datapoint, datapoint0)
            cfg["experiment"] = exp
            lattice = list(exp["lattice"])
            energy  = float(exp["energy"])
            energy0 = float(exp["energy0"])
            azir    = list(exp["azir"])
            _tpl    = exp["image_template"]

            psi = cfg["geometry"]["psi"]
            px  = cfg["geometry"]["px_unscaled"] * zoomval
            py  = cfg["geometry"]["py_unscaled"] * zoomval

            hkl_cfg = np.array(cfg["geometry"]["hkl"])
            hkl = hkl_cfg * energy / energy0
            hklint = np.round(hkl)

            self._cfgtable.set_config(cfg)

            im_raw = imageio.imread(os.path.join(scanpath, _tpl % imnum))
            im = ndimage.zoom(im_raw, zoomval, order=3)
            imdata = np.copy(im)
            imdata_max = imdata.max()

            thb = ts.bragg(lattice, hkl, energy).th()[0]
            _td = cfg["computation"]["thrange_delta"]
            thrange = [thb + _td[0], thb + _td[1]]
            hkllist = ts.pilkhlrange(
                lattice, hkl, energy, thrange[0], thrange[1]).hklscan(numsteps)
            hkllistrange = [thrange[0], thrange[1], numsteps]

            initial_guess[14] = energy + _energy_offset
            ig[:] = initial_guess

            builderargs = (
                reflist, hkllist, hklint, intensity, psirange, threshold, hkl,
                detvects, imdata.shape, simsigma, azir, psi, px, py, scatv,
                ig[10], ig[11], ig[12], ig[13], energy,
                ig, reflist2, list(ig[15:24])
            )
            kernel = ts.roibuilder_ico_hkl(builderargs)

            imcoeffs, linedatax, linedatay, fitpoints, rois, pcov = ts.multiroifit2(
                imdata, kernel, width, 0.02, 10.0)
            centres = np.array([imcoeffs[:, 2]]).T
            for _idx_str, _val in cfg["manual_centres"].items():
                centres[int(_idx_str)] = _val / 2 * zoomval

            dms = ts.dmsfit_ico_hkl(
                reflist, list(hkllistrange), hklint, psirange, width, centres, kernel,
                hkl, detvects, imdata, simsigma, azir, psi, px, py, scatv,
                bravais, detoptimize, energyopt,
                ig[10], ig[11], ig[12], ig[13], energy,
                reflist2, list(ig[15:24]), ig[0]
            )
            dms.setCalLattice(ig[:6].tolist())
            dms.setLattice(ig[:6].tolist())

            self._suppress = True
            for s_idx, fs in self._sliders.items():
                fs.setValue(ig[s_idx])
            self._suppress = False

            dms.hkllistrange[2] = numsteps_interactive
            try:
                dms.imcalc(extract_reduced(ig))
            except Exception:
                pass

            self._refresh_image(np.copy(imdata))
            self._update_sim_lines(
                [(np.copy(x), np.copy(y)) for x, y in dms.dmslines]
                if hasattr(dms, 'dmslines') else []
            )
            self._init_line_plot()
            self._title_lbl.setText(
                f'Scan {scannum}  dp={datapoint}  E={energy:.4f} keV')
            self._status.setText(
                f'Loaded scan {scannum} dp={datapoint}  E={energy:.4f} keV  '
                f'({kernel.shape[2]} ROIs)')

        except Exception as e:
            self._status.setText(f'Load failed: {e}')
            print(f'\nLoad failed: {e}')
            import traceback; traceback.print_exc()

    # ── ROI comparison ────────────────────────────────────────────────────────

    def _show_roi_comparison(self, simim):
        imcoeffs_sim, linedatasimx, linedatasimy, fitpointssim, rois2, covmat = \
            ts.multiroifit(simim, kernel, width, 10)

        subcellsx = cfg["display"].get("subcellsx", 7)
        subcellsy = cfg["display"].get("subcellsy", 4)

        self._roi_win = ROIWindow(
            linedatax, linedatay, fitpoints,
            linedatasimx, linedatasimy, fitpointssim,
            centres, ref_6d, kernel,
            show_centres, show_numbers, axis_off,
            subcellsx, subcellsy)
        self._roi_win.show()


# ── Launch ────────────────────────────────────────────────────────────────────

app = QtWidgets.QApplication(sys.argv)
app.setStyle('Fusion')
_p = QtGui.QPalette()
_dark   = QtGui.QColor(26,  26,  26)
_mid    = QtGui.QColor(42,  42,  42)
_light  = QtGui.QColor(58,  58,  58)
_text   = QtGui.QColor(210, 210, 210)
_hilit  = QtGui.QColor(42,  130, 218)
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
win = DMSWorkflow()
win.show()
sys.exit(app.exec_())
