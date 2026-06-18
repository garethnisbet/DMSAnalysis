#!/usr/bin/env python
"""
slider_quasi_AlPdMn_Annealed_hkl_v3.py
Interactive DMS simulation viewer – PyQtGraph, dark theme, background threading.
"""
import sys, os, time, itertools, threading, json, re, subprocess
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PyQt5')

PKGDIR  = os.path.abspath(os.path.dirname(__file__))
CONFIGS = os.path.join(PKGDIR, 'configs')

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

# ── initial guess (23-element: indices 0-22) ───────────────────────────────────
#   [0-5]  a b c α β γ   [6-8]  psicor hcor lcor   [9] detdist
#   [10-12] rotx roty rotz   [13] energy   [14-22] phason a11..a33
# NB: this matches dmscalc_ico.imcalc (3 corrections, no kcor); workflow.py uses a
# 24-element layout, so _slider_ig_to_workflow_ig24 bridges the two on export.
initial_guess = np.array([
    6.461053, 6.461053, 6.461053, 90., 90., 90.,
    -2.171374, -0.006984, -0.013387, 14480.587530 / 3 * zoomval,
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
    ('hcor',    7,    5.5,   '%0.6f'),
    ('lcor',    8,    5.5,   '%0.6f'),
    ('detdist', 9,  300.0,   '%0.3f'),
    ('rotx',   10,    5.0,   '%0.6f'),
    ('roty',   11,    5.0,   '%0.6f'),
    ('rotz',   12,   10.0,   '%0.6f'),
    ('energy', 13,    0.5,   '%0.6f'),
    ('a11',    14,   0.05,   '%0.7f'),
    ('a12',    15,   0.05,   '%0.7f'),
    ('a13',    16,   0.05,   '%0.7f'),
    ('a21',    17,   0.05,   '%0.7f'),
    ('a22',    18,   0.05,   '%0.7f'),
    ('a23',    19,   0.05,   '%0.7f'),
    ('a31',    20,   0.05,   '%0.7f'),
    ('a32',    21,   0.05,   '%0.7f'),
    ('a33',    22,   0.05,   '%0.7f'),
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

# ── initial reflist ────────────────────────────────────────────────────────────
_rl, _rl2       = build_reflist_from_6d(ref_6d_manual)
full_reflist    = np.array(_rl)
full_reflist2   = np.array(_rl2)
full_reflist_6d = np.array(ref_6d_manual)

_ig0   = initial_guess.copy()
_mtrx2 = list(_ig0[14:23])

_dms_init = ts.dmscalc_ico(
    np.matrix(full_reflist), hkllist, hklint, 1, psirange, 100,
    hkl, detvects, imdata, simsigma, azir, psi, px, py, scatv,
    _ig0[9], _ig0[10], _ig0[11], _ig0[12], _ig0[13],
    np.matrix(full_reflist2), _mtrx2)

_dms_full_init = ts.dmscalc_ico(
    np.matrix(full_reflist), hkllist, hklint, 1, psirange, 100,
    hkl, detvects, imdata, simsigma, azir, psi, px, py, scatv,
    _ig0[9], _ig0[10], _ig0[11], _ig0[12], _ig0[13],
    np.matrix(full_reflist2), _mtrx2)


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
    """Runs dms.full(ig) in a background thread; discards stale requests."""
    done = QtCore.pyqtSignal(object, object)   # rows ndarray, cols ndarray

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending  = None   # (ig_copy, dms_ref, hkl_copy, last_hkl_ref)
        self._mutex    = QtCore.QMutex()
        self._cond     = QtCore.QWaitCondition()
        self._quit     = False
        self.idle      = threading.Event()
        self.idle.set()
        self.lattice   = list(lattice)
        self.thrange   = list(thrange)

    def submit(self, ig, dms_ref, hkl_arr, last_hkl_ref):
        locker = QtCore.QMutexLocker(self._mutex)
        self._pending = (ig.copy(), dms_ref, hkl_arr.copy(), last_hkl_ref)
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
            ig, dms_ref, hkl_arr, last_hkl_ref = self._pending
            self._pending = None
            self._mutex.unlock()

            self.idle.clear()
            try:
                # Update hkl / hkllist if changed
                if not np.allclose(hkl_arr, last_hkl_ref):
                    dms_ref.sethkl(hkl_arr.copy())
                    hl = ts.pilkhlrange(
                        self.lattice, hkl_arr, ig[13], self.thrange[0], self.thrange[1]
                    ).hklscan(numsteps_interactive)
                    dms_ref.sethkllist(hl)
                    last_hkl_ref[:] = hkl_arr

                _, _, dmsindex, _ = dms_ref.full(ig)
                rows = dmsindex[0].astype(float)
                cols = dmsindex[1].astype(float)
                self.done.emit(rows, cols)
            except Exception as e:
                print('UpdateWorker error:', e)
            finally:
                self.idle.set()


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

        self._worker = UpdateWorker()
        self._worker.done.connect(self._on_update_done,
                                  QtCore.Qt.QueuedConnection)

        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(200)
        self._update_timer.timeout.connect(self._do_update)

        self._build_ui()
        self._do_update()

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

        btn_load_scan = QtWidgets.QPushButton('Load')
        btn_load_scan.setStyleSheet('background: #102020; color: #aaffff')
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

        # ── Workflow launcher ──────────────────────────────────────────────────
        wf_box = QtWidgets.QGroupBox('Workflow')
        wfl = QtWidgets.QGridLayout(wf_box)
        wfl.setSpacing(4)
        wfl.setContentsMargins(4, 4, 4, 4)

        self._lbl_wf_template = QtWidgets.QLabel(
            os.path.basename(self._workflow_template) if self._workflow_template
            else '(no template)')
        self._lbl_wf_template.setWordWrap(True)
        f_wt = self._lbl_wf_template.font()
        f_wt.setFamily('monospace')
        f_wt.setPointSize(7)
        self._lbl_wf_template.setFont(f_wt)
        wfl.addWidget(self._lbl_wf_template, 0, 0, 1, 3)

        btn_wf_tmpl = QtWidgets.QPushButton('Template…')
        btn_wf_tmpl.clicked.connect(self._on_browse_workflow_template)
        wfl.addWidget(btn_wf_tmpl, 1, 0)

        btn_wf_export = QtWidgets.QPushButton('Export JSON')
        btn_wf_export.setStyleSheet('background: #102030; color: #aaccff')
        btn_wf_export.clicked.connect(self._on_export_workflow_json)
        wfl.addWidget(btn_wf_export, 1, 1)

        btn_wf_launch = QtWidgets.QPushButton('Launch →')
        btn_wf_launch.setStyleSheet(
            'background: #102010; color: #aaffaa; font-weight: bold')
        btn_wf_launch.clicked.connect(self._on_launch_workflow)
        wfl.addWidget(btn_wf_launch, 1, 2)

        ctrl_col.addWidget(wf_box)

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

        # Reset / Print / Save JSON row
        btn_row = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton('Reset')
        btn_reset.setStyleSheet('background: #4a4a10; color: #ffffcc')
        btn_reset.clicked.connect(self._on_reset)
        btn_print = QtWidgets.QPushButton('Print ig')
        btn_print.setStyleSheet('background: #103050; color: #cce0ff')
        btn_print.clicked.connect(self._on_print)
        btn_save = QtWidgets.QPushButton('Save JSON')
        btn_save.setStyleSheet('background: #103010; color: #ccffcc')
        btn_save.clicked.connect(self._on_save_json)
        btn_load = QtWidgets.QPushButton('Load JSON')
        btn_load.setStyleSheet('background: #201030; color: #ddccff')
        btn_load.clicked.connect(self._on_load_json)
        btn_row.addWidget(btn_reset)
        btn_row.addWidget(btn_print)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_load)
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
        splitter.setSizes([1000, 340])
        self.resize(1380, 860)

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
        self._worker.submit(self.ig, self._dms, self._hkl, self._last_hkl)

    def _on_update_done(self, rows, cols):
        self._dms_scatter.setData(x=cols, y=rows)
        self._refresh_selected_arcs()
        self._status.setText('Ready')

    def _refresh_selected_arcs(self):
        arc_items = [a for a in self._pick_items if id(a) in self._arc_to_6d]
        if not arc_items:
            return
        hl = ts.pilkhlrange(
            [self.ig[0]] * 3 + [90, 90, 90], self._hkl, self.ig[13],
            self._thrange[0], self._thrange[1]).hklscan(numsteps)
        self._dms_full.hkllist = hl
        self._dms_full.sethkl(self._hkl.copy())
        for arc_item in arc_items:
            hkl_6d = self._arc_to_6d[id(arc_item)]
            rl1, rl2 = build_reflist_from_6d(hkl_6d.reshape(1, -1))
            self._dms_full.reflist  = np.matrix(rl1)
            self._dms_full.reflist2 = np.matrix(rl2)
            try:
                self._dms_full.imcalc(self.ig)
                pts2d = self._dms_full.pxv2d_all
                if pts2d.shape[0] == 0:
                    arc_item.setData(x=[], y=[])
                else:
                    x_arr = pts2d[:, 1].astype(float)
                    y_arr = pts2d[:, 0].astype(float)
                    arc_item.setData(x=x_arr, y=y_arr)
                    arc_item._x_data = x_arr
                    arc_item._y_data = y_arr
            except Exception as e:
                print('Refresh arc [%s]: %s' % (
                    ' '.join('%d' % v for v in hkl_6d), e))

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
        rl  = np.matrix(self.full_reflist[offset:end])
        rl2 = np.matrix(self.full_reflist2[offset:end])
        self._dms = ts.dmscalc_ico(
            rl, self._hkllist, self._hklint, 1, self._psirange, 100,
            self._hkl, detvects, self._imdata, simsigma, self._azir,
            self._psi, self._px, self._py, scatv,
            self.ig[9], self.ig[10], self.ig[11], self.ig[12], self.ig[13],
            rl2, list(self.ig[14:23]))
        self._last_hkl = np.full(3, np.inf)  # force sethkl on next update

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

    def _on_list_item_changed(self, list_item):
        """Checkbox toggle → show/hide the arc."""
        arc_item = list_item.data(QtCore.Qt.UserRole)
        if arc_item is not None:
            arc_item.setVisible(list_item.checkState() == QtCore.Qt.Checked)

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

    # ── Physics helpers ────────────────────────────────────────────────────────

    def _pixel_to_direction(self, row, col):
        a       = self.ig[0]
        thb_cur = ts.bragg([a, a, a, 90, 90, 90], self._hkl, self.ig[13]).th()[0]
        irmat   = np.array(
            ts.rotxyz([1, 0, 0], self.ig[10] + thb_cur).rmat() *
            ts.rotxyz([0, 1, 0], self.ig[11]).rmat() *
            ts.rotxyz([0, 0, 1], self.ig[12]).rmat()
        )
        pxvec    = np.array([row - self._dms.px, 0.0, self._dms.py - col])
        prepxvec = pxvec @ np.linalg.inv(irmat)
        centralv = -np.array(ts.psith2v(0.0, float(thb_cur))).flatten() * self.ig[9]
        diff     = prepxvec - centralv
        n        = np.linalg.norm(diff)
        if n < 1e-10:
            return None
        return diff / n

    def _ewald_scores(self, dirs):
        a  = self.ig[0]
        ko = self.ig[13] / 12.398
        bm = np.array(ts.bmatrix([a, a, a, 90, 90, 90]).bm())
        hkl002 = ts.PhasonDistoArray(
            np.array(self.full_reflist),
            np.array(self.full_reflist2),
            list(self.ig[14:23])
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

    def _plot_arc(self, hkl_6d, colour):
        rl1, rl2 = build_reflist_from_6d(hkl_6d.reshape(1, -1))
        self._dms_full.reflist  = np.matrix(rl1)
        self._dms_full.reflist2 = np.matrix(rl2)
        self._dms_full.sethkl(self._hkl.copy())
        try:
            self._dms_full.imcalc(self.ig)
            pts2d = self._dms_full.pxv2d_all
            if pts2d.shape[0] == 0:
                return
            x_arr = pts2d[:, 1].astype(float)
            y_arr = pts2d[:, 0].astype(float)
            arc = pg.ScatterPlotItem(
                x=x_arr, y=y_arr, size=3, pen=None, brush=pg.mkBrush(colour))
            arc._x_data = x_arr   # cached for hit-testing
            arc._y_data = y_arr
            arc._colour = pg.mkColor(colour)
            h6d = hkl_6d.copy()
            self._vb.addItem(arc)
            self._pick_items.append(arc)
            self._arc_to_6d[id(arc)] = h6d
        except Exception as e:
            print('Arc error [%s]: %s' % (' '.join('%d' % v for v in hkl_6d), e))

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
        hkllist_full = ts.pilkhlrange(
            [self.ig[0]] * 3 + [90, 90, 90], self._hkl, self.ig[13],
            self._thrange[0], self._thrange[1]).hklscan(numsteps)
        self._dms_full.hkllist = hkllist_full
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
        self._dms_full.hkllist = ts.pilkhlrange(
            [self.ig[0]] * 3 + [90, 90, 90], self._hkl, self.ig[13],
            self._thrange[0], self._thrange[1]).hklscan(numsteps)
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
        self._do_update()
        self._status.setText('Config updated')

    def _on_save_json(self):
        self._sync_ig()
        ref_6d, ref_6d_checked = [], []
        for i in range(self._arc_list.count()):
            item = self._arc_list.item(i)
            arc_item = item.data(QtCore.Qt.UserRole)
            hkl_6d = self._arc_to_6d.get(id(arc_item)) if arc_item is not None else None
            if hkl_6d is not None:
                ref_6d.append([int(v) for v in hkl_6d])
                ref_6d_checked.append(item.checkState() == QtCore.Qt.Checked)

        data = {
            'scannum':        int(scannum),
            'datapoint':      int(datapoint),
            'hkl':            self._hkl.tolist(),
            'initial_guess':  self.ig.tolist(),
            'ref_6d':         ref_6d,
            'ref_6d_checked': ref_6d_checked,
        }

        default_path = os.path.join(os.getcwd(), 'slider_state_%d_dp%d.json' % (scannum, datapoint))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save state as JSON', default_path, 'JSON files (*.json)')
        if not path:
            return
        text = json.dumps(data, indent=2)
        # Collapse inner integer arrays (ref_6d rows) onto a single line
        text = re.sub(
            r'\[\n\s+((?:-?\d+,\n\s+)*-?\d+)\n\s+\]',
            lambda m: '[' + ', '.join(
                x.strip() for x in re.split(r',\n\s*', m.group(1))) + ']',
            text)
        with open(path, 'w') as fh:
            fh.write(text + '\n')
        self._status.setText('Saved → %s' % os.path.basename(path))

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

        # Restore sliders / ig / hkl
        ig_loaded  = np.array(data['initial_guess'], dtype=float)
        # Accept a 24-element (workflow-layout) vector by dropping kcor at index 8.
        if ig_loaded.size == 24:
            ig_loaded = np.delete(ig_loaded, 8)
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

        # Clear existing arcs / picks
        self._on_clear_picks()

        # Re-plot selected reflections
        ref_6d_list  = data.get('ref_6d', [])
        checked_list = data.get('ref_6d_checked', [True] * len(ref_6d_list))

        if ref_6d_list:
            self._dms_full.hkllist = ts.pilkhlrange(
                [self.ig[0]] * 3 + [90, 90, 90], self._dms.hkl, self.ig[13],
                self._thrange[0], self._thrange[1]).hklscan(numsteps)
            palette = [pg.intColor(i, hues=10) for i in range(10)]
            for k, (hkl_6d_raw, checked) in enumerate(zip(ref_6d_list, checked_list)):
                hkl_6d  = np.array(hkl_6d_raw, dtype=int)
                n_before = len(self._pick_items)
                self._plot_arc(hkl_6d, palette[k % 10])
                if len(self._pick_items) <= n_before:
                    continue   # _plot_arc failed silently
                arc_item = self._pick_items[-1]
                self._add_arc_to_list(hkl_6d, arc_item)
                if not checked:
                    list_item = self._arc_to_list_item.get(id(arc_item))
                    if list_item is not None:
                        self._arc_list.blockSignals(True)
                        list_item.setCheckState(QtCore.Qt.Unchecked)
                        self._arc_list.blockSignals(False)
                        arc_item.setVisible(False)

        self._do_update()
        self._status.setText('Loaded → %s' % os.path.basename(path))

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
        cur_energy   = self.ig[13]
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
        self._initial_guess[13] = en_new
        self._en_scan           = en_new
        self._last_hkl[:] = np.inf

        # Sync worker
        self._worker.lattice = self._lattice
        self._worker.thrange = self._thrange

        # Rebuild DMS objects using current slider state (psi/hkl/energy unchanged)
        ig0  = self.ig.copy()
        rl   = np.matrix(self.full_reflist)
        rl2  = np.matrix(self.full_reflist2)
        mtrx = list(ig0[14:23])
        self._dms = ts.dmscalc_ico(
            rl, self._hkllist, self._hklint, 1, self._psirange, 100,
            self._hkl, detvects, self._imdata, simsigma, self._azir,
            self._psi, self._px, self._py, scatv,
            ig0[9], ig0[10], ig0[11], ig0[12], ig0[13],
            rl2, mtrx)
        self._dms_full = ts.dmscalc_ico(
            rl, self._hkllist, self._hklint, 1, self._psirange, 100,
            self._hkl, detvects, self._imdata, simsigma, self._azir,
            self._psi, self._px, self._py, scatv,
            ig0[9], ig0[10], ig0[11], ig0[12], ig0[13],
            rl2, mtrx)

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
        self._do_update()
        self._status.setText('Loaded scan %d dp=%d  E=%.4f keV' % (snum_new, dp, en_new))

    # ── Workflow export / launch ───────────────────────────────────────────────

    def _on_browse_workflow_template(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select workflow config template', CONFIGS,
            'JSON files (*.json);;All files (*)')
        if path:
            self._workflow_template = path
            self._lbl_wf_template.setText(os.path.basename(path))

    def _slider_ig_to_workflow_ig24(self, template_ig24=None):
        """Convert the slider's 23-element ig (used by dmscalc_ico: psicor/hcor/
        lcor + 3 corrections) to workflow.py's 24-element initial_guess_base
        (psicor/hcor/kcor/lcor).  Unit conventions also differ: detector distance
        is stored full/un-zoomed, energy is stored as an offset from the raw scan
        energy.  If a template is given its kcor (index 8) is preserved."""
        ig24 = np.zeros(24)
        a = self.ig[0]
        ig24[0:6] = [a, a, a, 90., 90., 90.]
        ig24[6]   = self.ig[6]                    # psicor
        ig24[7]   = self.ig[7]                    # hcor
        ig24[8]   = float(template_ig24[8]) if template_ig24 is not None else 0.0  # kcor
        ig24[9]   = self.ig[8]                    # lcor
        ig24[10]  = self.ig[9] * 2.0 / zoomval    # detdist → full, un-zoomed px
        ig24[11]  = self.ig[10]                   # dxrot
        ig24[12]  = self.ig[11]                   # dyrot
        ig24[13]  = self.ig[12]                   # dzrot
        ig24[14]  = self.ig[13] - self._en_scan   # energy offset (absolute – raw scan)
        ig24[15:24] = self.ig[14:23]              # phason a11…a33
        return ig24

    def _build_workflow_config(self):
        """Return a workflow-compatible config dict populated from the current
        slider state.  The template JSON (if set) supplies all the fixed
        experiment parameters; the scan, experiment, geometry, and crystal
        sections are overridden with live slider values."""
        self._sync_ig()

        # Load template
        tmpl_ig24 = None
        if self._workflow_template and os.path.exists(self._workflow_template):
            with open(self._workflow_template) as fh:
                cfg = json.load(fh)
            tmpl_ig24 = cfg['crystal'].get('initial_guess_base')
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

        # Collect selected reflections; fall back to manual list if none
        ref_6d = [
            [int(v) for v in hkl_6d]
            for hkl_6d in self._arc_to_6d.values()
        ]
        if not ref_6d:
            ref_6d = ref_6d_manual.tolist()

        ig24 = self._slider_ig_to_workflow_ig24(tmpl_ig24)

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

    def _on_launch_workflow(self):
        try:
            cfg = self._build_workflow_config()
            json_path = os.path.join(
                os.getcwd(), 'workflow_%d_dp%d.json' % (self._scannum, self._datapoint))
            with open(json_path, 'w') as fh:
                json.dump(cfg, fh, indent=2)
            subprocess.Popen([sys.executable, '-m', 'DMSAnalysis.workflow', json_path])
            self._status.setText(
                'Launched workflow.py — %d refs, scan %d dp=%d  [%s]' % (
                    len(cfg['crystal']['ref_6d']), self._scannum,
                    self._datapoint, os.path.basename(json_path)))
        except Exception as e:
            self._status.setText('Launch failed: %s' % str(e)[:80])
            import traceback; traceback.print_exc()

    def closeEvent(self, event):
        self._worker.stop()
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
