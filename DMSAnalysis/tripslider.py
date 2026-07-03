#!/usr/bin/env python
"""
Interactive GUI for the multiple-intersection (Renninger triple-intersection)
lattice fit — the pyqtgraph companion to ``tripfit.py``, styled to match the
slider workflow (dark theme, live sliders).

Refine a conventional lattice by watching the three Kossel lines of each
secondary-reflection triple move on the stereographic projection as you drag the
free lattice parameters; the panels update live and each shows its
triple-intersection residual.  ``Fit`` runs the optimiser in the background.

Run as a module from the repository root:

    python -m DMSAnalysis.tripslider [config.json]

With no argument it loads the shipped example
(``configs/tripfit_rhombohedral_PMN_PT_example.json``).  The engine
(``ts_quasi.tripfit`` and helpers) and the crystal-system lattice constraints
(``lattice_free_slots`` / ``expand_lattice``) are shared with ``tripfit.py`` and
the image fit, so the two workflows cannot drift.
"""

import os, sys, json, copy

os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PyQt5')

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping

from . import ts_quasi as ts

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

pg.setConfigOptions(background='#1a1a1a', foreground='#cccccc', antialias=True)

PKGDIR  = os.path.abspath(os.path.dirname(__file__))
CONFIGS = os.path.join(PKGDIR, 'configs')
DEFAULT_CONFIG = os.path.join(CONFIGS, 'tripfit_rhombohedral_PMN_PT_example.json')

# Kossel-line colours (match the r/g/b convention of tripfit.py's plotster).
LINE_PENS = ('#ff5555', '#55ff55', '#5aa0ff')
LATTICE_LABELS = ['a', 'b', 'c', 'α', 'β', 'γ']
# (half-range, format) for the lattice sliders: lengths in Å, angles in degrees.
LATTICE_SPAN = [(0.3, '%0.6f'), (0.3, '%0.6f'), (0.3, '%0.6f'),
                (1.5, '%0.6f'), (1.5, '%0.6f'), (1.5, '%0.6f')]

ALGO_METHODS = ['Powell', 'Nelder-Mead', 'COBYLA', 'TNC', 'L-BFGS-B',
                'GA', 'BHPowell', 'BHCOBYLA', 'BHNelderMead']

# Crystal-type selector entries: (display label, bravais value) — the 7
# conventional systems, mirroring the conventional subset of slider.py's
# CRYSTAL_TYPE_CHOICES so both GUIs present the same names.  tripfit is image-
# free and conventional-only (no icosahedral modes).  The tetragonal/monoclinic
# unique-axis variants are picked with the separate axis combo, not listed here.
CRYSTAL_TYPE_CHOICES = [
    ('Cubic',        'cubic'),
    ('Tetragonal',   'tetragonal'),
    ('Orthorhombic', 'orthorhombic'),
    ('Monoclinic',   'monoclinic'),
    ('Rhombohedral', 'rhombohedral'),
    ('Hexagonal',    'hexagonal'),
    ('Triclinic',    'triclinic'),
]


# ── slider widgets (self-contained, matching the slider workflow theme) ─────────
class _ValueReadout(QtWidgets.QLineEdit):
    """Value readout that looks like a label but becomes editable on double-click."""
    editRequested = QtCore.pyqtSignal()
    _READ_STYLE = 'QLineEdit { background: transparent; border: none; }'
    _EDIT_STYLE = 'QLineEdit { background: #202830; border: 1px solid #4488ff; }'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.setStyleSheet(self._READ_STYLE)
        self.setToolTip('Double-click to type an exact value')

    def mouseDoubleClickEvent(self, ev):
        if self.isReadOnly():
            self.editRequested.emit()
        else:
            super().mouseDoubleClickEvent(ev)


class FloatSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, label, val_init, val_min, val_max,
                 fmt='%0.6f', n_steps=100000, parent=None):
        super().__init__(parent)
        self._min, self._max, self._n, self._fmt = val_min, val_max, n_steps, fmt
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(4)

        lbl = QtWidgets.QLabel(label)
        lbl.setFixedWidth(34)
        lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self._sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._sl.setRange(0, n_steps)
        self._sl.setSingleStep(1)
        self._sl.setPageStep(max(1, n_steps // 100))

        self._vl = _ValueReadout()
        self._vl.setFixedWidth(92)
        f = self._vl.font(); f.setFamily('monospace'); self._vl.setFont(f)
        self._editing = False
        self._vl.editRequested.connect(self._begin_edit)
        self._vl.editingFinished.connect(self._commit_edit)

        row.addWidget(lbl)
        row.addWidget(self._sl, 1)
        row.addWidget(self._vl)

        self.setValue(val_init)
        self._sl.valueChanged.connect(self._emit)

    def _begin_edit(self):
        self._editing = True
        self._vl.setReadOnly(False)
        self._vl.setStyleSheet(self._vl._EDIT_STYLE)
        self._vl.setFocus(QtCore.Qt.MouseFocusReason)
        self._vl.selectAll()

    def _commit_edit(self):
        if not self._editing:
            return
        self._editing = False
        self._vl.setReadOnly(True)
        self._vl.setStyleSheet(self._vl._READ_STYLE)
        self._vl.deselect()
        try:
            v = float(self._vl.text().strip())
        except ValueError:
            self._vl.setText(self._fmt % self.val)
            return
        if v < self._min or v > self._max:
            half = (self._max - self._min) / 2.0
            self._min, self._max = v - half, v + half
        self.setValue(v)
        self.valueChanged.emit(v)

    def _to_int(self, v):
        return int(round((v - self._min) / (self._max - self._min) * self._n))

    def _to_float(self, i):
        return self._min + i / self._n * (self._max - self._min)

    def setRange(self, val_min, val_max):
        cur = self.val
        self._min, self._max = val_min, val_max
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


# ── residual (mirrors ts_quasi.tripfit.fit, computed from the intercepts) ───────
def residual_from_intercepts(interc, target=0.0):
    """Triple-intersection residual for the three selected pair-intercepts
    ``interc`` (3x2); matches tripfit.fit's scalar, computed without a second
    Kossel-line pass.  Returns a large value on a degenerate (all-zero) result."""
    interc = np.asarray(interc, dtype=float)
    if interc.shape != (3, 2) or not np.any(interc):
        return 500.0
    v1, v2, v3 = (np.matrix(interc[0]), np.matrix(interc[1]), np.matrix(interc[2]))
    try:
        scal = np.abs(np.linalg.norm(v1) - (v1 / np.linalg.norm(v1) * v2.T)
                      + np.linalg.norm(v2) - (v2 / np.linalg.norm(v2) * v3.T)
                      + np.linalg.norm(v1) - (v1 / np.linalg.norm(v1) * v3.T))
        return float(np.abs(np.array(scal)[0][0] - target))
    except Exception:
        return 500.0


# ── background fit worker ───────────────────────────────────────────────────────
class FitWorker(QtCore.QThread):
    """Runs the combined optimiser off the UI thread; supports a cooperative
    stop via a callback the objective checks each evaluation."""
    done = QtCore.pyqtSignal(object, float)      # (x, residual)
    progress = QtCore.pyqtSignal(float)          # current residual

    class _Stop(Exception):
        pass

    def __init__(self, objective, x0, method, bounds, tol, niter, strat, parent=None):
        super().__init__(parent)
        self._obj = objective
        self._x0 = np.atleast_1d(np.asarray(x0, dtype=float))
        self._method = method
        self._bounds = bounds
        self._tol = tol
        self._niter = niter
        self._strat = strat
        self._stop = False

    def request_stop(self):
        self._stop = True

    def _wrapped(self, x):
        if self._stop:
            raise FitWorker._Stop()
        r = self._obj(x)
        self.progress.emit(float(r))
        return r

    def run(self):
        best_x, best_f = self._x0, np.inf
        try:
            if self._method == 'GA':
                res = differential_evolution(self._wrapped, self._bounds,
                                             strategy=self._strat, polish=True)
                best_x, best_f = res.x, res.fun
            elif self._method.startswith('BH'):
                inner = self._method[2:] or 'Powell'
                res = basinhopping(self._wrapped, self._x0,
                                   minimizer_kwargs={'method': inner},
                                   niter=self._niter)
                best_x, best_f = res.x, res.fun
            elif self._method == 'TNC':
                res = minimize(self._wrapped, self._x0, method='TNC',
                               bounds=self._bounds, tol=self._tol)
                best_x, best_f = res.x, res.fun
            else:
                res = minimize(self._wrapped, self._x0, method=self._method,
                               tol=self._tol)
                best_x, best_f = res.x, res.fun
        except FitWorker._Stop:
            best_x, best_f = self._x0, self._obj(self._x0)
        except Exception as e:
            print('Fit error:', e)
            best_x, best_f = self._x0, self._obj(self._x0)
        self.done.emit(np.atleast_1d(np.asarray(best_x, dtype=float)), float(best_f))


# ── main window ─────────────────────────────────────────────────────────────────
class TripSlider(QtWidgets.QMainWindow):
    def __init__(self, cfg, cfg_path):
        super().__init__()
        self._cfg_path = cfg_path
        self._fit_worker = None
        self._load_state(cfg)

        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(60)
        self._update_timer.timeout.connect(self._redraw)

        self._build_ui()
        self._rebuild_engines()
        self._redraw()

    # ── config → state ──────────────────────────────────────────────────────────
    def _load_state(self, cfg):
        self._cfg = cfg
        geo = cfg['geometry']
        self._hkl = np.array([geo['hkl']], dtype=float)
        self._azir = list(geo['azir'])

        comp = cfg['computation']
        self._bravais = comp['bravais']
        if self._bravais not in ts.CONVENTIONAL_SYSTEMS:
            raise SystemExit('computation.bravais must be a conventional system, '
                             'got %r' % self._bravais)
        # Pseudo-cubic re-indexing (Table 1 of doi:10.1107/S1600576723004120),
        # mirroring slider.py / fit.py: computation.pseudocubic_transform (1-12,
        # 1 = identity) selects one of the 12 equivalent indexing matrices,
        # applied as hkl' = M @ hkl to the primary reflection, the azimuthal
        # reference and every triple's reflection list.  The stored config values
        # are the base indexing; the matrix is applied at load (see below).
        self._pc_idx = int(comp.get('pseudocubic_transform', 1))
        if not 1 <= self._pc_idx <= len(ts.PSEUDOCUBIC_TRANSFORMS):
            self._pc_idx = 1
        self._live_res = int(comp.get('live_resolution', 200))
        self._fit_res = int(comp.get('resolution', 1000))
        self._method = comp.get('opt_method', 'Powell')
        self._tol = float(comp.get('tolerance', 1e-12))
        self._boundrange = comp.get('boundrange', [-0.012, 0.012])
        self._niter = int(comp.get('bh_niter', 50))
        self._strat = ts.DE_Strategy.get(comp.get('de_strategy', 'best1bin'),
                                         'best1bin')
        self._six = np.array(cfg['crystal']['initial_guess'], dtype=float)
        self._groups_cfg = cfg['intersections']
        if not self._groups_cfg:
            raise SystemExit('config "intersections" must list at least one triple')
        # apply the configured pseudo-cubic matrix to the base indexing at load
        if self._pc_idx != 1:
            self._reindex(ts.pseudocubic_matrix(self._pc_idx))

    def _free_slots(self):
        return ts.lattice_free_slots(self._bravais)

    def _reduced(self):
        return np.array([self._six[s] for s in self._free_slots()], dtype=float)

    def _set_reduced(self, x):
        for s, v in zip(self._free_slots(), np.atleast_1d(x)):
            self._six[s] = float(v)

    # ── engines ───────────────────────────────────────────────────────────────
    def _rebuild_engines(self):
        self._groups = []
        for gi, gc in enumerate(self._groups_cfg):
            reflist = np.matrix(np.array(gc['reflist'], dtype=float))
            tf = ts.tripfit(self._hkl, reflist, self._azir,
                            self._live_res, self._bravais, float(gc['energy']),
                            float(gc.get('target', 0.0)))
            self._groups.append({
                'label': gc.get('label', 'T%d' % (gi + 1)),
                'reflist': reflist, 'target': float(gc.get('target', 0.0)),
                'tf': tf})

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle('TripFit — multiple-intersection GUI')
        # editor edits are debounced so typing doesn't rebuild on every keystroke
        self._table_timer = QtCore.QTimer(self)
        self._table_timer.setSingleShot(True)
        self._table_timer.setInterval(350)
        self._table_timer.timeout.connect(self._apply_table)
        outer = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.setCentralWidget(outer)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(splitter)

        # left control column
        ctrl = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(ctrl)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(6)
        ctrl.setMinimumWidth(320)
        ctrl.setMaximumWidth(430)

        # crystal type + unique axis + pseudo-cubic re-indexing (mirrors the
        # slider.py workflow: display names, and the tetragonal/monoclinic
        # unique-axis variants chosen from a separate axis combo).
        ct_box = QtWidgets.QGroupBox('Crystal type')
        ct_v = QtWidgets.QVBoxLayout(ct_box)
        ct_v.setContentsMargins(4, 2, 4, 2)
        ct_v.setSpacing(2)
        ct_l = QtWidgets.QHBoxLayout()
        self._crystal_combo = QtWidgets.QComboBox()
        for _disp, _name in CRYSTAL_TYPE_CHOICES:
            self._crystal_combo.addItem(_disp, _name)
        self._crystal_combo.setToolTip('Crystal system — sets which lattice '
                                       'parameters are free (constrained by '
                                       'symmetry via expand_lattice).')
        self._axis_combo = QtWidgets.QComboBox()
        self._axis_combo.setToolTip('Tetragonal: unique axis · Monoclinic: '
                                    'non-90° angle')
        self._apply_crystal_combos(self._bravais)   # set both combos from config
        self._crystal_combo.currentIndexChanged.connect(self._on_crystal_changed)
        self._axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        ct_l.addWidget(self._crystal_combo, 1)
        ct_l.addWidget(self._axis_combo, 1)
        ct_v.addLayout(ct_l)

        pc_row = QtWidgets.QHBoxLayout()
        pc_lbl = QtWidgets.QLabel('Pseudo-cubic M')
        pc_lbl.setToolTip(
            'Pseudo-cubic re-indexing matrix (Table 1 of Nisbet et al. (2023), '
            'J. Appl. Cryst. 56, 1046-1050).  Re-indexes the primary hkl, the '
            'azimuthal reference and every triple’s reflection list as M · hkl '
            'to test the equivalent indexing choices of a pseudo-cubic crystal '
            '(1 = identity).  The lattice parameters are left untouched.')
        self._pc_combo = QtWidgets.QComboBox()
        for _i in range(1, len(ts.PSEUDOCUBIC_TRANSFORMS) + 1):
            self._pc_combo.addItem('%2d  %s' % (_i, ts.pseudocubic_label(_i)), _i)
        _j = self._pc_combo.findData(self._pc_idx)
        self._pc_combo.setCurrentIndex(_j if _j >= 0 else 0)
        self._pc_combo.setToolTip(pc_lbl.toolTip())
        self._pc_combo.currentIndexChanged.connect(self._on_pc_transform_changed)
        pc_row.addWidget(pc_lbl)
        pc_row.addWidget(self._pc_combo, 1)
        ct_v.addLayout(pc_row)
        col.addWidget(ct_box)

        # lattice sliders
        self._slider_box = QtWidgets.QGroupBox('Lattice / geometry')
        self._slider_vbox = QtWidgets.QVBoxLayout(self._slider_box)
        self._slider_vbox.setSpacing(1)
        self._slider_vbox.setContentsMargins(4, 4, 4, 4)
        col.addWidget(self._slider_box)
        self._populate_sliders()

        # fit controls
        fit_box = QtWidgets.QGroupBox('Fit')
        fg = QtWidgets.QGridLayout(fit_box)
        fg.setContentsMargins(4, 4, 4, 4)
        fg.setSpacing(4)
        fg.addWidget(QtWidgets.QLabel('Method'), 0, 0)
        self._algo_combo = QtWidgets.QComboBox()
        for m in ALGO_METHODS:
            self._algo_combo.addItem(m)
        self._algo_combo.setCurrentIndex(max(0, ALGO_METHODS.index(self._method)
                                             if self._method in ALGO_METHODS else 0))
        fg.addWidget(self._algo_combo, 0, 1, 1, 3)

        fg.addWidget(QtWidgets.QLabel('Live res'), 1, 0)
        self._sp_live = QtWidgets.QSpinBox()
        self._sp_live.setRange(50, 4000); self._sp_live.setSingleStep(50)
        self._sp_live.setValue(self._live_res)
        self._sp_live.setToolTip('Kossel-line sampling used for the live overlay '
                                 '(lower = snappier)')
        self._sp_live.valueChanged.connect(self._on_live_res_changed)
        fg.addWidget(self._sp_live, 1, 1)
        fg.addWidget(QtWidgets.QLabel('Fit res'), 1, 2)
        self._sp_fit = QtWidgets.QSpinBox()
        self._sp_fit.setRange(50, 8000); self._sp_fit.setSingleStep(50)
        self._sp_fit.setValue(self._fit_res)
        self._sp_fit.setToolTip('Kossel-line sampling used during the fit '
                                '(higher = more accurate, slower)')
        fg.addWidget(self._sp_fit, 1, 3)

        self._btn_fit = QtWidgets.QPushButton('Fit')
        self._btn_fit.setStyleSheet('background: #1a5c1a; color: #ccffcc; font-weight: bold')
        self._btn_fit.clicked.connect(self._on_fit)
        fg.addWidget(self._btn_fit, 2, 0, 1, 2)
        self._btn_stop = QtWidgets.QPushButton('Stop')
        self._btn_stop.setStyleSheet('background: #5c1a1a; color: #ffcccc; font-weight: bold')
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        fg.addWidget(self._btn_stop, 2, 2, 1, 2)
        col.addWidget(fit_box)

        # config load/save
        cfg_row = QtWidgets.QHBoxLayout()
        btn_load = QtWidgets.QPushButton('Load config')
        btn_load.setStyleSheet('background: #201030; color: #ddccff')
        btn_load.clicked.connect(self._on_load_config)
        btn_save = QtWidgets.QPushButton('Save config')
        btn_save.setStyleSheet('background: #103010; color: #ccffcc')
        btn_save.clicked.connect(self._on_save_config)
        cfg_row.addWidget(btn_load)
        cfg_row.addWidget(btn_save)
        col.addLayout(cfg_row)

        self._total_lbl = QtWidgets.QLabel('Σ residual: —')
        f = self._total_lbl.font(); f.setPointSize(10); f.setBold(True)
        self._total_lbl.setFont(f)
        col.addWidget(self._total_lbl)
        col.addStretch(1)

        splitter.addWidget(ctrl)

        # right: stereographic panels
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setMinimumWidth(520)
        splitter.addWidget(self._gw)
        splitter.setStretchFactor(1, 1)
        self._build_panels()

        # intersection-group editor spans the full width below the panels
        outer.addWidget(self._build_intersections_editor())
        outer.setStretchFactor(0, 1)

        self._status = self.statusBar()
        self._status.showMessage('Ready — %s, %d groups'
                                 % (self._bravais, len(self._groups_cfg)))
        self.resize(1150, 780)

    def _build_panels(self):
        """One stereographic plot per intersection group, with three Kossel-line
        curves and the three intercept markers."""
        self._gw.clear()
        self._panels = []
        ncols = min(max(1, len(self._groups_cfg)), 4)   # wrap wide group counts
        for gi, gc in enumerate(self._groups_cfg):
            p = self._gw.addPlot(row=gi // ncols, col=gi % ncols)
            p.setAspectLocked(True)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setMenuEnabled(False)
            curves = [p.plot([], [], pen=pg.mkPen(LINE_PENS[k], width=1.5))
                      for k in range(3)]
            pts = pg.ScatterPlotItem(size=9, pen=pg.mkPen('#ffffff', width=1),
                                     brush=pg.mkBrush('#ffcc33'))
            p.addItem(pts)
            self._panels.append({'plot': p, 'curves': curves, 'pts': pts})

    def _populate_sliders(self):
        while self._slider_vbox.count():
            it = self._slider_vbox.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None); w.deleteLater()
        self._lat_sliders = {}
        for slot in self._free_slots():
            half, fmt = LATTICE_SPAN[slot]
            v = float(self._six[slot])
            fs = FloatSlider(LATTICE_LABELS[slot], v, v - half, v + half, fmt)
            fs.valueChanged.connect(self._on_slider(slot))
            self._slider_vbox.addWidget(fs)
            self._lat_sliders[slot] = fs
        # primary hkl — sets the pole the frame is rotated onto, i.e. the origin
        # of the stereographic projection.  Configuration only: these are NOT in
        # the fit's free-parameter vector (``_reduced`` = lattice free slots).
        self._hkl_sliders = []
        _lim = max(6.0, float(np.max(np.abs(self._hkl))) + 3.0)
        for i, lab in enumerate(('h', 'k', 'l')):
            fs = FloatSlider(lab, float(self._hkl[0, i]), -_lim, _lim, '%0.4f')
            fs.valueChanged.connect(self._on_hkl(i))
            fs.setToolTip('Primary reflection index — moves the stereographic '
                          'origin. Configuration only; not refined by Fit.')
            self._slider_vbox.addWidget(fs)
            self._hkl_sliders.append(fs)

    # ── intersection-group editor ────────────────────────────────────────────────
    _COLS = ['Label', 'Refl 1 (h k l)', 'Refl 2 (h k l)', 'Refl 3 (h k l)',
             'Energy', 'Target']

    def _build_intersections_editor(self):
        box = QtWidgets.QGroupBox('Triple intersections')
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        self._table = QtWidgets.QTableWidget(0, len(self._COLS))
        self._table.setHorizontalHeaderLabels(self._COLS)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setToolTip(
            'Each row is one triple intersection: three secondary reflections '
            'whose Kossel lines should meet.  Reflections are space-separated '
            'integers; energy is keV.  The tightest (closest) crossing triple is '
            'scored automatically.  Edits update the panels live.')
        f = self._table.font(); f.setPointSize(8); self._table.setFont(f)
        self._table.itemChanged.connect(lambda _i: self._table_timer.start())
        self._populate_table()
        v.addWidget(self._table)

        row = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton('Add triple')
        btn_add.setStyleSheet('background: #103018; color: #bfe6c8')
        btn_add.setToolTip('Append a new triple-intersection group')
        btn_add.clicked.connect(self._on_add_group)
        btn_dup = QtWidgets.QPushButton('Duplicate')
        btn_dup.setStyleSheet('background: #102030; color: #aaccff')
        btn_dup.setToolTip('Duplicate the selected group')
        btn_dup.clicked.connect(self._on_duplicate_group)
        btn_del = QtWidgets.QPushButton('Remove')
        btn_del.setStyleSheet('background: #3a1a1a; color: #ffcccc')
        btn_del.setToolTip('Remove the selected group(s)')
        btn_del.clicked.connect(self._on_remove_group)
        row.addWidget(btn_add)
        row.addWidget(btn_dup)
        row.addWidget(btn_del)
        row.addStretch(1)
        v.addLayout(row)
        box.setMaximumHeight(220)
        return box

    @staticmethod
    def _fmt_vec(vec):
        return ' '.join('%g' % v for v in np.asarray(vec).ravel())

    @staticmethod
    def _parse_vec(text, n=3):
        parts = [p for p in text.replace(',', ' ').split() if p]
        if len(parts) != n:
            raise ValueError('expected %d numbers, got %r' % (n, text))
        return [float(p) for p in parts]

    def _populate_table(self):
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._groups_cfg))
        for r, gc in enumerate(self._groups_cfg):
            refl = np.array(gc['reflist'], dtype=float)
            cells = [gc.get('label', 'T%d' % (r + 1)),
                     self._fmt_vec(refl[0]), self._fmt_vec(refl[1]),
                     self._fmt_vec(refl[2]),
                     '%g' % float(gc['energy']),
                     '%g' % float(gc.get('target', 0.0))]
            for c, txt in enumerate(cells):
                self._table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
        self._table.blockSignals(False)

    def _read_table(self):
        """Parse the editor into a list of group-config dicts.  Returns
        (groups, error_string); error_string is '' on success."""
        groups = []
        for r in range(self._table.rowCount()):
            def cell(c):
                it = self._table.item(r, c)
                return it.text().strip() if it is not None else ''
            try:
                reflist = [self._parse_vec(cell(1)), self._parse_vec(cell(2)),
                           self._parse_vec(cell(3))]
                energy = float(cell(4))
                target = float(cell(5) or 0.0)
            except ValueError as e:
                return None, 'row %d: %s' % (r + 1, e)
            groups.append({'label': cell(0) or 'T%d' % (r + 1),
                           'reflist': reflist, 'energy': energy,
                           'target': target})
        if not groups:
            return None, 'need at least one triple intersection'
        return groups, ''

    def _apply_table(self):
        groups, err = self._read_table()
        if err:
            self._status.showMessage('Intersection editor — %s' % err)
            return
        self._groups_cfg = groups
        self._cfg['intersections'] = groups
        self._rebuild_engines()
        self._build_panels()
        self._redraw()
        self._status.showMessage('Applied %d triple intersection(s)' % len(groups))

    def _on_add_group(self):
        template = (copy.deepcopy(self._groups_cfg[-1]) if self._groups_cfg else
                    {'reflist': [[0, 0, 2], [2, 0, 0], [2, 0, 2]],
                     'energy': self._groups_cfg[0]['energy']
                     if self._groups_cfg else 8.0,
                     'target': 0.0})
        template['label'] = 'T%d' % (self._table.rowCount() + 1)
        self._groups_cfg = list(self._groups_cfg) + [template]
        self._populate_table()
        self._apply_table()

    def _on_duplicate_group(self):
        rows = sorted({i.row() for i in self._table.selectedIndexes()})
        if not rows:
            self._status.showMessage('Select a group to duplicate')
            return
        groups, err = self._read_table()
        if err:
            self._status.showMessage('Intersection editor — %s' % err)
            return
        for r in reversed(rows):
            dup = copy.deepcopy(groups[r])
            dup['label'] = dup.get('label', 'T') + '*'
            groups.insert(r + 1, dup)
        self._groups_cfg = groups
        self._populate_table()
        self._apply_table()

    def _on_remove_group(self):
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            self._status.showMessage('Select a group to remove')
            return
        if self._table.rowCount() - len(rows) < 1:
            self._status.showMessage('Cannot remove the last triple intersection')
            return
        groups, err = self._read_table()
        if err:              # fall back to current config if a cell is mid-edit
            groups = list(self._groups_cfg)
        for r in rows:
            if 0 <= r < len(groups):
                del groups[r]
        self._groups_cfg = groups
        self._populate_table()
        self._apply_table()

    # ── interactions ────────────────────────────────────────────────────────────
    def _on_slider(self, slot):
        def handler(v):
            self._six[slot] = float(v)
            # keep symmetry-linked lattice slots visually in step
            self._six = np.array(ts.expand_lattice(self._bravais, self._six),
                                 dtype=float)
            self._update_timer.start()
        return handler

    def _on_hkl(self, i):
        def handler(v):
            self._hkl[0, i] = float(v)
            # hkl is the primary reflection stored live on each engine; moving it
            # re-poles the projection.
            for g in self._groups:
                g['tf'].hkl = self._hkl
            self._update_timer.start()
        return handler

    def _on_live_res_changed(self, v):
        self._live_res = int(v)
        for g in self._groups:
            g['tf'].resolution = self._live_res
        self._update_timer.start()

    # ── crystal type + unique axis (mirrors slider.py's base/axis split) ────────
    @staticmethod
    def _split_bravais(b):
        """Split a bravais value into (base, axis-suffix), e.g. 'tetragonal_a' ->
        ('tetragonal', '_a').  Only tetragonal/monoclinic have variants."""
        for base in ('tetragonal', 'monoclinic'):
            if b == base:
                return base, ''
            if b.startswith(base + '_'):
                return base, b[len(base):]
        return b, ''

    @staticmethod
    def _axis_options(base):
        """(label, suffix) options for the unique-axis combo, per base system."""
        if base == 'tetragonal':
            return [('unique c', ''), ('unique a', '_a'), ('unique b', '_b')]
        if base == 'monoclinic':
            return [('β ≠ 90', ''), ('α ≠ 90', '_a'), ('γ ≠ 90', '_c')]
        return []

    def _refresh_axis_combo(self, base, suffix=''):
        """Repopulate the unique-axis combo for ``base``, selecting ``suffix``;
        hidden when the base system has no axis choice."""
        self._axis_combo.blockSignals(True)
        self._axis_combo.clear()
        opts = self._axis_options(base)
        if opts:
            for _disp, _suf in opts:
                self._axis_combo.addItem(_disp, _suf)
            j = self._axis_combo.findData(suffix)
            self._axis_combo.setCurrentIndex(j if j >= 0 else 0)
        self._axis_combo.setVisible(bool(opts))
        self._axis_combo.blockSignals(False)

    def _apply_crystal_combos(self, b):
        """Set the crystal-type and unique-axis combos to reflect bravais ``b``
        (no signals emitted)."""
        base, suffix = self._split_bravais(b)
        self._crystal_combo.blockSignals(True)
        i = self._crystal_combo.findData(base)
        if i < 0:   # a system not in the list — keep it, don't switch
            self._crystal_combo.insertItem(0, base, base)
            i = 0
        self._crystal_combo.setCurrentIndex(i)
        self._crystal_combo.blockSignals(False)
        self._refresh_axis_combo(base, suffix)

    def _effective_bravais(self):
        """Bravais value combining the crystal-type and unique-axis combos."""
        base = self._crystal_combo.currentData()
        if base is None:
            return None
        suffix = self._axis_combo.currentData() if self._axis_combo.count() else ''
        return base + (suffix or '')

    def _switch_crystal(self, name):
        if not name or name == self._bravais:
            return
        self._bravais = name
        self._six = np.array(ts.expand_lattice(name, self._six), dtype=float)
        self._populate_sliders()
        self._rebuild_engines()
        self._redraw()
        self._status.showMessage('Crystal type: %s' % name)

    def _on_crystal_changed(self, _idx=None):
        base = self._crystal_combo.currentData()
        if base is None:
            return
        self._refresh_axis_combo(base)   # default axis for the new base system
        eff = self._effective_bravais()
        if eff:
            self._switch_crystal(eff)

    def _on_axis_changed(self, _idx=None):
        eff = self._effective_bravais()
        if eff:
            self._switch_crystal(eff)

    # ── pseudo-cubic re-indexing (Table 1, Nisbet et al. 2023) ──────────────────
    def _reindex(self, R):
        '''Re-index the primary hkl, azimuthal reference and every triple's
        reflection list in place by the integer matrix ``R`` (hkl' = R·hkl; for
        the stored row vectors, ``v' = v @ R.T``).  ``R`` is an orthogonal cubic
        rotation, so integer indices stay integer.'''
        R = np.asarray(R, dtype=float)
        self._hkl = np.round(self._hkl @ R.T)
        self._azir = [float(v) for v in np.round(R @ np.asarray(self._azir, float))]
        for gc in self._groups_cfg:
            refl = np.asarray(gc['reflist'], dtype=float)
            gc['reflist'] = np.round(refl @ R.T).tolist()

    def _on_pc_transform_changed(self, _idx=None):
        new = self._pc_combo.currentData()
        if new is None or new == self._pc_idx:
            return
        # flush any pending (debounced) table edits so we re-index what is shown
        groups, err = self._read_table()
        if not err:
            self._groups_cfg = groups
        # matrices are orthogonal, so the relative re-indexing from the current
        # setting is R = M_new · M_oldᵀ (same delta rule as slider.py)
        R = ts.pseudocubic_matrix(new) @ ts.pseudocubic_matrix(self._pc_idx).T
        self._reindex(R)
        self._pc_idx = new
        self._cfg['intersections'] = self._groups_cfg
        self._cfg.setdefault('computation', {})['pseudocubic_transform'] = new
        self._populate_table()
        self._rebuild_engines()
        self._redraw()
        self._status.showMessage('Pseudo-cubic transform %d applied: %s'
                                 % (new, ts.pseudocubic_label(new)))

    def _on_load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load tripfit config', os.getcwd(), 'JSON files (*.json)')
        if not path:
            return
        try:
            with open(path) as fh:
                cfg = json.load(fh)
            self._cfg_path = path
            self._load_state(cfg)
        except Exception as e:
            self._status.showMessage('Load failed: %s' % e)
            return
        # rebuild everything for the new config
        self._apply_crystal_combos(self._bravais)
        j = self._pc_combo.findData(self._pc_idx)
        self._pc_combo.blockSignals(True)
        self._pc_combo.setCurrentIndex(j if j >= 0 else 0)
        self._pc_combo.blockSignals(False)
        self._sp_live.setValue(self._live_res)
        self._sp_fit.setValue(self._fit_res)
        self._populate_sliders()
        self._populate_table()
        self._rebuild_engines()
        self._build_panels()
        self._redraw()
        self._status.showMessage('Loaded %s' % os.path.basename(path))

    def _on_save_config(self):
        cfg = self._export_config()
        default = os.path.join(os.getcwd(), 'tripfit_config.json')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save tripfit config', default, 'JSON files (*.json)')
        if not path:
            return
        with open(path, 'w') as fh:
            json.dump(cfg, fh, indent=2)
        self._status.showMessage('Saved %s' % os.path.basename(path))

    def _export_config(self):
        # capture any pending (un-debounced) edits from the intersection editor
        groups, err = self._read_table()
        if not err:
            self._groups_cfg = groups
        cfg = copy.deepcopy(self._cfg)
        cfg['intersections'] = copy.deepcopy(self._groups_cfg)
        cfg.setdefault('geometry', {})['hkl'] = [float(v) for v in self._hkl.ravel()]
        cfg['geometry']['azir'] = list(self._azir)
        cfg.setdefault('computation', {})
        cfg['computation']['bravais'] = self._bravais
        # hkl/azir/reflists above are already in the active pseudo-cubic
        # indexing, so reset the transform to 1 — the consumer must not re-apply
        # the matrix (mirrors slider.py's export).
        cfg['computation']['pseudocubic_transform'] = 1
        cfg['computation']['opt_method'] = self._algo_combo.currentText()
        cfg['computation']['resolution'] = int(self._sp_fit.value())
        cfg['computation']['live_resolution'] = int(self._sp_live.value())
        cfg['computation']['tolerance'] = self._tol
        cfg['computation']['boundrange'] = list(self._boundrange)
        cfg.setdefault('crystal', {})['initial_guess'] = [float(v) for v in self._six]
        return cfg

    # ── live redraw ───────────────────────────────────────────────────────────
    def _redraw(self):
        # keep lattice slider readouts in step with symmetry-linked values
        for slot, fs in self._lat_sliders.items():
            if abs(fs.val - self._six[slot]) > 1e-9:
                fs.blockSignals(True); fs.setValue(float(self._six[slot]))
                fs.blockSignals(False)
        # keep h/k/l readouts in step with programmatic hkl changes (re-index, load)
        for i, fs in enumerate(getattr(self, '_hkl_sliders', [])):
            if abs(fs.val - self._hkl[0, i]) > 1e-9:
                fs.blockSignals(True); fs.setValue(float(self._hkl[0, i]))
                fs.blockSignals(False)
        reduced = self._reduced()
        total = 0.0
        for g, panel in zip(self._groups, self._panels):
            tf = g['tf']
            try:
                interc, st0, st1, st2, _v0, _v1, _v2 = tf.full(reduced)
                sts = [np.asarray(st0), np.asarray(st1), np.asarray(st2)]
                for k, cv in enumerate(panel['curves']):
                    cv.setData(sts[k][:, 0], sts[k][:, 1])
                interc = np.asarray(interc)
                panel['pts'].setData(interc[:, 0], interc[:, 1])
                r = residual_from_intercepts(interc, g['target'])
            except Exception:
                for cv in panel['curves']:
                    cv.setData([], [])
                panel['pts'].setData([], [])
                r = 500.0
            total += r
            panel['plot'].setTitle('%s  ·  res=%.2e'
                                   % (g['label'], r), size='8pt')
        self._total_lbl.setText('Σ residual: %.4e' % total)

    # ── fitting ─────────────────────────────────────────────────────────────────
    def _fit_objective(self, x):
        """Combined residual across groups at the FIT resolution."""
        s = 0.0
        for g in self._groups:
            g['tf'].resolution = self._fit_res
            s += g['tf'].fit(x)
        return s

    def _on_fit(self):
        if self._fit_worker is not None:
            return
        method = self._algo_combo.currentText()
        self._fit_res = int(self._sp_fit.value())
        x0 = self._reduced()
        lb = x0 + self._boundrange[0]
        ub = x0 + self._boundrange[1]
        bounds = list(zip(lb, ub))
        self._btn_fit.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status.showMessage('Fitting with %s (fit res %d)…'
                                 % (method, self._fit_res))
        self._fit_worker = FitWorker(self._fit_objective, x0, method, bounds,
                                     self._tol, self._niter, self._strat)
        self._fit_worker.progress.connect(
            lambda r: self._status.showMessage('Fitting… current Σ res=%.4e' % r))
        self._fit_worker.done.connect(self._on_fit_done)
        self._fit_worker.start()

    def _on_stop(self):
        if self._fit_worker is not None:
            self._fit_worker.request_stop()
            self._status.showMessage('Stopping…')

    def _on_fit_done(self, x, residual):
        self._fit_worker = None
        self._btn_fit.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._set_reduced(x)
        self._six = np.array(ts.expand_lattice(self._bravais, self._six),
                             dtype=float)
        # restore live resolution for the overlay
        for g in self._groups:
            g['tf'].resolution = self._live_res
        self._redraw()
        lat = np.array2string(np.array(ts.expand_lattice(self._bravais, self._six)),
                              precision=6)
        self._status.showMessage('Fit done — Σ res=%.4e  lattice=%s'
                                 % (residual, lat))

    def closeEvent(self, ev):
        if self._fit_worker is not None:
            self._fit_worker.request_stop()
            self._fit_worker.wait(2000)
        super().closeEvent(ev)


# ── launch ──────────────────────────────────────────────────────────────────────
def _dark_palette():
    p = QtGui.QPalette()
    dark, mid, light = (QtGui.QColor(26, 26, 26), QtGui.QColor(42, 42, 42),
                        QtGui.QColor(58, 58, 58))
    text, hilit = QtGui.QColor(210, 210, 210), QtGui.QColor(42, 130, 218)
    p.setColor(QtGui.QPalette.Window, dark)
    p.setColor(QtGui.QPalette.WindowText, text)
    p.setColor(QtGui.QPalette.Base, mid)
    p.setColor(QtGui.QPalette.AlternateBase, light)
    p.setColor(QtGui.QPalette.Text, text)
    p.setColor(QtGui.QPalette.Button, light)
    p.setColor(QtGui.QPalette.ButtonText, text)
    p.setColor(QtGui.QPalette.ToolTipBase, mid)
    p.setColor(QtGui.QPalette.ToolTipText, text)
    p.setColor(QtGui.QPalette.Highlight, hilit)
    p.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
    p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(100, 100, 100))
    p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(100, 100, 100))
    return p


def main():
    cfg_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setPalette(_dark_palette())
    win = TripSlider(cfg, cfg_path)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
