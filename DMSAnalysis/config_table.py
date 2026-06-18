#!/usr/bin/env python
"""
config_table — a shared, editable Qt table view of a DMS config dict.

Both the slider (``slider_quasi_*.py``) and the fitter (``workflow.py``) embed
this widget so the experiment metadata and key analysis scalars are visible and
editable in the GUI, then saved back to JSON.  Large arrays (``ref_6d``,
``initial_guess_base``, ``manual_centres``) are intentionally excluded — those
keep their dedicated UIs (sliders, arc list).

API
---
    t = ConfigTable()
    t.set_config(cfg)          # populate from a config dict (stored as a copy)
    cfg2 = t.to_config()       # config dict with the user's edits merged in
    t.configChanged.connect(...)   # emitted (with the merged dict) on any edit

A "Save config…" button is built in; it writes ``to_config()`` to a JSON file
chosen via a dialog (default path supplied with :meth:`set_save_path`).
"""

import copy
import json

from PyQt5 import QtWidgets, QtCore


# (path, label, type)  — type ∈ {'int','float','str','bool','intlist','floatlist'}
# Paths support dotted keys and ``[i]`` array indices.  Rows whose path is absent
# from the config are silently skipped, so the same spec works for any config.
DEFAULT_FIELDS = [
    ('scan.scannum',                 'scan / scannum',          'int'),
    ('scan.datapoint',               'scan / datapoint',        'int'),
    ('scan.datapoint0',              'scan / datapoint0',       'int'),
    ('experiment.lattice[0]',        'lattice a (Å)',           'float'),
    ('experiment.lattice[1]',        'lattice b (Å)',           'float'),
    ('experiment.lattice[2]',        'lattice c (Å)',           'float'),
    ('experiment.lattice[3]',        'lattice α (°)',           'float'),
    ('experiment.lattice[4]',        'lattice β (°)',           'float'),
    ('experiment.lattice[5]',        'lattice γ (°)',           'float'),
    ('experiment.energy',            'energy (keV)',            'float'),
    ('experiment.energy0',           'energy0 (keV)',           'float'),
    ('experiment.azir[0]',           'azir h',                  'float'),
    ('experiment.azir[1]',           'azir k',                  'float'),
    ('experiment.azir[2]',           'azir l',                  'float'),
    ('experiment.image_template',    'image template',          'str'),
    ('geometry.psi',                 'geometry / psi',          'float'),
    ('geometry.px_unscaled',         'geometry / px_unscaled',  'float'),
    ('geometry.py_unscaled',         'geometry / py_unscaled',  'float'),
    ('geometry.scatv',               'geometry / scatv',        'int'),
    ('display.zoomval',              'display / zoomval',       'int'),
    ('display.colourlim',            'display / colourlim',     'intlist'),
    ('computation.numsteps',         'computation / numsteps',  'int'),
    ('computation.simsigma_per_zoom','computation / simsigma',  'float'),
    ('computation.thrange_delta',    'computation / thrange',   'floatlist'),
    ('flags.save',                   'flags / save',            'bool'),
    ('flags.fit',                    'flags / fit',             'bool'),
    ('flags.firstplot',              'flags / firstplot',       'bool'),
    ('flags.detoptimize',            'flags / detoptimize',     'bool'),
    ('flags.energyopt',              'flags / energyopt',       'bool'),
    ('flags.autoreflist',            'flags / autoreflist',     'bool'),
]


def _split_path(path):
    """'a.b[2].c' -> [('a',None),('b',2),('c',None)]"""
    parts = []
    for tok in path.split('.'):
        if tok.endswith(']') and '[' in tok:
            name, idx = tok[:-1].split('[')
            parts.append((name, int(idx)))
        else:
            parts.append((tok, None))
    return parts


def _resolve(cfg, parts, create=False):
    """Walk to the container holding the final leaf. Returns (container, key)
    where key is a dict key or a list index, or (None, None) if not present."""
    cur = cfg
    for name, idx in parts[:-1]:
        if not isinstance(cur, dict) or name not in cur:
            if not create:
                return None, None
            cur[name] = {}
        cur = cur[name]
        if idx is not None:
            if not isinstance(cur, list) or idx >= len(cur):
                return None, None
            cur = cur[idx]
    name, idx = parts[-1]
    if not isinstance(cur, dict) or name not in cur:
        return None, None
    if idx is None:
        return cur, name
    lst = cur[name]
    if not isinstance(lst, list) or idx >= len(lst):
        return None, None
    return lst, idx


def cfg_get(cfg, path):
    parts = _split_path(path)
    container, key = _resolve(cfg, parts)
    if container is None:
        return None, False
    return container[key], True


def cfg_set(cfg, path, value):
    parts = _split_path(path)
    container, key = _resolve(cfg, parts)
    if container is None:
        return False
    container[key] = value
    return True


def _parse(text, typ):
    text = text.strip()
    if typ == 'int':
        return int(round(float(text)))
    if typ == 'float':
        return float(text)
    if typ == 'bool':
        return text.lower() in ('1', 'true', 'yes', 'on')
    if typ in ('intlist', 'floatlist'):
        conv = (lambda x: int(round(float(x)))) if typ == 'intlist' else float
        return [conv(x) for x in text.replace('[', '').replace(']', '').split(',') if x.strip() != '']
    return text


def _format(value, typ):
    if typ == 'bool':
        return 'true' if value else 'false'
    if typ in ('intlist', 'floatlist'):
        return ', '.join(str(v) for v in value)
    if typ == 'float':
        return repr(float(value))
    return str(value)


class ConfigTable(QtWidgets.QWidget):
    """Editable two-column (Field / Value) view of a config dict."""

    configChanged = QtCore.pyqtSignal(dict)

    def __init__(self, fields=None, parent=None):
        super().__init__(parent)
        self._fields = fields if fields is not None else DEFAULT_FIELDS
        self._cfg = {}
        self._rows = []          # parallel to table rows: (path, typ)
        self._save_path = ''
        self._suppress = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QtWidgets.QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(['Field', 'Value'])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table)

        btn_save = QtWidgets.QPushButton('Save config…')
        btn_save.clicked.connect(self._on_save)
        layout.addWidget(btn_save)

    # ── public API ──────────────────────────────────────────────────────────
    def set_config(self, cfg):
        self._cfg = copy.deepcopy(cfg)
        self._rebuild()

    def to_config(self):
        return copy.deepcopy(self._cfg)

    def set_save_path(self, path):
        self._save_path = path or ''

    # ── internals ───────────────────────────────────────────────────────────
    def _rebuild(self):
        self._suppress = True
        self.table.setRowCount(0)
        self._rows = []
        for path, label, typ in self._fields:
            value, present = cfg_get(self._cfg, path)
            if not present:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)

            name_item = QtWidgets.QTableWidgetItem(label)
            name_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.table.setItem(row, 0, name_item)

            val_item = QtWidgets.QTableWidgetItem(_format(value, typ))
            val_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
                              | QtCore.Qt.ItemIsSelectable)
            self.table.setItem(row, 1, val_item)

            self._rows.append((path, typ))
        self._suppress = False

    def _on_cell_changed(self, row, col):
        if self._suppress or col != 1 or row >= len(self._rows):
            return
        path, typ = self._rows[row]
        text = self.table.item(row, col).text()
        try:
            value = _parse(text, typ)
        except (ValueError, TypeError):
            # revert to the stored value on a bad entry
            self._suppress = True
            cur, _ = cfg_get(self._cfg, path)
            self.table.item(row, col).setText(_format(cur, typ))
            self._suppress = False
            return
        cfg_set(self._cfg, path, value)
        # normalise the displayed text to the parsed value
        self._suppress = True
        self.table.item(row, col).setText(_format(value, typ))
        self._suppress = False
        self.configChanged.emit(self.to_config())

    def _on_save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save config', self._save_path, 'JSON files (*.json)')
        if not path:
            return
        with open(path, 'w') as fh:
            json.dump(self.to_config(), fh, indent=2)
        self._save_path = path
