"""The Qt binding every GUI in the package imports from.

PyQt6 is preferred: it ships wheels for arm64 (Apple silicon, aarch64 Linux)
where PyQt5 frequently has none.  PyQt5 is the fallback, so an existing install
keeps working.  Setting ``PYQTGRAPH_QT_LIB`` to ``PyQt5`` or ``PyQt6`` forces
the choice.

The code is written to the API the two share: scoped enums
(``QtCore.Qt.AlignmentFlag.AlignLeft``, which PyQt5 5.15 also accepts) and
``exec()`` rather than ``exec_()``.  ``QShortcut`` moved from QtWidgets to QtGui
in Qt6, so it is exported from here.

This module must be imported before ``pyqtgraph``, which reads
``PYQTGRAPH_QT_LIB`` at import time to pick the same binding.
"""

import os

_forced = os.environ.get('PYQTGRAPH_QT_LIB')
_order = [_forced] if _forced in ('PyQt6', 'PyQt5') else ['PyQt6', 'PyQt5']

for QT_LIB in _order:
    try:
        if QT_LIB == 'PyQt6':
            from PyQt6 import QtWidgets, QtCore, QtGui
        else:
            from PyQt5 import QtWidgets, QtCore, QtGui
        break
    except ImportError:
        continue
else:
    raise ImportError('The DMSAnalysis GUIs need PyQt6 (or PyQt5): '
                      'pip install PyQt6 pyqtgraph')

os.environ['PYQTGRAPH_QT_LIB'] = QT_LIB

QShortcut = getattr(QtGui, 'QShortcut', None) or QtWidgets.QShortcut


def check_state(state):
    """A ``stateChanged`` argument as a ``Qt.CheckState``.  The signal carries
    a plain int in PyQt6, which never compares equal to the enum."""
    return QtCore.Qt.CheckState(state)
