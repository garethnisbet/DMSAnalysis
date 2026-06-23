"""
DMSAnalysis — X-ray multiple-scattering analysis for icosahedral quasicrystals.

Library modules:
    ts_quasi      crystallography, MS geometry, fitting, ROI builders
    loader        reads Diamond Light Source ``.dat`` scan files
    dat2config    extracts scan metadata from a ``.dat`` into a config dict
    config_table  shared editable Qt table view of a config

Applications (run with ``python -m DMSAnalysis.<name>``):
    slider        interactive refinement → build integrated curves → fit (the GUI)
    fit           batch fivefold-axis fitting script
"""

from . import ts_quasi, loader, dat2config

__all__ = ["ts_quasi", "loader", "dat2config", "config_table"]
