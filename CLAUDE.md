# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the analysis scripts

Everything lives in the importable `DMSAnalysis` package. Run the apps as modules
from the repository root (no install or build step required):

```bash
python -m DMSAnalysis.slider    [config.json]   # interactive slider refinement
python -m DMSAnalysis.workflow  [config.json]   # slider refinement → automated fitting
python -m DMSAnalysis.fit       [config.json]   # batch fivefold-axis fitting
python -m DMSAnalysis.dat2config scan.dat out.json --datapoint N --datapoint0 M
```

Each app falls back to the example config in `DMSAnalysis/configs/` when no path is given.

## Architecture

```
DMS/                          # repository root
├── DMSAnalysis/              # the package
│   ├── ts_quasi.py           # Core library: crystallography, MS geometry, fitting, ROI builders
│   ├── loader.py             # Reads Diamond Light Source .dat scan files into a dict-like object
│   ├── dat2config.py         # Extracts scan metadata from a .dat into a config (the only .dat reader)
│   ├── config_table.py       # Shared editable Qt table view of a config dict
│   ├── slider.py             # Interactive slider visualiser for quasicrystal MS simulation
│   ├── workflow.py           # Unified slider refinement → automated fitting
│   ├── fit.py                # Fitting script: loads data, builds ROIs, runs optimiser
│   ├── configs/              # Example JSON configs shipped with the package
│   └── README.md             # Full library API documentation
└── Processing/               # Timestamped output snapshots (auto-created when save=1, in CWD)
```

`ts_quasi.py` is the core library module. Apps use package-relative imports
(`from . import ts_quasi as ts`, `from . import loader as do`). Full API
documentation is in `DMSAnalysis/README.md`.

## JSON configuration

Each app reads a JSON config (passed as an argument, or the `configs/` default). Key sections:

| Section | Purpose |
|---------|---------|
| `scan` | `scannum`, `scanpath`, `datapoint`, `datapoint0` — which scan file and image to load |
| `flags` | `save`, `fit`, `firstplot`, `detoptimize`, `energyopt` — boolean run controls |
| `display` | `zoomval` (1 or 2), `colourlim`, `colmap` — image display settings |
| `roi` | `width_per_zoom`, `comwidth_per_zoom` — ROI extraction widths (scaled by `zoomval`) |
| `geometry` | `hkl`, `psi`, `px_unscaled`, `py_unscaled` — primary reflection and detector origin |
| `computation` | `numsteps`, `simsigma_per_zoom`, `thrange_delta`, `bravais`, `opt_method`, `tolerance` |
| `crystal` | `lattice2`, `initial_guess_base`, `ref_6d` — starting parameters and 6D reference reflections |
| `manual_centres` | Dict of `"roi_index": pixel_position` overrides for poorly fitted ROI centres |
| `paths` | `cif_file` — path to CIF file used by `loadcif()` |

## Initial guess parameter vector (fit script)

`initial_guess_base` in the JSON is a 24-element array. Indices:

```
0        a (lattice parameter, Å)
1–2      b, c  (unused for icosahedral — cubic constraint applied)
3–5      alpha, beta, gamma  (unused for icosahedral)
6–9      psicor, hcor, kcor, lcor  (azimuthal/hkl corrections)
10       detdist (detector distance, pixels; halved and scaled by zoomval at runtime)
11–13    dxrot, dyrot, dzrot  (detector rotation angles, degrees)
14       energy offset (added to loaded energy value)
15–23    phason strain matrix elements (3×3 upper-triangular packed)
```

The `bravais` flag selects which subset of indices are passed to the optimiser. For `icosahedral`, parameters [0, 6–9, 10–13, 15–23] (with optional energy) are optimised; lattice parameters 1–5 are locked by symmetry.

## Processing output

When `save=1`, the script creates a timestamped directory under `Processing/`:

```
Processing/YYYYMMDDHHMM_<imnum>_<scannum>_<description>_<fittype>/
    fit.py                   # snapshot of the script
    ts_quasi.py              # snapshot of the library
    <config>.json            # snapshot of the config used
    IM_<scannum>.png
    _PLOT_<scannum>.svg
    Result.txt
    res.x.txt
    ROIS<scannum>.png
```

These directories are immutable run records — do not modify them.

## Physics context

This code analyses **X-ray multiple scattering (MS)** in an **icosahedral quasicrystal** (Al-Pd-Mn) measured at Diamond Light Source beamline i16. The quasicrystal is indexed in 6D using pairs `(h, k, l)` and `(h', k', l')` where the physical reciprocal vector is `h + h'·τ` (with τ = golden ratio). Phason strain is a 3×3 matrix coupling the perpendicular-space component; it is the main physically interesting quantity being refined. Bragg geometry, Ewald sphere construction, and ROI-based Gaussian peak fitting are all handled by `ts_quasi.py`.

## Dependencies

```
numpy  scipy  matplotlib  PIL(Pillow)  shapely  imageio  joblib
PyQt5  pyqtgraph   (for the slider/workflow GUIs)
cctbx  (optional, for loadcif)
```

`cctbx`/`iotbx` imports are commented out in `ts_quasi.py`; `loadcif()` requires them at runtime only when `autoreflist=1`.
