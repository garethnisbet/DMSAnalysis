# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the analysis scripts

Everything lives in the importable `DMSAnalysis` package. Run the apps as modules
from the repository root (no install or build step required):

```bash
python -m DMSAnalysis.slider    [config.json]   # GUI: refine → build curves → fit
python -m DMSAnalysis.fit       [config.json]   # batch fivefold-axis fitting
python -m DMSAnalysis.dat2config scan.dat out.json --datapoint N --datapoint0 M
```

The slider is the single interactive app: refine geometry with the sliders, click
arcs to select reflections, **Build curves** to integrate the ROIs for the checked
reflections, then **Fit**. `fit.py` is the non-interactive/batch path.

Each app falls back to the example config in `DMSAnalysis/configs/` when no path is given.

## Architecture

```
DMS/                          # repository root
├── DMSAnalysis/              # the package
│   ├── ts_quasi.py           # Core library: crystallography, MS geometry, fitting, ROI builders
│   ├── loader.py             # Reads Diamond Light Source .dat scan files into a dict-like object
│   ├── dat2config.py         # Extracts scan metadata from a .dat into a config (the only .dat reader)
│   ├── config_table.py       # Shared editable Qt table view of a config dict
│   ├── slider.py             # The GUI: refine → build integrated curves → fit
│   ├── fit.py                # Batch fitting script: loads data, builds ROIs, runs optimiser
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
| `computation` | `numsteps`, `simsigma_per_zoom`, `thrange_delta`, `bravais`, `opt_method`, `peak_method` (`gauss`/`centroid`), `tolerance` |
| `crystal` | `lattice2`, `initial_guess_base`, `ref_6d` (quasicrystal 6D reflections) **or** `reflist_hkl` (conventional 3-index reflections) — starting parameters and reference reflections |
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

## Conventional crystals

The same engine and apps also handle **ordinary (non-quasicrystal) crystals**
indexed with plain 3-element Miller indices `[h,k,l]`. Set `computation.bravais`
to one of the 7 standard crystal systems and supply reflections as a 3-index
list:

```
cubic  tetragonal  orthorhombic  monoclinic  rhombohedral  hexagonal  triclinic
```

In this mode there is **no cut-and-projection and no phason matrix** — the
perpendicular reflection component and the phason block (indices 15–23) are held
at zero, and the lattice slots [0–5] = `[a,b,c,α,β,γ]` carry the real cell. Each
system frees only its symmetry-allowed lattice parameters (e.g. tetragonal frees
`a` and `c` and forces `b=a, α=β=γ=90`; monoclinic uses the b-unique setting with
free `β`). The free-parameter mapping is table-driven in
`ts_quasi.py`: `CONVENTIONAL_SYSTEMS`, `lattice_free_slots`, `expand_lattice`,
`reduced_param_indices`, and `hklgen_3d` (the 3D analogue of the 6D reflection
generator), all shared by `slider.py` and `fit.py` so the parameter packing
cannot drift.

Reflections are supplied via `crystal.reflist_hkl` (a list of `[h,k,l]`), the
depth-based generator (`hklgen_3d` / the slider's **Auto reflist** + **Depth**),
or the slider's **Geo 3-click** identify — exactly as for the quasicrystal, but
with 3-element vectors. See
`configs/fit_conventional_tetragonal_PMN_PT_example.json` for a worked example.

In the slider, the **Crystal type** dropdown switches the active mode at runtime
between Icosahedral (and the `icosahedral_fixed_a` / `cubic_no_strain` variants)
and the 7 conventional systems. Switching rebuilds the lattice sliders for the
new symmetry and regenerates the reflection list; because 6D and 3-index
reflections are incompatible, the current selection is cleared. `fit.py` (batch)
takes its mode from `computation.bravais` in the config.

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
PyQt5  pyqtgraph   (for the slider GUI)
cctbx  (optional, for loadcif)
```

`cctbx`/`iotbx` imports are commented out in `ts_quasi.py`; `loadcif()` requires them at runtime only when `autoreflist=1`.
