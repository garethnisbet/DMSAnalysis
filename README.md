# DMSAnalysis

Analysis of **X-ray multiple scattering (MS)** in icosahedral quasicrystals
(Al–Pd–Mn), measured at Diamond Light Source beamline i16. The quasicrystal is
indexed in 6D using reflection pairs `(h, k, l)` and `(h′, k′, l′)`, where the
physical reciprocal vector is `h + h′·τ` (τ = golden ratio). The main quantity
being refined is the **phason strain** — a 3×3 matrix coupling the
perpendicular-space component of each reflection.

The package provides a core geometry/fitting library plus three applications:
an interactive slider for manual refinement, a unified slider→fit workflow, and
a batch fitting script.

## Installation

No build step. Clone the repository and run from its root. Requirements:

```
numpy  scipy  matplotlib  Pillow  shapely  imageio  joblib
PyQt5  pyqtgraph          # for the slider / workflow GUIs
cctbx                     # optional, only for loadcif() when autoreflist=1
```

```bash
pip install numpy scipy matplotlib Pillow shapely imageio joblib PyQt5 pyqtgraph
```

## Usage

All apps run as modules from the repository root and accept an optional config
path (falling back to the example config in `DMSAnalysis/configs/`):

```bash
# Interactive slider — manually refine the initial guess over the detector image
python -m DMSAnalysis.slider [config.json]

# Unified workflow — slider refinement → automated optimiser
python -m DMSAnalysis.workflow [config.json]

# Batch fivefold-axis fitting
python -m DMSAnalysis.fit [config.json]

# Convert a Diamond .dat scan file into a config (the only .dat reader)
python -m DMSAnalysis.dat2config scan.dat out.json --datapoint N --datapoint0 M
```

Typical flow: run `dat2config` (or the slider's **Import**) to extract scan
metadata into a config, refine geometry in the **slider**, then **Launch** the
**workflow** to fit. The config is the single source of truth — once the
`experiment` block is populated, the apps never read the `.dat` again.

### Using the library

```python
from DMSAnalysis import ts_quasi as ts

lattice = [6.458, 6.458, 6.458, 90, 90, 90]
thb = ts.bragg(lattice, [1, 1, 1], 6.3).th()[0]
```

Full API documentation: [`DMSAnalysis/README.md`](DMSAnalysis/README.md).

## Layout

```
DMS/                          # repository root
├── DMSAnalysis/              # the package
│   ├── ts_quasi.py           # core: crystallography, MS geometry, fitting, ROI builders
│   ├── loader.py             # reads Diamond .dat scan files
│   ├── dat2config.py         # extracts scan metadata from a .dat into a config
│   ├── config_table.py       # shared editable Qt table view of a config
│   ├── slider.py             # interactive slider visualiser
│   ├── workflow.py           # slider refinement → automated fitting
│   ├── fit.py                # batch fivefold-axis fitting
│   ├── configs/              # example JSON configs
│   └── README.md             # library API reference
└── Processing/               # timestamped run snapshots (created in CWD when save=1)
```

## Configuration

Each app reads a JSON config. Key sections:

| Section | Purpose |
|---------|---------|
| `scan` | `scannum`, `scanpath`, `datapoint`, `datapoint0` — which scan/image to load |
| `experiment` | `lattice`, `energy`, `energy0`, `azir`, `image_template` — metadata extracted from the `.dat` |
| `geometry` | `hkl`, `psi`, `px_unscaled`, `py_unscaled`, `scatv` — primary reflection and detector origin |
| `display` | `zoomval`, `colourlim`, `colmap` — image display |
| `roi` | `width_per_zoom`, `comwidth_per_zoom` — ROI extraction widths |
| `computation` | `numsteps`, `simsigma_per_zoom`, `thrange_delta`, `bravais`, `opt_method`, `tolerance` |
| `crystal` | `lattice2`, `initial_guess_base` (24-element vector), `ref_6d` — starting parameters |
| `flags` | `save`, `fit`, `firstplot`, `detoptimize`, `energyopt` — run controls |
| `paths` | `cif_file` — CIF used by `loadcif()` when `autoreflist=1` |

See [`CLAUDE.md`](CLAUDE.md) for the parameter-vector index map and developer notes.

## Output

With `save=1`, the fit creates an immutable snapshot under
`Processing/YYYYMMDDHHMM_<imnum>_<scannum>_<description>_<fittype>/` containing
the script, library, config, fit results, and rendered images.

## Author & license

Dr Gareth Nisbet, Diamond Light Source. Apache 2.0.
