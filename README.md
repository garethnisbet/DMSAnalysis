# DMSAnalysis

Analysis of **X-ray multiple scattering (MS)** in icosahedral quasicrystals
(Al–Pd–Mn), measured at Diamond Light Source beamline i16. The quasicrystal is
indexed in 6D using reflection pairs `(h, k, l)` and `(h′, k′, l′)`, where the
physical reciprocal vector is `h + h′·τ` (τ = golden ratio). The main quantity
being refined is the **phason strain** — a 3×3 matrix coupling the
perpendicular-space component of each reflection.

The same engine also fits **ordinary (non-quasicrystal) crystals** indexed with
plain 3-index Miller reflections and constrained by crystal system, including
support for the **pseudo-cubic indexing ambiguity** (re-indexing by any of the 12
equivalent matrices of Nisbet et al., 2023).

The package provides a core geometry/fitting library plus four applications, in
two families:

- **Image-based** (match predicted multiple-scattering streaks in a detector
  image): `slider` — an interactive GUI to refine geometry, build integrated
  curves, then fit; and `fit` — the batch equivalent.
- **Image-free** (match the multiple-diffraction geometry directly, via the
  coincidence of Kossel lines on the stereographic projection): `tripslider` — an
  interactive GUI; and `tripfit` — the batch equivalent. These refine a
  conventional lattice by driving secondary-reflection triples to a common
  triple-intersection point, the sensitive probe for small lattice distortions of
  pseudo-symmetric crystals.

## Installation

No build step. Clone the repository and run from its root. Requirements:

```
numpy  scipy  matplotlib  Pillow  shapely  imageio  joblib
PyQt5  pyqtgraph          # for the slider / tripslider GUIs
cctbx                     # optional, only for loadcif() when autoreflist=1
```

```bash
pip install numpy scipy matplotlib Pillow shapely imageio joblib PyQt5 pyqtgraph
```

## Usage

All apps run as modules from the repository root and accept an optional config
path (falling back to the example config in `DMSAnalysis/configs/`):

```bash
# Image-based interactive GUI — refine geometry, build integrated curves, then fit
python -m DMSAnalysis.slider [config.json]

# Image-based batch fitting (non-interactive)
python -m DMSAnalysis.fit [config.json]

# Image-free multiple-intersection GUI — refine a lattice from Kossel-line triples
python -m DMSAnalysis.tripslider [config.json]

# Image-free multiple-intersection batch fit
python -m DMSAnalysis.tripfit [config.json]

# Convert a Diamond .dat scan file into a config (the only .dat reader)
python -m DMSAnalysis.dat2config scan.dat out.json --datapoint N --datapoint0 M
```

Typical flow in the **slider** (image-based):
1. **Browse… → Load Scan** — the scan's own lattice, energy, primary `hkl` and
   `psi` come off the `.dat` and seed the sliders (untick **Seed sliders from
   .dat** to load the image against the geometry already on screen).
2. Refine geometry with the sliders over the detector image.
3. Click arcs to select reflections; check/uncheck them in the list
   (right-click a line to remove it).
4. **Build curves** — integrate the ROIs for the checked reflections.
5. **Fit** — run the optimiser; fitted parameters flow back to the sliders, and
   the run is written to `Processing/` (see *Output*). Choosing **No Fit
   (evaluate + save)** from the algorithm list writes that record for the
   current guess without refining anything.

**Save config** writes the current state (incl. selected reflections) for batch
runs via `python -m DMSAnalysis.fit`. The config is the single source of truth —
once the `experiment` block is populated, the apps never read the `.dat` again.
The three panel dividers and the window geometry are remembered between
sessions.

Typical flow in the **tripslider** (image-free):
1. Add or edit triple intersections in the **Triple intersections** table (each
   row is three secondary reflections that should meet at one point).
2. Drag the free lattice / ψ sliders and watch each triple's Kossel lines and its
   residual update live on a stereographic panel; switch crystal system from the
   dropdown.
3. **Fit** — run the optimiser in the background (**Stop** to interrupt).
4. **Save config** to re-run the exact setup in `python -m DMSAnalysis.tripfit`.

### Conventional crystals and pseudo-cubic re-indexing

For an ordinary crystal, set `computation.bravais` to a standard system (`cubic`,
`tetragonal`, `orthorhombic`, `monoclinic`, `rhombohedral`, `hexagonal`,
`triclinic`) and supply 3-index reflections via `crystal.reflist_hkl`. To test the
equivalent pseudo-cubic indexing choices, set `computation.pseudocubic_transform`
(1–12; 1 = identity) — this re-indexes the primary reflection, azimuthal
reference and reflection list by the chosen matrix from Table 1 of
[Nisbet et al. (2023), *J. Appl. Cryst.* **56**, 1046–1050](https://doi.org/10.1107/S1600576723004120).
The slider also exposes a **Pseudo-cubic M** dropdown that applies these live. The
12 matrices are a coset decomposition (verifiable via
`ts_quasi.verify_pseudocubic_transforms`; tests in
`DMSAnalysis/tests/test_pseudocubic.py`).

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
│   ├── slider.py             # image-based GUI: refine → build curves → fit
│   ├── fit.py                # image-based batch fitting
│   ├── tripfit.py            # image-free multiple-intersection batch lattice fit
│   ├── tripslider.py         # image-free multiple-intersection GUI
│   ├── configs/              # example JSON configs
│   ├── tests/                # self-verification tests (geometry, ROI kernels, GUI behaviour)
│   └── README.md             # library API reference
└── Processing/               # timestamped run snapshots (created in CWD when save=1)
```

## Configuration

The **image-based** apps (`slider`, `fit`) read a JSON config with these sections:

| Section | Purpose |
|---------|---------|
| `scan` | `scannum`, `scanpath`, `datapoint`, `datapoint0` — which scan/image to load |
| `experiment` | `lattice`, `energy`, `energy0`, `azir`, `hkl`, `psi`, `image_template` — metadata extracted from the `.dat` |
| `geometry` | `hkl`, `psi`, `px_unscaled`, `py_unscaled`, `scatv` — primary reflection and detector origin |
| `display` | `zoomval`, `colourlim`, `colmap` — image display |
| `roi` | `width_per_zoom`, `comwidth_per_zoom` — ROI extraction widths |
| `computation` | `numsteps`, `simsigma_per_zoom`, `thrange_delta`, `bravais`, `pseudocubic_transform` (conventional only), `opt_method`, `tolerance` |
| `crystal` | `lattice2`, `initial_guess_base` (24-element vector), `ref_6d` (6D) **or** `reflist_hkl` (3-index) — starting parameters |
| `flags` | `save`, `fit`, `firstplot`, `detoptimize`, `energyopt` — run controls |
| `paths` | `cif_file` — CIF used by `loadcif()` when `autoreflist=1` |

The **image-free** apps (`tripfit`, `tripslider`) read a lighter, separate schema
(no `scan`/`roi`/`ref_6d`): `geometry` (`hkl`, `psi`, `azir`), `computation`
(`bravais`, `resolution`, `opt_method`, `tolerance`, `boundrange`, …),
`crystal.initial_guess` (6-element lattice), and an `intersections` list — one
entry per triple `{label, reflist (3×3), energy, intercepts, target}`. See
[`configs/tripfit_rhombohedral_PMN_PT_example.json`](DMSAnalysis/configs/tripfit_rhombohedral_PMN_PT_example.json).

See [`CLAUDE.md`](CLAUDE.md) for the parameter-vector index map, the full tripfit
schema, and developer notes.

## Output

Every fit finished in the **slider** writes a run record under
`Processing/<scannum>_dp<datapoint>_<YYYYMMDD-HHMMSS>_<method>/`:

| File | What it is |
|------|------------|
| `Result.txt` | the solution — residual, refined parameters, per-ROI centres — and the recipe to rerun it: starting guess, optimiser, free parameters and their bounds, sampling and peak settings |
| `IM_<scan>_dp<dp>.png` | the detector image with the simulated DMS lines over it |
| `PLOT_<scan>_dp<dp>.svg` | the integrated ROI curves, as vector art |

**Save fit snapshot → Processing** writes those three again in a folder of its
own, plus the code, config and `res.x.txt` for reproducibility.

With `save=1`, the batch image fit creates an immutable snapshot under
`Processing/YYYYMMDDHHMM_<imnum>_<scannum>_<description>_<fittype>/` containing
the script, library, config, fit results, and rendered images. `tripfit` writes a
lighter snapshot under `Processing/YYYYMMDDHHMM_TripFit/` (script, library,
config, `PLOT.svg`, `Result.txt`).

## Author & license

Dr Gareth Nisbet, Diamond Light Source. Apache 2.0.
