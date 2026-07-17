# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the analysis scripts

Everything lives in the importable `DMSAnalysis` package. Run the apps as modules
from the repository root (no install or build step required):

```bash
python -m DMSAnalysis.slider     [config.json]   # GUI: refine → build curves → fit (image-based)
python -m DMSAnalysis.fit        [config.json]   # batch fivefold-axis fitting (image-based)
python -m DMSAnalysis.tripfit    [config.json]   # batch multiple-intersection lattice fit (image-free)
python -m DMSAnalysis.tripslider [config.json]   # GUI for the multiple-intersection fit (image-free)
python -m DMSAnalysis.dat2config scan.dat out.json --datapoint N --datapoint0 M
```

The slider is the single interactive app: refine geometry with the sliders, click
arcs to select reflections, **Build curves** to integrate the ROIs for the checked
reflections, then **Fit**. `fit.py` is the non-interactive/batch path.

`tripfit.py` is a separate, image-free batch app: it refines a conventional
lattice by driving the three Kossel lines of one or more secondary-reflection
triples to a common point on the stereographic projection (the Renninger
triple-intersection / multiple-diffraction geometry). It needs no detector image
— only the reflection geometry — and is the sensitive probe for the small
lattice distortions of pseudo-symmetric crystals (see the pseudo-cubic
re-indexing note under *Conventional crystals*). `tripslider.py` is its
interactive pyqtgraph GUI (dark theme, matching `slider.py`): drag the free
lattice / ψ sliders and watch each triple's Kossel lines and its residual update
live on a stereographic panel, switch crystal system from a dropdown, pick the
pseudo-cubic re-indexing from the **Pseudo-cubic** dropdown (the 12 Table-1
matrices, `pseudocubic_transform`, exactly as in `slider.py` — selecting one
re-indexes the primary hkl, the azimuthal reference and every triple's reflection
list live as `hkl' = M·hkl`, so you can read off which indexing gives the lowest
triple-intersection residual), and run the optimiser in the background with
**Fit** / **Stop**. The **Triple intersections**
table at the bottom edits the group list at runtime — **Add triple**,
**Duplicate**, **Remove**, and per-cell editing of each group's label, three
reflections, energy, intercepts and target; the panels rebuild live (wrapping to
a grid past four groups). Each row's label cell carries a tick box (`enabled` in
the config): unticking drops that triple from the fit and the summed residual
while its panel stays visible, dimmed and titled *(excluded)*, so you can watch a
triple you are not refining against. **Hide excluded** instead drops the unticked
panels out of the grid so the remaining ones reflow into the freed space; panels
keep their zoom across hiding, ticking and add/remove. `tripfit.py` honours the
same `enabled` flag. It reads and writes the same config schema as
`tripfit.py`, so a config saved from the GUI runs unchanged in the batch app.

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
│   ├── tripfit.py            # Batch image-free lattice fit via Kossel-line triple intersections
│   ├── tripslider.py         # pyqtgraph GUI for the multiple-intersection (tripfit) fit
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
| `computation` | `numsteps`, `simsigma_per_zoom`, `thrange_delta`, `bravais`, `pseudocubic_transform` (1–12, conventional only), `opt_method`, `peak_method` (`gauss`/`centroid`), `tolerance` |
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

**Pseudo-cubic re-indexing.** Indexing mistakes are easy to make on pseudo-cubic
samples. `computation.pseudocubic_transform` (1–12, default 1 = identity) selects
one of the 12 equivalent-indexing matrices from Table 1 of Nisbet et al. (2023),
*J. Appl. Cryst.* **56**, 1046–1050 (doi:10.1107/S1600576723004120), applied as
`hkl' = M @ hkl` to the primary hkl, the azimuthal reference and the reflection
list (conventional modes only; lattice parameters untouched). The matrices live
in `ts_quasi.py` (`PSEUDOCUBIC_TRANSFORMS`, `pseudocubic_matrix`,
`pseudocubic_label`). In the slider, the **Pseudo-cubic M** combo in the Crystal
type box switches the active matrix at runtime, re-indexing the current
selection in place; exported workflow configs always carry already-re-indexed
values with `pseudocubic_transform` reset to 1 so the matrix is never applied
twice.

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

`tripfit.py` writes a lighter snapshot under `Processing/<YYYYMMDDHHMM>_TripFit/`
(`tripfit.py`, `ts_quasi.py`, the config, `PLOT.svg`, `Result.txt`).

## Multiple-intersection (tripfit) configuration

`tripfit.py` reads its own JSON schema (image-free — there is no `scan`, `roi` or
`ref_6d` section):

| Section | Purpose |
|---------|---------|
| `flags` | `save`, `fit` — run controls |
| `geometry` | `hkl` (primary reflection), `azir` (azimuthal reference) |
| `computation` | `bravais` (a `CONVENTIONAL_SYSTEMS` name), `resolution` (Kossel-line sampling for the fit), `opt_method` (any name in `ts_quasi.TRIPFIT_METHODS` — see *Optimiser methods* below), `tolerance`, `boundrange` `[lo,hi]` added to the guess for bounds, optional `rr` (azimuthal pre-rotation, deg), `bh_niter`, `de_strategy`, `fd_step` (finite-difference step for the gradient methods; omit/`null` to use SciPy's default), `pseudocubic_transform` (1–12, GUI only — the Table-1 pseudo-cubic indexing matrix applied to the base indexing at load, same key/semantics as `fit.py`/`slider.py`; 1 = identity), and (GUI only) `live_resolution` for the interactive overlay |
| `crystal` | `initial_guess` — full 6-element lattice `[a,b,c,α,β,γ]`; only the crystal system's free slots are refined |
| `intersections` | list of triples, each `{label, reflist (3×3), energy, target, enabled}` — the three secondary reflections whose Kossel lines must meet. Which crossing of each line pair to score is chosen automatically: the engine takes the tightest (mutually-closest) triple, so the selection stays consistent and the residual doesn't jump as the lattice varies. `enabled` (default `true`, the GUI's per-row tick box) drops a triple from the objective while still plotting it, dimmed. (A legacy `intercepts` index vector, if present, is ignored.) |
| `display` | `lim`, `dpi` — plot settings |

The lattice constraints reuse `ts_quasi.lattice_free_slots` / `expand_lattice`
(the same table-driven layer as the image fit), so e.g. `rhombohedral` refines
`[a, α]` only. The objective is the summed triple-intersection residual over all
groups; `fit=0` just evaluates and plots at the initial guess. The engine
(`ts_quasi.kosscalc`, `stereoproj`, `intersections`, `tripfit`) is ported from
the standalone `calcms/ts_light.py` so the whole workflow lives in the package.
See `configs/tripfit_rhombohedral_PMN_PT_example.json`.

### The residual

Per triple, `ts_quasi.tripfit.fit` scores the three pairwise Kossel-line
crossings `v1, v2, v3` (the tightest triple, picked by `_intercepts`) with
`ts_quasi.triple_spread` — the summed **squared** pairwise distance

```
S = |v1-v2|² + |v2-v3|² + |v1-v3|²
```

and returns `|S - target|`; the app sums that over the enabled triples. `S` is
the least-squares spread of the three crossings: zero only when the lines meet
at a point, every term non-negative (so a wide triple cannot score low through
cancellation), invariant under relabelling the points, and smooth/quadratic at
the minimum so the gradient methods behave. Being squared, `S` scales as the
*square* of the miss distance — a residual of 1e-10 means the crossings are
~1e-5 apart. The same function backs the GUI's live residual
(`tripslider.residual_from_intercepts`), so the two cannot drift. A failed
evaluation (e.g. a line pair that does not intersect) scores the flat penalty
`500`.

> This replaces the original `ts_light.py` expression
> `Σ(|vᵢ| - v̂ᵢ·vⱼ)`, which summed signed terms inside a single `abs`: it used
> `|v1|` twice and `|v3|` never, was not invariant under relabelling, and could
> score a widely-separated triple near zero through cancellation. Residuals
> from before this change (including those in older `Processing/` snapshots) are
> **not comparable** to current ones.

### Optimiser methods

`opt_method` accepts any name in `ts_quasi.TRIPFIT_METHODS`, dispatched by the
shared `ts_quasi.run_tripfit_optimiser` (used by both `tripfit.py` and the GUI,
so the two cannot drift):

| Family | Methods | Notes |
|--------|---------|-------|
| Direct search | `Powell`, `Nelder-Mead`, `COBYLA` | No derivatives; slowest but grind closest to machine precision |
| Gradient-based | `L-BFGS-B`, `SLSQP`, `TNC`, `BFGS`, `CG` | Finite-difference gradients; typically 1–2 orders of magnitude fewer evaluations |
| Global | `GA` (differential evolution), `BH<local>` (basin hopping, e.g. `BHPowell`, `BHL-BFGS-B`) | For escaping the local minima the objective does have |

Bounds (from `boundrange`) are passed only to the methods that accept them
(`L-BFGS-B`, `SLSQP`, `TNC`, and `GA`); the rest run unbounded. Each method is
given only the SciPy `options` keys it actually understands — note `Powell` takes
`xtol`/`ftol` but `Nelder-Mead` takes `xatol`/`fatol`, and COBYLA takes neither.
The legacy name `BHNelderMead` is accepted as an alias for `BHNelder-Mead`
(`ts_quasi.tripfit_method`); it previously fed SciPy an invalid inner method.

## Physics context

This code analyses **X-ray multiple scattering (MS)** in an **icosahedral quasicrystal** (Al-Pd-Mn) measured at Diamond Light Source beamline i16. The quasicrystal is indexed in 6D using pairs `(h, k, l)` and `(h', k', l')` where the physical reciprocal vector is `h + h'·τ` (with τ = golden ratio). Phason strain is a 3×3 matrix coupling the perpendicular-space component; it is the main physically interesting quantity being refined. Bragg geometry, Ewald sphere construction, and ROI-based Gaussian peak fitting are all handled by `ts_quasi.py`.

## Dependencies

```
numpy  scipy  matplotlib  PIL(Pillow)  shapely  imageio  joblib
PyQt5  pyqtgraph   (for the slider GUI)
cctbx  (optional, for loadcif)
```

`cctbx`/`iotbx` imports are commented out in `ts_quasi.py`; `loadcif()` requires them at runtime only when `autoreflist=1`.
