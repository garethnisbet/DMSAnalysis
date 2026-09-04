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
reflections, then **Fit**. `fit.py` is the non-interactive/batch path. The
**Curves** combo in the *Fit* box picks how the DMS lines themselves are computed
— the sampled θ-sweep, or the circles they analytically are (see *DMS curve
method* below).

Two sets of view toggles, both off by default and neither touching the fit:
**ROIs** (next to *DMS lines* / *Labels*) outlines the integration strips on the
detector image — see *The ROI pair* below; and above the integrated-curve grid,
**Axes** puts ticks on each ROI panel (x is the position across the integration
width, the units the centres and residual are in; y the integrated intensity),
**Drag zoom** makes left-drag zoom to a rectangle on a panel instead of panning,
and **Reset zoom** rescales every panel to its curves and re-enables
auto-scaling. Left-click still selects a ROI and right-click still assigns its
centre in either mouse mode.

Missing scan data never stops the slider from opening. If the config's `.dat` or
its detector image cannot be read (a beamline path that does not exist on this
machine, data on another disk, …), the app starts on placeholder metadata
(`slider.fallback_experiment` — lattice from `crystal.initial_guess_base`, energy
set to put the primary reflection at a 20° Bragg angle) and a blank frame, lists
what was missing in a startup dialog, and lets the user browse to the real file
with the Scan loader (**Browse…** → **Load**), which replaces all of it. A scan
whose `.dat` reads but whose image is absent loads too — metadata is applied and
the blank frame is kept, noted in the status line. `fit.py` (batch) still fails
loudly on missing data.

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
| `computation` | `numsteps`, `simsigma_per_zoom`, `thrange_delta`, `bravais`, `pseudocubic_transform` (1–12, conventional only), `curve_method` (`sweep`/`circle` — see *DMS curve method*), `opt_method`, `peak_method` (`gauss`/`centroid`), `tolerance` |
| `crystal` | `lattice2`, `initial_guess_base`, `ref_6d` (quasicrystal 6D reflections) **or** `reflist_hkl` (conventional 3-index reflections) — starting parameters and reference reflections |
| `manual_centres` | Dict of `"roi_index": pixel_position` overrides for poorly fitted ROI centres |
| `paths` | `cif_file` — path to CIF file used by `loadcif()` |

## Initial guess parameter vector (fit script)

`initial_guess_base` in the JSON is a 24-element array. Indices:

```
0        a (lattice parameter, Å)
1–2      b, c  (unused for icosahedral — cubic constraint applied)
3–5      alpha, beta, gamma  (unused for icosahedral)
6–9      psicor, chicor, thcor, lcor  (azimuthal / chi-axis / Bragg-angle
         corrections; slots 7 and 8 were formerly hcor/kcor, and slot 9 is
         unused — every branch of imcalc holds the hkl corrections at zero)
10       detdist (detector distance, pixels; halved and scaled by zoomval at runtime)
11–13    dxrot, dyrot, dzrot  (detector rotation angles, degrees)
14       energy offset (added to loaded energy value)
15–23    phason strain matrix elements (3×3 upper-triangular packed)
```

The `bravais` flag selects which subset of indices are passed to the optimiser. For `icosahedral`, parameters [0, 6–9, 10–13, 15–23] (with optional energy) are optimised; lattice parameters 1–5 are locked by symmetry.

## DMS curve method: the sampled sweep, or the circle the cone is

A DMS (Kossel) line comes from **one secondary reflection**: the doubly-diffracted
radiation leaves the sample along a **cone** of exit directions, all satisfying
that plane's diffraction condition. A cone of unit vectors is a circle on the
sphere, so each locus is **exactly a circle** in exit-direction space — the
θ-scan only samples it. `computation.curve_method` (slider: the **Curves** combo
in the *Fit* box) picks how that is delivered:

| Method | What the engine draws | What `numsteps` (**Points**) buys |
|--------|-----------------------|-----------------------------------|
| `sweep` (default) | the sampled scan points, joined | both the smoothness of the curve *and* where it ends |
| `circle` | each continuous run reduced to the circle it lies on, re-sampled at ~0.5 px | only where each arc **ends**; the curve between the ends is exact at any resolution |

**It is not a free speedup.** Circle mode runs the whole θ-sweep first (that is
where the arc ends come from), then fits, measures and re-samples, so at the
*same* Points it costs 2.5–5x more. The win is at equal *quality* — 40
conventional reflections over a 1200x1200 plate, worst gap against a 4000-point
sweep:

| | ms | worst gap |
|---|---|---|
| sweep @ 100 | 4.4 | 107 px |
| **circle @ 100** | **20.0** | **1.0 px** |
| sweep @ 1000 | 17.8 | 19 px |
| circle @ 1000 | 43.6 | 1.0 px |
| sweep @ 4000 | 72.0 | — |

So circles at 100 points cost about what a 1000-point sweep costs and are ~19x
more faithful, or a third of the 4000-point sweep that would match them. 1.0 px
is the engine's own whole-pixel rounding — the floor, not a limit of the method.
The rule is to switch the method *and* drop Points; leaving Points at 1000 just
makes every overlay update and fit evaluation slower for a curve that was
already right.

Both the overlay and the fit engine take the setting, so the residual is
always scored on the curves that are on screen; it is saved in the session, in
an exported fit config, and read back by `fit.py`. The **ROI kernel** is still
built from the sampled scan (`roibuilder_ico_hkl`): it only has to lay a path
along the line, and it is built once and reused.

**The ROIs do not change with the curve method.** Each reflection gets *two*
ROIs — the builder cuts the line in half along itself and makes each half a
kernel plane (`2*i`, `2*i+1`). A rigid shift of the line moves both halves
together; a rotation moves them oppositely, which is where the fit's sensitivity
to the line's orientation comes from. `roibuilder_ico_hkl` builds its own
`dmscalc_ico_hkl`, a class with no `curve_method` at all, so in circle mode the
pair is formed exactly as in sweep mode, from the sampled scan. One consequence
worth knowing: the kernel path is interpolated between *sampled* points, so
dropping Points (which circle mode invites) bows the ROI off the arc it is meant
to follow by the chord sagitta — a few pixels at 100 points, inside a typical
45 px width but not zero. The slider's **ROIs** overlay toggle draws the strips
(`ts_quasi.roi_outline`) so this is visible rather than assumed.

### The ROI engine must find the same lines as the fit engine

The ROI builder runs `dmscalc_ico_hkl`; the overlay and the fit run
`dmsfit_ico_hkl`. The two share the geometry code but *did not* share the
physical-solution test: the fit engine drops the θ steps where the Ewald
construction has no solution (`|sin| > 1`, or a negative discriminant), while
the ROI engine clamped both quantities and so turned every non-physical step
into a solution. Those invented points form a perfectly smooth curve of their
own somewhere else on the plate, so the builder could lay a ROI along a line
that does not exist — on `TestExample.json`, `[-1 1 1]` drew a near-vertical arc
and got a near-horizontal ROI 835 px away, and `msroi` then integrated whatever
that strip happened to cross. `dmscalc_ico_hkl.imcalc` now carries the same
`valid` mask and NaN-drops before the integer cast, as the fit engine does.

A cross-engine check is worth keeping in mind for anything else that touches
either `imcalc`: the two must put a reflection's line in the same place, and
`DMSAnalysis/tests/test_roi_kernels.py` asserts it on both a cubic and an
icosahedral fixture (only the icosahedral one has non-physical steps, so it is
the one that catches this class of bug).

### Making the pair from a locus that is not one tidy curve

The path the builder is handed is the reflection's on-detector pixels *in scan
order*, and three things make it messy. Each is handled explicitly, because the
result is not visible in the fit — only in the ROI curves it is scored on:

* **Both psi solutions are in it.** The engine walks ψ₁ and ψ₂ and concatenates
  them, and in many geometries they trace the *same* detector line, so the raw
  index is that line twice over. `roi_dedupe_path` keeps one visit per pixel, in
  first-seen order.
* **The locus can be in several pieces** — it leaves the physical region or the
  plate and comes back. `roi_split_runs` cuts on the gaps (6× the median step,
  floored at 4 px) and the pair is built from the **longest** run; the build
  prints which reflections had more than one piece and how much of the index it
  used.
* **It is not a function of either detector axis.** A DMS line is an arc, and a
  curved one doubles back in whichever axis you sort by.

Before this, the builder sorted the whole index by its dominant detector axis,
cut at the median and ran `interp1d` over each half. All three cases above break
that: the sort interleaves separate pieces and duplicate branches, and the
interpolation then wires them together, so a ROI could leave its line and shoot
hundreds of pixels across the plate. `msroi` integrated whatever that crossed and
took its perpendicular direction from the ROI's first and last pixel, which for
such a ROI means nothing — so the affected reflections contributed a meaningless
curve, and a meaningless centre, to the fit. On the 18-reflection
`TestExample.json` two reflections were affected; their kernels held jumps of up
to 830 px.

Also fixed there: the builder passed the *whole* `reflist2` (perpendicular
components) while passing a single parallel reflection, and `PhasonDistoArray`
broadcasts, so every ROI was built from N loci instead of one — N identical
copies for a conventional crystal, N slightly different ones for a quasicrystal,
each carrying another reflection's phason shift. It now passes row `i` only.

Tests: `DMSAnalysis/tests/test_roi_kernels.py`, and
`test_roi_outline.py` for the drawn strip.

The engine (`ts_quasi.dms_circle_curves` and friends, ported from the sibling
`ReciprocalSpaceVisualisation` project's `dms_compute.py`, which does the same
thing in reciprocal space) leaves the worst deviation of any run from its fitted
circle in `dmsfit_ico_hkl.circle_residual`; the slider prints it in the status
line. It is ~1e-13 rad in both lattice modes, at any ψ, with or without phason
strain or a χ correction. Two things to know:

- **A θ range spanning zero** — the default `[θ_B-27, θ_B+10]` does whenever
  θ_B < 27° — reverses the scan vector mid-sweep, re-aligning the crystal and
  putting the rest of the sweep on a *different* circle. The runs are cut there
  (`dms_split_runs`); without that cut such a run fits a circle wrong by ~0.2 rad.
- **A θ correction** (slot 8) shears the locus slightly off-plane, first order in
  θcor (~2e-4 rad at 1°, sub-pixel at a 3000 px detector distance). This is the
  one case where the circle is not exact, which is why the residual is reported
  rather than assumed.

A run too short to fit a circle to, or one the tolerance rejects, is kept as the
points the sweep sampled, so switching method can add resolution but never loses
a curve. Tests: `DMSAnalysis/tests/test_dms_curves.py`.

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
