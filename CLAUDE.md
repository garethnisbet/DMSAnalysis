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

### Loading a scan: the .dat is the source

A `.dat` measures the lattice, the energy, the primary reflection `h k l` and
the azimuth `psi`, and `dat2config.extract_metadata` now returns all four (the
`experiment` block gained `hkl` and `psi`; a scan carrying neither leaves the
keys out). **Every** scan load in the slider seeds the sliders from them —
lattice, energy and h/k/l get their values and their slider ranges recentred,
psi is set — so a new scan needs no config editing: load it, refine from there,
and **Export workflow JSON** / **Save Session** writes what the .dat said or
wherever the sliders have since been dragged. The status line names what was
seeded.

Same scan or not is deliberately *not* part of the decision. The auto-saved
session restores onto the scan it was saved on, so an explicit **Load Scan** of
that same .dat would otherwise be a no-op — which is exactly the surprise it
caused. Prev/Next and a datapoint change seed too.

**Seed sliders from .dat** (in the *Scan* box, on by default) is the off switch,
for loading a scan's image against the geometry currently on the sliders. Only
in that branch does hkl follow the energy ratio across a datapoint step
(`hkl · E[dp]/E[prev]`) instead of the file — running both would apply the ratio
twice. Session restore forces seeding off: the session's own geometry (now
including `psi`, which it did not store before) wins over the file's.

`hkl` at a datapoint comes from the scanned `h`/`k`/`l` columns when the scan has
them. Most scans here (fixed-angle energy scans — 913232, 913123) do not: they
carry one metadata hkl, recorded at the metadata energy `en`. The diffractometer
does not move during those, so the indices at a datapoint are that position
scaled by the Bragg energy ratio, `hkl · E[dp]/en` — on a scan carrying both,
this reproduces the columns to six decimals (`dat2config._hkl_at`). Without it
a datapoint step on such a scan would not move hkl at all.

`_do_load_scan(path, dp, dp0, seed_from_metadata=None)` holds this: `None` asks
the tick box, `True`/`False` force it. It also keeps the module-level `hklint`
(the integer reference reflection every engine is built from) in step with the
live hkl — before, a load left it at the value the app started on. Test:
`DMSAnalysis/tests/test_scan_load_seeding.py`, which drives the real GUI
headlessly over synthetic `.dat` files.

### Right-click removes a line; one empty arc used to disarm it

Every selected reflection owns an arc whose on-detector points are cached on the
item (`_x_data`/`_y_data`) for hit-testing. An arc can hold **no** points — the
reflection's line is off the plate at the current geometry (a scan load moves
plenty of them there), or a bulk add created it empty and the overlay pass has
not traced it yet. `_nearest_arc_at` took `min()` over those arrays unguarded,
so one empty arc raised `ValueError: zero-size array to reduction operation
minimum` *inside the mouse-click slot* — and right-click then removed nothing,
whichever line was clicked. `_nearest_selectable` (middle-click add) had always
guarded this; the removal path had not. `_on_update_done` now also clears the
cache when a reflection drops off the plate, so right-click cannot pick an arc
by where it used to be. Test: `DMSAnalysis/tests/test_arc_picking.py`.

### Identifying a line: psi_err ranks, |q_perp| breaks the tie

Both three-click identify paths — **Geo 3-click** (`_run_geo_search`, the
candidates under `psi_tol`) and the nearest-ref search it replaces when
unticked — score candidates with `_ewald_scores`, the mean ψ error over the
three clicked directions. That score is *purely geometric*: it asks whether a
reflection's cone can pass through those directions, never whether the
reflection is observable. Three consequences, all of which bite hardest on a
quasicrystal:

* It is a **local** test. Three bunched clicks sample a short arc, and distinct
  reflections whose lines osculate there diverge elsewhere. Spread the clicks
  along the feature.
* 6D indexing makes the candidate pool **dense**, so several candidates
  routinely score alike and the ranking alone cannot choose between them.
* It can only return an index **already in the list**. A "no match" at shallow
  **Depth** is not evidence the feature is unindexable — widen **Auto reflist**
  and re-run.

For an icosahedral quasicrystal the structure factor is modulated by the
perpendicular-space component and falls off steeply with |q_perp|, so only a
modest subset of that dense set ever produces a visible line. `_perp_strengths`
supplies it: every ranked candidate now prints `|q_perp|` and the dimensionless
`perp/par` ratio next to `psi_err`, and the best one carries `|q⟂|` on the pick
label. Among candidates the geometry cannot separate, the smallest |q_perp| is
the physically likely one; a large one is almost certainly not producing a
visible line however well it fits the clicks.

It is **reported, not ranked on**. The order stays by `psi_err`, because a
composite score would hide a poor geometric match behind a plausible |q_perp| —
the two numbers answer different questions and are meant to be read together.
The norms are taken on the projected components as `build_reflist_from_6d`
returns them (both in the same units, neither converted to Å⁻¹), which is all a
ranking aid needs. A conventional crystal has no cut-and-projection, so the
helper returns `None` and the column is omitted rather than printed as zeros.

The alignment is the thing to preserve: the helper indexes `full_reflist` /
`full_reflist2` with positions derived from `full_reflist_6d`, so the three
arrays must stay row-aligned (both assignment sites set all three together) and
the caller's order must survive — candidates arrive sorted by ψ error, not in
list order. Test: `DMSAnalysis/tests/test_perp_strengths.py`, which pins that
and drives both identify paths on a real icosahedral config.

The **|q⟂| ≤** slider in the *Reflist* box attacks the density problem from the
other end: it narrows the pool the identify will *consider* to the reflections
whose perpendicular component is small enough to be producing a visible line.
The count under it reads `identify pool: 2366 / 15624`.

**It narrows the identify only.** The generated list, the overlay slice and the
fit all keep every reflection. That is deliberate, and it is what makes the
slider usable: filtering the list meant a full `_regenerate_reflist` per slider
event — at Depth 3 that is ~118k reflections reprojected and the overlay engine
rebuilt, ~96 ms, on every one of a 2000-step slider's events. The cut is now a
compare against `_qperp_all`, a norm array cached once per regeneration, at
~0.02 ms. So the slider acts while it is being dragged, which is the only way it
is worth having: you are looking for the cut that leaves the right candidate and
drops the rest, and that is a thing you find by moving it.

**The drawn set is capped, and is deliberately not nested.** Only
`GEO_MAX_DRAWN` (10) arcs are traced, and the ranking is taken over the surviving
pool, so loosening the cut lets better-scoring candidates in and pushes drawn
ones out. The *pool* nests as the cut loosens and so do the candidates within
`psi_tol` — it is only the display that churns. Ranking over the unfiltered
candidates instead would make the drawn set nest, but it would show almost
nothing exactly when the cut is working: on the example config at Depth 2, a cut
leaving 73 candidates draws 10 lines this way and would draw **1** the other,
because the best-scoring candidates are mostly the high-|q_perp| reflections the
cut exists to remove. The pick label therefore states the cap — `showing 10 of
73`, or `9 candidates` once it stops binding — since the churn is only
confusing while the cap is invisible.

Dragging also re-ranks **the candidates already on screen**, through a 120 ms
debounce. `_show_geo_candidates` is shared by the three-click search and the
re-filter, so a drag shows exactly what fresh clicks would. Candidate arcs are
cached by reflection row and hidden rather than destroyed — each costs a
one-reflection imcalc to trace, so a candidate that leaves the drawn set and
comes back as you move the slider is redrawn for free. An arc that gets selected
(or removed) leaves that cache, so a later re-filter cannot hide a reflection
that is in the selected list.

The rest of its semantics:

* **The ceiling is absolute; the range is not.** |q_perp| does not depend on
  Depth, so a cut you have set keeps meaning the same thing when Depth grows the
  set around it — the range extends past it rather than resetting it.
* **Untouched means off.** The cut is held as `None` until dragged, which pins
  the slider to the top on every regeneration. Without that, an untouched slider
  would inherit the previous set's maximum as a cutoff the moment Depth went up.
  Dragging back to the top restores `None` rather than recording that number.
* **The travel is monotonic.** The slider ranges from the *smallest* |q_perp|
  present to the largest, not from zero, so every position leaves something to
  identify against and dragging down never brings candidates back. Ranging from
  zero gave the bottom a dead zone where the ceiling passed nothing, which then
  fell back to the whole list — so the slider showed the same candidates at the
  bottom as at the top and different ones in between. A ceiling below everything
  (reachable only by carrying a cut onto a different reflection set) now keeps
  the smallest shell, never the whole list.
* **Equivalents are not split by float noise.** Symmetry-equivalent reflections
  have mathematically equal |q_perp| but norms differing in the last bits, so
  the compare (`_qperp_le`) carries a relative tolerance — far above that noise,
  far below any real separation between shells. Without it the bottom of the
  slider admits one member of a star and drops another.
* **Nothing to cut on** — a conventional crystal, or norms not yet cached —
  disables the slider and stands the mask down. `_qperp_usable` is the single
  predicate behind the mask, the enabled state and the label, so the three
  cannot disagree; the label used to ask the widget instead, which described the
  crystal the window was *built* on rather than the mode it is in. The all-zero
  case would also give `FloatSlider` a zero-width range, which raises on its
  first `setValue`; `_qperp_range_max` never returns zero.

`_qperp_mask` stands down if the cached norms fall out of step with the list,
since masking by position would then read another reflection's row. Like Depth,
Max N and Thresh, the cut is live UI state and is not saved in the session.
Test: `DMSAnalysis/tests/test_qperp_filter.py`.

Missing scan data never stops the slider from opening. If the config's `.dat` or
its detector image cannot be read (a beamline path that does not exist on this
machine, data on another disk, …), the app starts on placeholder metadata
(`slider.fallback_experiment` — lattice from `crystal.initial_guess_base`, energy
set to put the primary reflection at a 20° Bragg angle) and a blank frame, lists
what was missing in a startup dialog, and lets the user browse to the real file
with the Scan loader (**Browse…** → **Load**), which replaces all of it. A scan
whose `.dat` reads but whose image is absent loads too — metadata is applied and
the blank frame is kept, noted in the status line. The config's own
`geometry.hkl` / `geometry.psi` still win at **startup** (a config is a saved
refinement); only a scan *load* seeds. `fit.py` (batch) still fails
loudly on missing data.

`tripfit.py` is a separate, image-free batch app: it refines a lattice — a
conventional one, or an icosahedral quasicrystal's `a` and phason strain (see
*Quasicrystals* under the tripfit configuration) — by driving the three Kossel
lines of one or more secondary-reflection
triples to a common point on the stereographic projection (the Renninger
triple-intersection / multiple-diffraction geometry). It needs no detector image
— only the reflection geometry — and is the sensitive probe for the small
lattice distortions of pseudo-symmetric crystals (see the pseudo-cubic
re-indexing note under *Conventional crystals*). `tripslider.py` is its
interactive pyqtgraph GUI (dark theme, matching `slider.py`): drag the free
lattice / ψ sliders and watch each triple's Kossel lines and its residual update
live on a stereographic panel, switch crystal type from a dropdown (the slider's
list, the icosahedral quasicrystal types included), pick the
pseudo-cubic re-indexing from the **Pseudo-cubic** dropdown (the 12 Table-1
matrices, `pseudocubic_transform`, exactly as in `slider.py` — selecting one
re-indexes the primary hkl, the azimuthal reference and every triple's reflection
list live as `hkl' = M·hkl`, so you can read off which indexing gives the lowest
triple-intersection residual), and run the optimiser in the background with
**Fit** / **Stop**. Each slider the fit can refine carries a tick box, as in
`slider.py`: unticking it locks that parameter at its current value, so **Fit**
optimises only the ticked ones. Unlike the slider, the locks are saved
(`computation.locked`) and `tripfit.py` honours them. The **Triple intersections**
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

That subset is `ts_quasi.imcalc_param_indices(bravais, detopt, energyopt)`, the
one table `slider.py` (`reduced_slots_for`) and `fit.py` both use. It is derived
from the offsets `dmsfit_ico_hkl.imcalc` reads in each branch, and must stay
that way. The slider and fit.py used to carry hand-written copies that had
drifted from the engine. `cubic_no_strain` and `calibrate` with the detector or
energy refined were a slot short, so every evaluation raised IndexError. In the
slider that happened inside the overlay worker thread, which only printed it,
so the DMS lines stopped following the sliders after a switch to *Cubic (no
strain)*. `icosahedral_fixed_a` with the detector refined read the energy in as
dzrot. A failed overlay update now shows in the status line
(`UpdateWorker.failed`). Test: `DMSAnalysis/tests/test_imcalc_param_indices.py`,
which round-trips every mode × detopt × energyopt through `inputarray`.

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

### The objective's cost, and why its fast paths are exact

Every optimiser in `fit.py` and the slider scores through `dmsfit_ico_hkl.fit`:
`imcalc` (geometry → simulated image → smear) then, per ROI, `msroi` + a peak
fit. The vectorised geometry was never the cost. Two things were, and both now
have replacements that reproduce the old result **bit for bit** — the optimisers
see the last bit of the objective, so "close" would change fits:

* **The smear.** `ndimage.convolve` of the whole frame with the 15×15 Gaussian
  was ~80% of an evaluation (~0.25 s per Pilatus 2M frame, ~1.2 s at zoom 2),
  though only the line pixels are non-zero. `convolve_binary` adds their
  weights directly. It is exact because ndimage sums `input*weight` over the
  footprint in C order of the flipped kernel, skipping `|w| <= DBL_EPSILON`, and
  on a binary image each term is `+0.0` (a no-op) or exactly `w` — so adding
  `w` one kernel offset at a time, mirrored at the edges for `reflect`, is the
  same sequence of additions.
* **The ROIs.** `msroi` found each ROI's pixels with an `np.where` over a whole
  detector-sized kernel plane on every evaluation. They depend only on kernel,
  width and image shape, so `msroi_sampler` works them out once and the engine
  caches one per ROI.

Measured on synthetic 1679×1475 scenes (10–12 ROIs): ~300 ms → ~8–10 ms per
evaluation at zoom 1, ~1.2 s → ~7 ms at zoom 2, with objective values, residuals,
the simulated image and `dmsindex` identical. What is left is spread thinly over
the geometry, the smear and the per-ROI `curve_fit`s. Anything that changes the
smear or the ROI sums must keep
`DMSAnalysis/tests/test_fit_objective_speedups.py` passing — it compares with
`array_equal`, not a tolerance.

**The ROI kernel is sparse.** `roibuilder_ico_hkl` returns a `RoiKernel`: each
plane's lit pixels, not a dense `(H, W, n)` stack (0.6 MB instead of 832 MB for
the 42 ROIs of `data/slider_state_913232_*`). It keeps the two things callers
used — `kernel.shape[2]` and `kernel[:, :, i]` (rebuilt exactly) — plus
`kernel[:, :, selection]`, `.copy()`, and `np.asarray(kernel)` for the dense
stack; `roi_pixels(kernel, i)` gives `np.where(plane > 0)` for either form
without building the plane. `multiroifit`/`multiroifit2` return their ROI mask
stack the same way. Only the degenerate "No ROIS used!" fallback still returns a
dense all-ones array, exactly as before.

**Multi-start fits run in processes, not threads.** The slider ran its parallel
starts (COBYLA, Nelder-Mead, Powell, L-BFGS-B, TNC) on threads. Those overlapped
only while the objective sat in GIL-free C — the old full-frame convolve — so
once the smear stopped dominating, the starts took turns. Each start now runs
`ts.run_scaled_start` in a joblib worker process; the sparse kernel is what makes
sending the engine to a worker cheap (~10 MB, mostly the image). Stop reaches the
workers through `ts.StopFlag` (a flag file — a `threading.Event` cannot cross a
process boundary, and a `multiprocessing.Manager` event needs an auth key
joblib's workers lack), and `FitStopped` lives in `ts_quasi` so the class raised
in a worker is the one the slider catches. Anything a worker executes must live
in `ts_quasi`, never `slider.py`, or unpickling it imports the GUI.

On the real session (scan 913232, 42 ROIs), the slider's 4-start COBYLA took
149.8 s before any of this, 34.4 s with the fast objective on threads, and 9.0 s
with processes — every start taking the same number of evaluations to the same
parameters in all three. In the app itself (the session restored in the real
slider, fit through `FitWorker`): 152.5 s → 9.0 s, the same χ², the same refined
vector, and a `Result.txt` that differs only in its timestamp and elapsed time;
Stop ends a running fit within ~0.2 s.

`DMSAnalysis/tests/test_fitworker_processes.py` drives the real `FitWorker`:
the result must equal the same starts run in-process, and Stop must end a fit
mid-run. It connects to the worker's signals with `DirectConnection` rather than
pumping the event loop — `processEvents()` in a GUI test also runs the slider's
queued startup task, whose modal missing-scan prompt (`_prompt_missing_scan`)
blocks forever offscreen when the test config's data is not on the machine.

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

### The window layout persists

The slider is three panels — detector image | controls | integrated curves —
divided by one `QSplitter`. Where the dividers are dragged, and the window's own
geometry, are remembered between sessions in Qt's per-user settings
(`QSettings('DMSAnalysis', 'slider')` → `~/.config/DMSAnalysis/slider.conf` on
Linux), not in the auto-saved session: the layout is how the window is set up,
not what is being analysed, so it comes back whether or not the user resumes the
previous session. Written on every drag (`splitterMoved`) as well as in
`closeEvent`, so a killed app does not lose it — which is how a GUI usually ends
on a beamline.

The splitter stores absolute pixel sizes, so the window geometry is restored
alongside it; restoring one without the other gives panels that do not match the
window they are in. A geometry that lands on no attached screen (the display
setup changed) is discarded rather than applied, so the window cannot come back
invisible (`_restore_layout` / `_on_a_screen`). Test:
`DMSAnalysis/tests/test_layout_persistence.py`.

### Every fit leaves a run record

A completed fit in the slider writes, without being asked,

```
Processing/<scannum>_dp<datapoint>_<YYYYMMDD-HHMMSS>_<method>/
    Result.txt              # the solution, and the recipe to rerun it
    IM_<scan>_dp<dp>.png    # the detector image with the DMS lines over it
    PLOT_<scan>_dp<dp>.svg  # the integrated ROI curves, as vector art
```

Seconds are in the stamp because a fit takes seconds: two fits in one minute
must not share a folder. The method comes last, as in `fit.py`'s batch
directories, so a scan's runs still sort chronologically. The record is written *after* the post-fit curve
rebuild lands (`_flush_fit_snapshot`, called from `_on_build_done`), so the
curves in the SVG are the refined ones, not the ones the fit started from; if
that rebuild never starts or fails, it is written immediately instead. The
status line gains `→ Processing/<folder>`.

`Result.txt` carries what the fit *did* and what it was *given*: the residual,
method and elapsed time; the geometry (hkl, psi, azir, beam centre, scan
energy); all 24 refined slots by name (`IG_SLOT_NAMES`) plus a paste-ready
`initial_guess = np.array([…])`; the per-ROI target/simulated centres and their
residuals; and a **fit setup** block — the starting guess, which slots were
free, their bounds, the optimiser, parallel starts, points, tolerance, ROI
width, peak and curve methods — so the run can be set up again. That block is
captured in `_do_fit` at launch (`self._fit_setup`), not reconstructed
afterwards: by the time the fit reports back the sliders hold the *refined*
values, so the starting point is no longer on screen anywhere.

The SVG is drawn with matplotlib (`Figure` + `FigureCanvasSVG` directly — no
pyplot, which would pull a second GUI backend into the running Qt app) from the
same arrays the on-screen panels hold, sharing `sim_curve_scale` with
`_draw_sim_lines` so the exported curves cannot drift from the drawn ones.

**No Fit.** The last entry in the algorithm combo (`NoFit`) runs no optimiser:
**Fit** scores and renders the geometry currently on the sliders and writes its
run record, so the current guess can be exported — overlay, curves, Result.txt —
without refining anything. It needs no enabled fit parameters, `Result.txt`
says `no optimiser ran — the guess below was scored as-is` and lists what
*would* have been refined, and the folder is tagged `…_NoFit` like any other
method. In `FitWorker` it is a branch that skips straight to the final
scoring/rendering the optimising paths end in, so the result dict it emits is
the same shape as a real fit's.

**Save fit snapshot → Processing** writes the same three files again in a folder
of its own, plus the reproducibility extras the manual save always had:
`slider.py`, `ts_quasi.py`, `config_<scan>.json` and `res.x.txt`. Test:
`DMSAnalysis/tests/test_fit_snapshot.py`.

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
| `computation` | `bravais` (a `ts_quasi.TRIPFIT_SYSTEMS` name: a conventional system, or `icosahedral` / `icosahedral_fixed_a` / `cubic_no_strain`), `resolution` (Kossel-line sampling during the fit — how the lines are drawn and where each cone's plane is fitted from, *not* something the residual depends on; see *The crossings are solved on the cones* below), `opt_method` (any name in `ts_quasi.TRIPFIT_METHODS` — see *Optimiser methods* below), `tolerance`, `boundrange` `[lo,hi]` added to the guess for bounds, optional `rr` (azimuthal pre-rotation, deg; conventional only), `locked` (parameter names held at their starting value by the fit — any of `a b c alpha beta gamma a11 … a33`, `ts_quasi.TRIPFIT_PARAM_NAMES`; the GUI's unticked slider boxes), `bh_niter`, `de_strategy`, `fd_step` (finite-difference step for the gradient methods; omit/`null` to use SciPy's default), `pseudocubic_transform` (1–12, GUI only, conventional only — the Table-1 pseudo-cubic indexing matrix applied to the base indexing at load, same key/semantics as `fit.py`/`slider.py`; 1 = identity), and (GUI only) `live_resolution` for the interactive overlay |
| `crystal` | `initial_guess` — full 6-element lattice `[a,b,c,α,β,γ]`; for a quasicrystal type also `phason` (9 elements, a11…a33, default zero) and `tau_approx` (default 55/34, the slider's). Only the type's free slots are refined |
| `intersections` | list of triples, each `{label, reflist (3×3 h k l, or 3×6 6D indices for a quasicrystal type), energy, target, enabled}` — the three secondary reflections whose Kossel lines must meet. Which crossing of each line pair to score is chosen automatically: the engine takes the tightest (mutually-closest) triple, so the selection stays consistent and the residual doesn't jump as the lattice varies. `enabled` (default `true`, the GUI's per-row tick box) drops a triple from the objective while still plotting it, dimmed. (A legacy `intercepts` index vector, if present, is ignored.) |
| `display` | `lim`, `dpi` — plot settings |

The lattice constraints reuse `ts_quasi.lattice_free_slots` / `expand_lattice`
(the same table-driven layer as the image fit), so e.g. `rhombohedral` refines
`[a, α]` only. The objective is the summed triple-intersection residual over all
groups; `fit=0` just evaluates and plots at the initial guess. The engine
(`ts_quasi.kosscalc`, `stereoproj`, `intersections`, `tripfit`) is ported from
the standalone `calcms/ts_light.py` so the whole workflow lives in the package.
See `configs/tripfit_rhombohedral_PMN_PT_example.json`.

### Quasicrystals

The icosahedral types are handled exactly as `slider.py` handles a quasicrystal,
and that lives in the engine (`ts_quasi.tripfit`), so the batch app and the GUI
get it identically. A triple's reflections are 6D indices, projected with
`Projection6dArrayApproximant(ref, tau_approx)`; the cell is `[a,a,a,90,90,90]`;
and each reflection is `par + M·perp` for the phason matrix `M` — the
`PhasonDistoArray` step of `dmsfit_ico_hkl.imcalc`. The primary `hkl` and `azir`
are the non-integer vectors the slider carries. The parameter vector grows to 15
elements, `[a,b,c,α,β,γ, a11…a33]` (`tripfit_params`), and the type picks what is
refined (`tripfit_free_slots`), matching the image fit's modes:

| `bravais` | Refined | Held |
|-----------|---------|------|
| `icosahedral` | `a` + phason | — |
| `icosahedral_fixed_a` | phason | `a` (still shown as a slider, as in `slider.py`) |
| `cubic_no_strain` | `a` | phason at zero |

A conventional system keeps exactly its old slots and reduced vector, so existing
configs and residuals are unchanged.

`tau_approx` defaults to `ts_quasi.TAU_APPROX` = 55/34, which `slider.py` also
reads — one source. `fit.py` still projects with `Projection6d`, which uses the
exact golden ratio, so the image batch fit does not index a quasicrystal quite
as the slider does; tripfit follows the slider.

In the GUI the **Pseudo-cubic** combo is disabled on a quasicrystal type (a 3×3
matrix cannot re-index 6D indices) and the table's reflection columns take six
indices. h k l and 6D reflections cannot be converted into each other, so
switching the crystal type across the two families sets the current triples
aside and brings back the ones last used with the other family (a starter triple
the first time). `rr` is refused for a quasicrystal type. See
`configs/tripfit_icosahedral_AlPdMn_example.json`: its geometry, `a` and phason
are the slider refinement of scan 913123, and its triples are ones predicted to
meet there — a worked start, not measured intersections. Test:
`DMSAnalysis/tests/test_tripfit_quasi.py`.

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

### The crossings are solved on the cones, not on the sampled lines

`kosscalc` sweeps each secondary reflection's exit direction a full 360° about
that reflection's own axis, so a Kossel line is a **cone** of unit vectors —
exactly a circle on the sphere, `{v : |v| = 1, v·n = c}` — which the sweep only
samples. Crossing the sampled polylines with shapely (`ts_quasi.intersections`)
therefore crosses *chords*: each crossing is off by about the chord sagitta
(~1/steps²) and `triple_spread`, a squared distance, by ~1/steps⁴.

That made the residual a function of `computation.resolution` rather than of the
lattice, and left the optimiser free to reach machine zero by tuning the cell to
the polygon instead of closing the triple — on the rhombohedral example one
triple scored 8.1e-17 at resolution 1000 where its crossings are really 2.2e-11
apart (over 100/400/1000/4000 steps: 1.0e-6, 6.3e-10, 8.1e-17, 1.2e-11). It is
also why `tripslider` reported one fit as two numbers: the status bar's came
from the fit resolution and the control panel's Σ from the live one.

`tripfit._intercepts` now fits each locus's plane (`ts_quasi.sphere_circle_plane`,
one SVD over the sampled points), intersects the two cones of a pair in closed
form (`ts_quasi.cone_pair_directions`), and projects the directions through
`stereoproj` — so the analytic crossings and the drawn lines cannot end up on
different conventions. The residual is then a property of the lattice alone:
identical to 9 significant figures over 50 → 4000 steps on both example configs,
with the loci meeting their fitted circles to ~1e-15.

**`resolution` therefore buys nothing but a smoother plot.** A low value is free
— at `resolution = 50` the rhombohedral example fits ~6× faster and, in that
run, to a slightly *better* minimum than at 1000. **Residuals from before this
change are not comparable**, including those in older `Processing/` snapshots.

A locus that is not usable as a circle — deviation above `KOSSEL_CIRCLE_TOL`
(1e-9; a real one sits at ~5e-16), points that fix no plane, or a coaxial pair —
falls back to the sampled-polyline crossing, and `tripfit.circle_residual` is
then `None`. A pair that is well posed but simply does not meet still raises and
scores 500, as before: that is an answer about the geometry, not a reason to drop
to a cruder method. `tripfit.py` prints the worst circle deviation (`circle fit
:`, and `circlefit` in `Result.txt`); the GUI names any triple that fell back
next to the Σ residual. Test:
`DMSAnalysis/tests/test_tripfit_intercepts.py`, which pins the residual across
resolutions, checks each crossing against the two cones it solves, exercises the
fallback, and drives the real GUI to assert its live label and its fit objective
agree.

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
PyQt6  pyqtgraph   (for the slider GUI; PyQt5 also works)
```

**Qt binding.** Every GUI module imports Qt from `DMSAnalysis/qt.py`, never
from `PyQt5`/`PyQt6` directly. It prefers PyQt6, which has arm64 wheels, and
falls back to PyQt5; `PYQTGRAPH_QT_LIB` forces one. It must be imported before
`pyqtgraph`, which picks its binding from that variable. Write to the API the
two share:

* **Scoped enums**: `QtCore.Qt.AlignmentFlag.AlignLeft`, not `QtCore.Qt.AlignLeft`.
  PyQt6 has no unscoped names, and PyQt5 5.15 accepts the scoped form.
* **`exec()`**, not `exec_()` (it is gone in PyQt6). The headless test harness
  therefore stubs `QApplication.exec`.
* **`QShortcut`** is imported from `.qt`, because it moved to QtGui in Qt6.
* **`stateChanged` carries an int** in PyQt6, and an int never equals
  `Qt.CheckState.Checked`, so wrap it with `qt.check_state(s)`. `toggled(bool)`
  has no such problem.

The GUI tests run offscreen on a 2560×1440 screen
(`tests/offscreen_screen.json`). Qt6's `restoreGeometry` clamps the window to
the 800×800 default screen, and the layout-persistence test fails on that.

Every dependency is installable from PyPI; nothing here needs `cctbx`. The
CIF-driven reflection list (`loadcif`, `flags.autoreflist`, `paths.cif_file`)
was the only thing that did, and it had been dead code for some time — its
`iotbx` import was commented out, so calling it raised `NameError`. Reflection
lists come from `crystal.reflist_hkl` / `crystal.ref_6d`, the depth-based
generator (`hklgen_3d` / the slider's **Auto reflist**), or the slider's
**Geo 3-click** identify.
