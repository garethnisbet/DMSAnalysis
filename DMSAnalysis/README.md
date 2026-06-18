# DMSAnalysis/ts_quasi.py — Module Documentation

**Author:** Dr Gareth Nisbet, Diamond Light Source
**License:** Apache 2.0
**Purpose:** X-ray diffraction geometry calculations, multiple scattering (MS) simulation, and detector-image fitting for crystalline and quasicrystalline materials.

---

## Constants

| Name | Value | Description |
|------|-------|-------------|
| `TAU` | `0.5 + 0.5*√5 ≈ 1.618` | Golden ratio, used in quasicrystal icosahedral indexing |

---

## Crystallographic Utilities

### `fib(n)` → `list`
Returns the first `n` Fibonacci numbers. Used in quasicrystal indexing sequences.

---

### `class bmatrix(lattice)`
Converts lattice parameters to the crystallographic **B-matrix** (Cartesian reciprocal space).

`lattice = [a, b, c, alpha, beta, gamma]` (angles in degrees)

| Method | Returns | Description |
|--------|---------|-------------|
| `bm()` | `np.matrix` (3×3) | The B-matrix |
| `ibm()` | `np.matrix` (3×3) | Inverse B-matrix |
| `mt()` | `np.matrix` (3×3) | Metric tensor (direct space) |
| `rmt()` | `np.matrix` (3×3) | Reciprocal metric tensor |
| `volume()` | `float` | Unit cell volume |
| `reciprocal_parameters(lp2=[])` | `np.ndarray` (6,) | Reciprocal lattice parameters [a*, b*, c*, α*, β*, γ*] |
| `direct_matrix()` | `np.ndarray` (3×3) | Direct-space orientation matrix |

---

### `class mmatrix(lattice)`
Alternative metric matrix; transforms fractional to Cartesian coordinates using a different convention from `bmatrix`.

| Method | Returns | Description |
|--------|---------|-------------|
| `mm()` | `np.matrix` (3×3) | The M-matrix |
| `imm()` | `np.matrix` (3×3) | Inverse M-matrix |

---

### `class rotxyz(u, angle)`
Rotation matrix about an arbitrary axis using Rodrigues' formula.

- `u` — rotation axis vector `[ux, uy, uz]`
- `angle` — rotation angle in degrees

| Method | Returns |
|--------|---------|
| `rmat()` | `np.matrix` (3×3) rotation matrix |

---

### `class dhkl(lattice, hkl)`
Calculates d-spacings for a list of reflections using the reciprocal metric tensor.

| Method | Returns |
|--------|---------|
| `d()` | Array of d-spacings (Å) |

---

### `class interplanarangle(lattice, hkl1, hkl2)`
Calculates interplanar angles (degrees) between pairs of reflections.

| Method | Returns |
|--------|---------|
| `ang()` | Array of angles in degrees |

---

### `class bragg(lattice, hkl, energy)`
Calculates the Bragg angle θ for a reflection at a given photon energy (keV).

| Method | Returns |
|--------|---------|
| `th()` | Bragg angle(s) in degrees |

---

## HKL Generation

### `class hklgen(depth)`
Generates all 3D integer hkl indices within `[-depth, +depth]` in each direction.

| Method | Returns |
|--------|---------|
| `v()` | `np.ndarray` (N×3) of integer hkl triples |

---

### `class hklgen_ico(depth)`
Generates all 6D icosahedral index vectors `(n1, n2, n3, n4, n5, n6)` within `[-depth, +depth]`.
Used for quasicrystal diffraction indexing.

| Method | Returns |
|--------|---------|
| `v()` | `np.ndarray` (N×6) of integer 6D indices |

---

### `class hklgen_ico_5d(depth)`
As `hklgen_ico` but generates 5D index vectors.

| Method | Returns |
|--------|---------|
| `v()` | `np.ndarray` (N×5) of integer 5D indices |

---

### `class vfind(vlist, v)`
Finds row indices in `vlist` that exactly match rows in `v` (3-component vectors).

| Method | Returns |
|--------|---------|
| `vindex()` | List of matching row indices |

---

### `class vfind_ico(vlist, v)`
As `vfind` but requires 6-component matches (for icosahedral indexing).

---

## Multiple Scattering Geometry

### `class calcms(lattice, hkl, hklint, hkl2, energy, azir, F=[], F2=[])`
Core multiple scattering geometry engine. For a primary reflection `hkl` and a list of secondary reflections `hkl2`, computes the azimuthal angles (ψ) at which each secondary beam satisfies the Ewald sphere condition simultaneously with the primary.

**Key parameters:**
- `lattice` — `[a, b, c, α, β, γ]`
- `hkl` — primary reflection `[h, k, l]`
- `hklint` — intermediate hkl (= `hkl + hkl2`)
- `hkl2` — list of secondary reflections (N×3)
- `energy` — photon energy in keV
- `azir` — azimuthal reference vector `[h, k, l]`
- `F`, `F2` — optional structure factor arrays for primary and secondary reflections

| Method | Returns | Description |
|--------|---------|-------------|
| `full()` | N×7 array | `[hkl2, ψ1, ψ2, θ_Bragg, energy]` per secondary reflection |
| `psi()` | array | Both ψ solutions concatenated |
| `bragg()` | array | Bragg angles |
| `pol(polv)` | array | Polarisation factors `[σ, π, P]` for incident polarisation vector `polv` |
| `pol2(polv)` | array | Two-beam polarisation factor (secondary scattering) |
| `polfull(polv)` | array | `full()` + `F·F2·P2` intensity |
| `polfull2(polv)` | array | `full()` + full polarisation matrix |
| `sfonly()` | array | `full()` + `F·F2` (structure factors only) |
| `sf1only()` | array | `full()` + `F` |
| `pol1only(polv)` | array | `full()` + single polarisation factor |
| `pol2only(polv)` | array | `full()` + two-beam polarisation factor |
| `geometry()` | array | Alias for `full()` |
| `bvects()` | tuple | Secondary beam vectors (two ψ solutions) |
| `bvects2()` | tuple | Tertiary beam vectors |
| `angs()` | tuple | Angles between k₀ and secondary beam vectors |
| `trv()` | tuple | Transformed and rotated secondary reciprocal vectors |
| `trvt()` | tuple | Transformed and rotated tertiary reciprocal vectors |
| `ov()` | matrix | Original hkl2 list |
| `orig()` | matrix | Reciprocal space origin vector |
| `kv()` | float | Wavenumber k₀ |
| `SF()` | matrix | Structure factors F |
| `SF2()` | matrix | Structure factors F2 |

---

### `vor_euler(mu, eta, chi, phi, ub, reflist)`
Applies a Eulerian cradle rotation sequence (Voigt geometry: μ, η, χ, φ) to a list of reciprocal vectors using the UB matrix. Returns transformed vectors.

---

### `kosscalc1(lattice, energy, ref, startval, endval, steps)`
Computes beam vectors tracing a Kossel cone around a single reflection over an azimuthal range. Returns an array of 3D vectors normalised to k₀.

---

### `kosscalc(lattice, energy, ref1, ref2, azir, psi, startval, endval, steps)`
Computes Kossel lines for a list of secondary reflections `ref2` relative to a primary reflection `ref1`. Returns an array of `[x, y, z, ψ, θ]` per step.

---

## Detector Geometry

### `class decttrans(im, originx, originz, eta, psi, detxrot, detyrot, detzrot, detdistance, pxsize, geom, ...)`
Maps a 2D detector image to reciprocal-space / angular coordinates.

| Method | Returns | Description |
|--------|---------|-------------|
| `v()` | `(psi2, theta)` | Azimuthal and polar angles per pixel |
| `thpsi()` | `(psi_img, theta_img)` | As 2D arrays matching image shape |
| `delgam()` | `(delta_img, gamma_img)` | Delta/gamma angles as 2D arrays |
| `hkls()` | N×4 array | `[h, k, l, intensity]` per pixel |
| `mesh()` | meshgrid | x–z pixel coordinate meshgrid |
| `irmat()` | matrix | Inverse detector rotation matrix |

---

### `class pilkhlrange(lattice, hkl, energy, botangle, topangle)`
Determines the hkl range accessible to a detector given its angular acceptance.

| Method | Returns | Description |
|--------|---------|-------------|
| `hklrange()` | 2×3 array | Lower and upper hkl limits |
| `hklscan(numsteps)` | N×3 array | Linearly spaced hkl scan path |

---

### `dms2px(detv1, detv2, o, v)`
Projects 3D scattering vectors `v` onto a detector plane defined by two edge vectors `detv1`, `detv2` and sample origin `o`. Returns 3D intersection coordinates.

---

### `psith2v(psi, th)`
Converts azimuthal (ψ) and polar (θ) angles in degrees to unit Cartesian vectors `[X, Y, Z]`.

---

## Simulation & Fitting

### `class dmscalc(...)`
Computes a simulated detector image of multiple-scattering streak positions for a given set of lattice parameters. Used as the objective function in lattice refinement.

| Method | Returns | Description |
|--------|---------|-------------|
| `imcalc(inputs)` | — | Computes `self.imsim` (simulated image) and `self.dmsindex` (pixel hit list) |
| `full(inputs)` | `(score, imsim, dmsindex, imdata)` | Returns cross-correlation score, simulated image, pixel indices, and data |
| `roiindex(inputs)` | N×2 array | Pixel coordinates of predicted streak positions |

---

### `class dmsfit(...)`
Extends `dmscalc` with crystal-system-aware parameter unpacking and Gaussian-peak fitting along ROI line profiles. Supports Bravais lattice types: `triclinic`, `tetragonalA/B/C`, `monoclinicA/B/C`, `rhombohedral`, `orthorhombic`, `cubic`, `calibrate`.

| Method | Returns | Description |
|--------|---------|-------------|
| `imcalc(inputs)` | — | Builds simulated image from lattice + geometry parameters |
| `fit(inputs)` | `float` | Sum-of-squared residuals between predicted and observed peak centres |
| `full(inputs)` | `(score, imsim, dmsindex, imdata, inputarray)` | Full output including parameter array |
| `stats(inputs)` | `(residuals, covariances)` | Peak centre residuals and covariances |

---

### `class intercepting_vects(...)`
Finds scattering vectors that intersect the detector for a given set of crystal parameters.

---

### `class minimize_gauss(xdata, ydata)`
Minimisation wrapper: evaluates the sum-of-squared residuals for a Gaussian-plus-slope model.

---

### `class gaussfit(x, y)`
Alternative Gaussian fitting class using direct minimisation rather than `curve_fit`.

---

## ROI Builders
These functions build a stack of binary 2D kernel images, one per predicted multiple-scattering streak, used as regions of interest (ROIs) during fitting.

| Function | Description |
|----------|-------------|
| `roibuilder(args)` | Standard crystal ROI builder (2 ROIs per reflection) |
| `roibuilderInterp(args)` | As above with deduplication of overlapping ROIs |
| `roibuilder_ico(args)` | Icosahedral quasicrystal ROI builder (2 per reflection, with phason matrix) |
| `roibuilder_ico_hkl(args)` | As `roibuilder_ico` using `dmscalc_ico_hkl` |
| `roibuilder_ico_x3(args)` | Icosahedral ROI builder with 3 ROIs per reflection |

---

### `msroi(img, kernel, width)` / `msroi2(img, kernel, width)`
Extracts an intensity line profile perpendicular to a kernel streak in an image.

- **Returns:** `(sumvals, roi_coords)` — integrated intensity per slice and corresponding pixel coordinates.

---

## Peak Fitting Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gauss` | `(x, sigma, intensity, centre, bg)` | Gaussian model |
| `gaussS` | `(x, sigma, intensity, centre, bg, slope)` | Gaussian with linear slope |
| `gauss2` | `(x, σ1, σ2, I1, I2, c1, c2, bg)` | Double Gaussian |
| `SF_Scaler` | `(x, sigma, intensity, peakpos, background)` | Alias for Gaussian (structure-factor scaling) |
| `fitgauss` | `(xdata, ydata)` | Auto-estimates and fits a single Gaussian |
| `fitgaussS` | `(xdata, ydata, constraints)` | Fits Gaussian with slope and parameter bounds |
| `fitgauss2` | `(xdata, ydata, sig)` | Fits single or double Gaussian depending on width |
| `fitgauss1from2` | `(xdata, ydata, sig)` | Fits double Gaussian, returns the stronger component |
| `poly2/3/4` | `(x, ...)` | Polynomial models (degree 2, 3, 4) |
| `poly2/3/4fit` | `(xdata, ydata)` | Convenience wrappers for polynomial curve_fit |
| `expfunc` | `(x, a, b, c)` | Exponential model `a^(x·b) + c` |
| `com` | `(xdata, ydata, width)` | Centre-of-mass peak position |
| `spline` | `(xdata, ydata, samplingfactor, stiffness)` | Smoothing spline with peak location |
| `gconv` | `(xdata, ydata, kerneldef, sigma, interpfactor)` | Gaussian-kernel convolution peak finder |

---

## Image Processing

| Function | Signature | Description |
|----------|-----------|-------------|
| `gaussfilter` | `(image, siglow, sigblur, orderval)` | High-pass Gaussian filter (subtract low-pass, then smooth) |
| `makekernel` | `(func, size, sigma, sigma2)` | Creates 2D convolution kernel: `'gauss'`, `'lorentz'`, `'custom1'`, `'custom2'` |
| `fft2_filter` | `(img, lp_box_r, lp_box_c, hp_box_r, hp_box_c)` | 2D FFT bandpass filter; returns `(real, imag, abs, mask)` |

---

## Utility Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `loadcif(ciffile, energy)` | `(SF, reflist, lattice, crystal, sfc)` | Loads a CIF file via `iotbx` and returns structure factors (requires `cctbx`) |
| `reducebypsirange(mslist, psirange)` | array | Filters an MS list to a ψ window and merges both ψ solutions into one column |
| `uniquearray(inarray)` | array | Returns unique rows of a 2D array |
| `cmap()` | OrderedDict | Maps short colour-map keys to display names |
| `printfig(fformat)` | — | Saves current matplotlib figure to `/tmp/fig.<fformat>` and sends to printer via `lpr` |

---

## Supporting Classes

### `class sph2cart(sv)`
Converts spherical `[θ, φ, R]` coordinates to Cartesian `[x, y, z]`.

### `class allowedref(lattice, prim, spacegroup, energy)` *(deprecated)*
Filters a reflection list to systematically allowed reflections for space groups 161 and 225.

### `class loader(filename)`
Loads reflection data from a cctbx-format text file; returns `hkl()` and `F()` arrays.

---

## Dependencies

```
numpy, scipy, matplotlib, PIL (Pillow), shapely, itertools, subprocess, copy
```
Optional (commented out): `iotbx.cif`, `cctbx.sgtbx` — required only for `loadcif()` and `allowedref()`.

---

## Usage Example

```python
from DMSAnalysis import ts_quasi as ts

lattice = [5.431, 5.431, 5.431, 90, 90, 90]  # Silicon
hkl     = [1, 1, 1]                            # Primary reflection
energy  = 8.0                                  # keV
azir    = [0, 0, 1]                            # Azimuthal reference

# Generate secondary reflection list
refs = ts.hklgen(3).v()

# Compute multiple-scattering geometry
ms = ts.calcms(lattice, hkl, hkl+refs[0], refs, energy, azir)
print(ms.full())   # [hkl2, psi1, psi2, theta_Bragg, energy]

# Bragg angle
theta = ts.bragg(lattice, hkl, energy).th()

# d-spacing
d = ts.dhkl(lattice, [hkl]).d()
```
