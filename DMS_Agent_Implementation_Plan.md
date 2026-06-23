# Diffuse Multiple Scattering (DMS) Simulation Agent — Implementation Plan

## 1. Problem Statement

Bragg multiple scattering (MS) produces sharp features at well-defined reciprocal lattice points on the Ewald sphere. **Diffuse multiple scattering (DMS)** involves at least one scattering event into a continuum — thermal diffuse scattering (TDS), Huang scattering, phason-disorder streaks — and does **not** map to discrete reciprocal space positions. During automated diffraction experiments, DMS contamination is difficult to predict and can corrupt intensity measurements. An agent that can simulate DMS in real time would allow the experiment to navigate around contaminated regions of orientation/energy space.

## 2. Physics of Diffuse Multiple Scattering

### 2.1 What makes DMS different from Bragg MS

In Bragg MS, both scattering events satisfy exact Laue conditions — the problem reduces to finding pairs of reciprocal lattice points that simultaneously lie on the Ewald sphere. This is a discrete, finite search (implemented in `calcms_ico` and the vectorised `dmsfit_ico_hkl.imcalc()`).

In DMS, one (or both) scattering events is into a **continuous distribution**:

| Scattering path | First event | Second event | Character |
|-----------------|-------------|--------------|-----------|
| Bragg–Bragg MS  | Bragg peak  | Bragg peak   | Sharp spots/lines (current code) |
| Bragg–TDS       | Bragg peak  | Thermal diffuse | Broad arcs near MS lines |
| TDS–Bragg       | Thermal diffuse | Bragg peak  | Broad arcs near MS lines |
| Bragg–Phason    | Bragg peak  | Phason diffuse | Streaks along specific directions |

### 2.2 Thermal diffuse scattering model

For a monatomic crystal at temperature T, the first-order TDS intensity at wavevector transfer **Q** is:

```
I_TDS(Q) ∝ Σ_j |Q · e_j(q)|² / ω_j(q)² × [n(ω_j) + 1]
```

where the sum is over phonon branches j at reduced wavevector **q**, **e_j** is the polarisation vector, ω_j is the frequency, and n(ω) is the Bose occupation factor. For a Debye model this simplifies to:

```
I_TDS(Q) ∝ |Q|² × T / (M × ω_D²)
```

This can be parameterised as a single isotropic B-factor: `I_TDS(Q) ∝ B × |Q|²`, or for more fidelity, as an anisotropic displacement tensor.

### 2.3 Quasicrystal-specific diffuse scattering

For icosahedral Al-Pd-Mn, there are additional diffuse contributions:

- **Phason diffuse scattering**: arises from phason fluctuations (analogous to phonons but in perpendicular space). Intensity scales as `|Q_perp|² / |Q|²` and produces characteristic anisotropic patterns.
- **Structured TDS**: the phonon dispersion in quasicrystals is complex; pseudo-Brillouin-zone effects create non-trivial intensity modulations.

## 3. Architecture

### 3.1 Two-tier design

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT AGENT                          │
│                                                             │
│  Decision loop:                                             │
│    1. Propose next (ψ, energy, hkl) measurement             │
│    2. Query Tier 1: coarse avoidance map                    │
│    3. If flagged → Query Tier 2: detailed DMS simulation    │
│    4. If severity > threshold → re-plan                     │
│    5. Else → proceed with measurement                       │
└─────────────────┬────────────────────┬──────────────────────┘
                  │                    │
          ┌───────▼───────┐    ┌───────▼───────┐
          │   TIER 1      │    │   TIER 2      │
          │  Avoidance Map │    │  DMS Forward  │
          │  (precomputed) │    │  Model        │
          │               │    │  (on-demand)  │
          │  Lookup: O(1) │    │  Compute: ~1s │
          └───────────────┘    └───────────────┘
```

**Tier 1 — Coarse avoidance map** (precomputed):
- For each primary reflection (hkl), sweep ψ and energy over the experimental range.
- At each (ψ, E) point, compute a scalar DMS susceptibility: the integrated diffuse intensity that overlaps the detector region of interest.
- Store as a 2D lookup table per reflection. Threshold to get binary safe/unsafe regions.
- Fast enough for real-time steering (~μs per lookup).

**Tier 2 — Detailed DMS forward model** (on-demand):
- Full simulation: for a given (ψ, E, hkl, phason matrix), compute the 2D DMS intensity pattern on the detector.
- Used when the agent needs to assess whether a flagged region is actually problematic (the coarse map is conservative).
- Target latency: <1 s per evaluation (achievable with vectorised numpy).

### 3.2 Module structure

```
DMS/
├── calcms/
│   ├── ts_quasi.py          # Existing: Bragg MS geometry, fitting, ROIs
│   ├── loader.py            # Existing: scan file reader
│   ├── diffuse.py           # NEW: diffuse scattering intensity models
│   ├── dms_simulator.py     # NEW: DMS forward model (Tier 2)
│   └── avoidance.py         # NEW: precomputed avoidance map builder (Tier 1)
├── agent/
│   ├── dms_agent.py         # NEW: experiment steering agent
│   ├── strategy.py          # NEW: measurement planning strategies
│   └── config.py            # NEW: agent configuration and thresholds
├── scripts/
│   ├── build_avoidance_map.py   # NEW: offline map generation script
│   └── run_agent.py             # NEW: agent entry point
└── tests/
    ├── test_diffuse.py
    ├── test_dms_simulator.py
    └── test_avoidance.py
```

## 4. Implementation Steps

### Phase 1: Diffuse Scattering Intensity Model (`calcms/diffuse.py`)

**Goal**: a callable `I_diffuse(Q, Q_perp, params)` that returns diffuse intensity at arbitrary Q-vectors.

#### 4.1 `DebyeTDS` class

```python
class DebyeTDS:
    """Isotropic thermal diffuse scattering using Debye model."""

    def __init__(self, B_iso, temperature=300):
        """
        B_iso: isotropic displacement parameter (Å²)
        temperature: sample temperature (K)
        """

    def intensity(self, Q_mag):
        """Return TDS intensity at given |Q| values.
        
        I_TDS ∝ B × |Q|² × coth(ℏω_D / 2kT)
        
        For high-T limit: I_TDS ∝ B × |Q|²
        """
```

#### 4.2 `AnisotropicTDS` class

```python
class AnisotropicTDS:
    """Anisotropic TDS using displacement tensor U_ij."""

    def __init__(self, U_tensor, temperature=300):
        """U_tensor: 3x3 symmetric displacement tensor (Å²)"""

    def intensity(self, Q_vectors):
        """I_TDS(Q) ∝ Q^T · U · Q"""
```

#### 4.3 `PhasonDiffuse` class

```python
class PhasonDiffuse:
    """Phason diffuse scattering for icosahedral quasicrystals."""

    def __init__(self, K_phason, projection_6d):
        """
        K_phason: phason elastic constant ratio (K2/K1)
        projection_6d: Projection6d instance for parallel/perp decomposition
        """

    def intensity(self, Q_parallel, Q_perpendicular):
        """
        I_phason(Q) ∝ |Q_perp|² / (K1 |Q_par|² + K2 |Q_perp|²)
        
        Uses the elastic theory of Jarić & Nelson for icosahedral symmetry.
        """
```

#### 4.4 `CompositeDiffuse` class

```python
class CompositeDiffuse:
    """Combine multiple diffuse scattering contributions."""

    def __init__(self, models, weights=None):
        """models: list of diffuse model instances"""

    def intensity(self, Q_vectors, Q_perp=None):
        """Weighted sum of all model contributions."""
```

**Integration with existing code**: These models are independent of the Ewald sphere geometry — they take Q-vectors and return intensities. This keeps them testable and reusable.

### Phase 2: DMS Forward Model (`calcms/dms_simulator.py`)

**Goal**: given sample orientation and energy, compute the DMS intensity pattern on the detector.

#### 4.5 Core algorithm

For each primary Bragg reflection G_primary that is excited (near Laue condition):

1. **Compute the Bragg-scattered wavevector** k_scattered using existing `psith2v()`.

2. **Generate a fan of secondary Q-vectors**: the scattered beam can re-scatter into any direction. The set of accessible secondary wavevector transfers is the Ewald sphere centred on k_scattered. Parameterise this sphere by (θ₂, φ₂).

3. **For each secondary Q on the sphere**: evaluate `I_diffuse(Q₂)` from the diffuse model. If above threshold, project the doubly-scattered beam onto the detector via `dms2px()`.

4. **Accumulate** the DMS intensity on the detector image.

```python
class DMSSimulator:
    """Full DMS forward model for one sample orientation."""

    def __init__(self, lattice, energy, hkl_primary, psi, 
                 diffuse_model, detector_geometry, 
                 phason_matrix=None, projection_6d=None):
        """
        lattice: [a, b, c, alpha, beta, gamma]
        energy: photon energy (keV)
        hkl_primary: primary reflection indices
        psi: azimuthal angle (degrees)
        diffuse_model: CompositeDiffuse instance
        detector_geometry: dict with detdist, detvects, origin, rotations
        """

    def compute(self, n_theta=360, n_phi=180, threshold=1e-3):
        """
        Compute DMS intensity on detector.

        Algorithm:
        1. Bragg angle and scattered k-vector for primary reflection
        2. Ewald sphere of secondary scattering centred on k_scattered
        3. Sample sphere at (n_theta × n_phi) points
        4. For each point:
           a. Q_secondary = k_final - k_scattered
           b. I = diffuse_model.intensity(Q_secondary)
           c. If I > threshold: project k_final onto detector
        5. Return 2D detector image

        Returns:
            detector_image: 2D numpy array of DMS intensity
            metadata: dict with per-reflection breakdown
        """

    def compute_vectorised(self, n_theta=360, n_phi=180, threshold=1e-3):
        """
        Vectorised version using numpy broadcasting.

        The secondary Ewald sphere sampling is done as:
          theta_grid, phi_grid = meshgrid(...)
          k_final = k_scattered + k0 * [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
          Q_secondary = k_final - k_scattered  (broadcast over grid)
          I_diffuse = diffuse_model.intensity(Q_secondary)  (vectorised)
          
        Then project all above-threshold k_final vectors through dms2px in one call.
        """
```

#### 4.6 Integration with existing Ewald sphere code

The existing `calcms_ico` class computes Bragg MS by finding **exact** intersections of reciprocal lattice points with the Ewald sphere. For DMS, we replace the discrete search with a continuous sampling:

```
EXISTING (Bragg MS):                    NEW (DMS):
                                        
reflist → exact hkl points             secondary Ewald sphere → continuous sampling
     ↓                                       ↓
Ewald sphere intersection test          diffuse_model.intensity(Q) at each sample point
     ↓                                       ↓
psi1, psi2 angles (discrete)           weighted intensity (continuous)
     ↓                                       ↓
dms2px → pixel positions               dms2px → pixel positions (same function)
     ↓                                       ↓
binary mark on detector image           intensity-weighted mark on detector image
     ↓                                       ↓
Gaussian convolution (PSF)              Gaussian convolution (PSF) — same
```

The key reuse points from `ts_quasi.py`:
- `bmatrix` (line 36): lattice → Cartesian conversion
- `bragg` (line 172): Bragg angle calculation
- `rotxyz` (line 102): rotation matrices
- `psith2v` (line 481): (ψ, θ) → unit vector
- `dms2px` (line 473): reciprocal space → detector pixels
- `PhasonDistoArray` (line 825): phason strain application
- `Projection6d` (line 866): 6D → parallel/perpendicular decomposition
- `makekernel` (line 487): Gaussian convolution kernel

### Phase 3: Avoidance Map Builder (`calcms/avoidance.py`)

**Goal**: precompute a DMS susceptibility map over (ψ, energy) space for each primary reflection.

#### 4.7 Map builder

```python
class AvoidanceMapBuilder:
    """Precompute DMS susceptibility over orientation/energy space."""

    def __init__(self, lattice, hkl_primary, diffuse_model,
                 detector_geometry, phason_matrix=None):
        """Store fixed experimental parameters."""

    def build(self, psi_range, energy_range, psi_steps=360, energy_steps=50):
        """
        For each (ψ, E) grid point:
          1. Instantiate DMSSimulator
          2. Run compute_vectorised with coarse sampling (n_theta=90, n_phi=45)
          3. Integrate DMS intensity over detector ROI
          4. Store scalar susceptibility value

        Returns:
            AvoidanceMap with psi_grid, energy_grid, susceptibility_2d
        """

    def build_parallel(self, psi_range, energy_range, 
                       psi_steps=360, energy_steps=50, n_workers=4):
        """Parallel version using multiprocessing.Pool."""


class AvoidanceMap:
    """Lookup table for DMS susceptibility."""

    def __init__(self, psi_grid, energy_grid, susceptibility):
        """Store precomputed map."""

    def query(self, psi, energy):
        """Interpolated susceptibility at (ψ, E). O(1) lookup."""

    def is_safe(self, psi, energy, threshold=0.1):
        """Binary safe/unsafe decision."""

    def safe_psi_ranges(self, energy, threshold=0.1):
        """Return list of (psi_min, psi_max) safe intervals at given energy."""

    def save(self, filepath):
        """Serialize to .npz for fast reload."""

    @classmethod
    def load(cls, filepath):
        """Load from .npz file."""
```

#### 4.8 Precomputation budget

| Parameter | Coarse (steering) | Fine (validation) |
|-----------|-------------------|-------------------|
| ψ steps | 360 (1° resolution) | 3600 (0.1°) |
| Energy steps | 50 | 200 |
| Secondary sphere sampling | 90 × 45 = 4,050 | 360 × 180 = 64,800 |
| Points per map | 18,000 | 720,000 |
| Estimated time | ~30 s | ~20 min |
| Storage | ~150 KB (.npz) | ~6 MB (.npz) |

The coarse map is sufficient for real-time steering. The fine map can be built overnight for post-experiment analysis.

### Phase 4: Experiment Steering Agent (`agent/dms_agent.py`)

**Goal**: an agent that proposes safe measurement orientations and adapts in real time.

#### 4.9 Agent decision loop

```python
class DMSAgent:
    """Automated experiment steering agent for DMS avoidance."""

    def __init__(self, avoidance_maps, dms_simulator_factory,
                 measurement_plan, config):
        """
        avoidance_maps: dict of {hkl: AvoidanceMap}
        dms_simulator_factory: callable(hkl, psi, E) → DMSSimulator
        measurement_plan: list of target (hkl, psi_nominal, E) measurements
        config: AgentConfig with thresholds and strategy parameters
        """

    def next_measurement(self):
        """
        Propose the next safe measurement point.

        Algorithm:
        1. Pop next target from measurement_plan
        2. Query avoidance_map for target (ψ, E)
        3. If safe → return target unchanged
        4. If unsafe:
           a. Find nearest safe ψ within tolerance
           b. If found → return adjusted target
           c. If not found → try energy offset
           d. If still not found → flag as problematic, skip or queue for manual review
        """

    def assess_contamination(self, hkl, psi, energy):
        """
        Tier 2 assessment: run full DMS simulation.

        Returns:
            severity: float (0 = clean, 1 = fully contaminated)
            dms_image: 2D detector image showing DMS pattern
            affected_rois: list of ROI indices that overlap DMS
        """

    def suggest_alternative(self, hkl, psi, energy, 
                            psi_tolerance=5.0, energy_tolerance=0.05):
        """
        Find the nearest (ψ, E) that avoids DMS contamination.

        Uses avoidance map to find closest safe point within tolerance.
        Returns None if no safe point exists within constraints.
        """

    def run(self, beamline_interface):
        """
        Main agent loop for live experiment.

        while measurements remain:
            target = self.next_measurement()
            if target is None:
                break
            beamline_interface.move_to(target.psi, target.energy)
            data = beamline_interface.collect()
            self.log_measurement(target, data)
        """
```

#### 4.10 Strategy patterns

```python
class NearestSafeStrategy:
    """Move to the closest safe ψ value."""

    def find_safe_point(self, avoidance_map, target_psi, target_energy, tolerance):
        safe_ranges = avoidance_map.safe_psi_ranges(target_energy)
        # Find closest point in any safe range to target_psi
        # Return None if all safe points are beyond tolerance


class EnergyOffsetStrategy:
    """Shift energy slightly to move DMS features away from ROI."""

    def find_safe_point(self, avoidance_map, target_psi, target_energy, 
                        psi_tolerance, energy_tolerance):
        # Search energy grid at fixed ψ
        # Small energy shifts (tens of eV) can move DMS features significantly


class CombinedStrategy:
    """Try ψ adjustment first, then energy, then both."""

    def find_safe_point(self, avoidance_map, target_psi, target_energy,
                        psi_tolerance, energy_tolerance):
        # 1. Try NearestSafeStrategy (ψ only)
        # 2. Try EnergyOffsetStrategy (energy only)
        # 3. Try joint (ψ, E) grid search within tolerance box
```

### Phase 5: Beamline Interface (`agent/beamline.py`)

```python
class BeamlineInterface(ABC):
    """Abstract interface to diffractometer control."""

    @abstractmethod
    def move_to(self, psi, energy, hkl): ...

    @abstractmethod
    def collect(self, exposure_time): ...

    @abstractmethod
    def current_position(self): ...


class SimulatedBeamline(BeamlineInterface):
    """For testing: uses existing DMS simulation as ground truth."""

    def __init__(self, lattice, phason_matrix, detector_geometry):
        """Set up simulated crystal and detector."""

    def collect(self, exposure_time):
        """Generate synthetic detector image including Bragg MS and DMS."""


class I16Beamline(BeamlineInterface):
    """Interface to Diamond I16 via GDA/EPICS."""
    # Concrete implementation for the real beamline
```

## 5. Integration with Existing Codebase

### 5.1 Reuse from `ts_quasi.py`

The following existing classes and functions are used directly (not duplicated):

| Used by | From `ts_quasi.py` | Purpose |
|---------|---------------------|---------|
| `diffuse.py` | `bmatrix`, `Projection6d` | Q-vector computation, 6D decomposition |
| `dms_simulator.py` | `bragg`, `rotxyz`, `psith2v`, `dms2px`, `makekernel` | Geometry, detector projection |
| `dms_simulator.py` | `PhasonDistoArray` | Phason strain on reflection positions |
| `avoidance.py` | `pilkhlrange` | Reflection enumeration |
| `agent/` | `roibuilder_ico_hkl`, `msroi` | ROI overlap assessment |

### 5.2 New imports required

```
numpy          — already a dependency
scipy.interpolate — for avoidance map interpolation (scipy already a dependency)
multiprocessing   — stdlib, for parallel map building
```

No new external dependencies.

### 5.3 Hooking into the existing simulation for validation

The existing `dmsfit_ico_hkl.imcalc()` (line 1629 of `ts_quasi.py`) produces a simulated Bragg MS image. To add DMS:

```python
# In dmsfit_ico_hkl.imcalc(), after line 1822:
#   self.imsim = ndimage.convolve(imsim, makekernel('gauss', 15, simsigma))
#
# Add optional DMS contribution:
if self.diffuse_model is not None:
    dms_sim = DMSSimulator(
        lattice, energy, hkl, psi,
        self.diffuse_model, detector_geometry, phason_matrix
    )
    self.imsim_dms = dms_sim.compute_vectorised()
    self.imsim += self.imsim_dms  # additive contribution
```

This is **optional and backwards-compatible** — the diffuse_model parameter defaults to None and existing scripts are unaffected.

## 6. Validation Plan

### 6.1 Unit tests

- `test_diffuse.py`: verify `DebyeTDS.intensity()` scales as |Q|², `PhasonDiffuse.intensity()` scales as |Q_perp|²/|Q|², both return correct units.
- `test_dms_simulator.py`: for a known simple case (cubic crystal, single Bragg reflection, isotropic TDS), verify that DMS arcs appear at the correct angular positions.
- `test_avoidance.py`: verify that the avoidance map correctly identifies known DMS-prone orientations.

### 6.2 Comparison with existing Bragg MS

- Run existing `fit_fivefold_axis` script → Bragg MS image.
- Run DMS simulator at the same orientation → DMS image.
- Verify DMS features are **broader** and **weaker** than Bragg MS, and are centred on the same angular positions (they should be — DMS is the diffuse halo around Bragg MS lines).

### 6.3 Experimental validation

- Compare DMS predictions against experimental data from Diamond I16 where DMS contamination is known to occur.
- Use the existing `multiroifit2` peak fitting: if a ROI shows anomalous broadening or shifted centre beyond what Bragg MS predicts, check if DMS simulation explains the residual.

## 7. Implementation Priority

| Priority | Component | Effort | Rationale |
|----------|-----------|--------|-----------|
| 1 | `diffuse.py` — `DebyeTDS` | 1 day | Simplest model, enables all downstream work |
| 2 | `dms_simulator.py` — vectorised forward model | 3 days | Core computation, most complex piece |
| 3 | `diffuse.py` — `PhasonDiffuse` | 1 day | Quasicrystal-specific, needed for Al-Pd-Mn |
| 4 | `avoidance.py` — map builder + lookup | 2 days | Enables real-time steering |
| 5 | `agent/dms_agent.py` — steering logic | 2 days | Decision-making layer |
| 6 | `agent/beamline.py` — `SimulatedBeamline` | 1 day | Testing without hardware |
| 7 | Validation against experimental data | 2 days | Confidence before deployment |
| 8 | `agent/beamline.py` — `I16Beamline` | 2 days | Real beamline integration (needs GDA access) |

**Total estimated effort: ~2 weeks**

## 8. Open Questions

1. **Diffuse scattering input data**: should the diffuse model be purely analytical (Debye/phason elastic theory), or should it also accept measured diffuse scattering maps (e.g., from a preliminary wide-angle survey)?

2. **Multi-beam DMS**: the current plan handles two-beam DMS (Bragg + diffuse). Three-beam paths (Bragg → Bragg → diffuse, or Bragg → diffuse → Bragg) exist but are much weaker — include or defer?

3. **Agent communication protocol**: what interface does the beamline control system expect? GDA Jython scripts, EPICS PVs, or a REST API? This determines the `I16Beamline` implementation.

4. **Threshold calibration**: the binary safe/unsafe threshold for the avoidance map needs calibration against experimental data. What level of DMS contamination is acceptable for the specific measurement being made?
