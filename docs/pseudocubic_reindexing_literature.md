# Literature support for pseudo-cubic re-indexing via multiple diffraction

Assessment of whether the 12-matrix pseudo-cubic re-indexing approach — from
Nisbet et al. (2023), *J. Appl. Cryst.* **56**, 1046–1050
([doi:10.1107/S1600576723004120](https://doi.org/10.1107/S1600576723004120)) —
has support in the wider literature beyond that paper.

**Short answer:** Yes. The two ingredients of the method each rest on
well-established crystallography; the specific *combination* — enumerating the
12 pseudo-cubic indexing choices and using a multiple-diffraction triple-
intersection "truth table" to select the correct one — appears to be the
paper's own contribution.

---

## 1. The 12 matrices are a coset decomposition (well established)

Table 1 is, in group-theoretic terms, the set of coset representatives of the
pseudo-cubic **metric** point group with respect to the true **crystal** point
group:

- Cubic metric symmetry m3̄m: order 48 (24 proper rotations).
- Rhombohedral crystal point group 3̄m: order 12 (6 proper rotations).
- 24 / 6 = **4 distinct orientational domains**; the three equivalent matrices
  within each table row are the crystal's own symmetry operations.

This is exactly the standard method for deriving twin / orientation laws by
(pseudo-)merohedry. The indexing ambiguity being corrected is formally the same
object as an orientational twin law, so the matrices can be **independently
regenerated** rather than only transcribed — a useful correctness check.

| Reference | Relevance |
|-----------|-----------|
| Flack, H. D. (1987). *Acta Cryst.* A**43**, 564–568. [The derivation of twin laws for (pseudo-)merohedry by coset decomposition](https://journals.iucr.org/paper?S0108767387099008=) | Canonical method that generates the same set of matrices |
| [IUCr twinning literature index](http://www.cryst.chem.uu.nl/lutz/twin/twin_lit.html) | Background bibliography on twinning by (pseudo-)merohedry |
| [Practical hints and tips for solution of pseudo-merohedric twins (Acta E, 2021)](https://journals.iucr.org/e/issues/2021/05/00/hb7973/) | Worked pseudo-merohedric examples |

**Independent check performed on the coded table:** all 12 matrices are proper
rotations (determinant +1) and integer-orthogonal (`M · Mᵀ = I`); as printed,
matrix 9 duplicates matrix 2.

---

## 2. Multiple diffraction / Renninger scans as the discriminator (established in adjacent problems)

Azimuthal (Renninger) scans and multiple-diffraction peaks are a long-standing,
highly sensitive probe of the small distortions that split cubic degeneracy —
precisely the regime where ordinary Bragg-peak splitting is too small to see.

| Reference | Relevance |
|-----------|-----------|
| [Phase transition & thermal-expansion coefficients from secondary Renninger reflections](https://ouci.dntb.gov.ua/en/works/ldLAJzkl/) (Morelhão / Cardoso school) | Renninger scans resolve sub-Å lattice-parameter variation |
| [Hybrid multiple diffraction in Renninger scan for heteroepitaxial layers](https://www.academia.edu/25112318/Hybrid_multiple_diffraction_in_Renninger_scan_for_heteroepitaxial_layers) | Multiple-diffraction geometry sensitive to orientation / interface effects |
| [High-accuracy lattice-parameter determination by n-beam diffraction (arXiv cond-mat/0411508)](https://arxiv.org/pdf/cond-mat/0411508) | n-beam precision to ~1/10000 Å |
| [Superstructure reflections in tilted perovskites (PMC11363165)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11363165/) | Confirms peak splitting in pseudo-cubic ferroics is often too small to observe directly |

---

## 3. What appears to be original to the paper

The specific pairing of (a) the enumerated 12 pseudo-cubic indexing choices with
(b) the multiple-diffraction **triple-intersection truth table** as the
selection criterion is not something I found in prior work. The closest
analogues solve related but different problems:

| Reference | How it differs |
|-----------|----------------|
| Flack (1987), above | Resolves twin fractions by least-squares refinement, not by a multiple-diffraction truth table |
| [Resolving indexing ambiguities in XFEL patterns (PMC6400252)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6400252/) | Serial femtosecond crystallography; different physical setting |
| [Indexing ambiguity in SFX via expectation-maximization (PMC4224458)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4224458/) | Algorithmic (EM) resolution of the merohedral ambiguity, not multiple diffraction |

---

## Bottom line

The 12-matrix re-indexing option is on firm crystallographic footing: it is a
coset decomposition (Flack 1987), independently derivable, and the use of
multiple diffraction to detect the underlying distortion is well precedented in
the Renninger-scan literature. The novel step — using the triple-intersection
truth table to *pick* the correct indexing among the 12 — is the paper's own.

### Suggested follow-ups for the codebase

- Add a Flack (1987) citation alongside the existing paper reference in
  `PSEUDOCUBIC_TRANSFORMS` (`ts_quasi.py`).
- Optionally add a helper that regenerates the matrices by coset decomposition,
  so the hard-coded table becomes self-verifying rather than transcribed.
