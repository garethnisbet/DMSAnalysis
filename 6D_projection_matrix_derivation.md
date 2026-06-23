# Derivation of the 6D Projection ("τ") Matrix in `ts_quasi.py`

This note explains the origin of the 6×6 matrix with `τ` (golden-ratio) entries used in
`Projection6d` / `Projection6dArrayApproximant` in `DMSAnalysis/ts_quasi.py`, tying each
piece to the foundational quasicrystal literature.

## Why six dimensions at all

An icosahedral quasicrystal's diffraction pattern is a *Z*-module of rank 6, not 3: every
Bragg peak's reciprocal vector can be written as an integer combination of six basis vectors,

```
Q = Σ_{i=1..6} n_i e_i ,   n_i ∈ Z
```

but those six **e_i** are linearly *dependent* over the reals (they point to 6 of the 12
vertices of an icosahedron, related by τ), so you need 6 integers to label peaks that live in
physical 3-space. The resolution is to lift the structure into a 6D periodic hypercubic
lattice **Z⁶** and split **R⁶ = E∥ ⊕ E⊥** into a *parallel* (physical) 3-space and a
*perpendicular* (phason) 3-space. That decomposition is exactly what the 6×6 matrix performs.

### Literature

- **V. Elser**, *Indexing problems in quasicrystal diffraction*, Phys. Rev. B **32**, 4892 (1985).
- **J. W. Cahn, D. Shechtman, D. Gratias**, *Indexing of icosahedral quasiperiodic crystals*,
  J. Mater. Res. **1**, 13 (1986).
- General superspace formalism: **T. Janssen**, Acta Cryst. A**42**, 261 (1986);
  **P. Bak**, PRL **54**, 1517 (1985).

## The geometry that fixes the matrix entries

The icosahedron has 12 vertices at the cyclic permutations of `(0, ±1, ±τ)`, where
`τ = (1+√5)/2` (line 33, `TAU = 0.5+0.5*5**0.5`). Pick one vertex from each of the 6
antipodal pairs as the basis. In `Projection6d` the **columns** of the matrix are the images
of the six 6D unit vectors **e_i**; rows 1–3 are their projection into E∥ and rows 4–6 into E⊥:

```
e_i  →  ( parallel part | perpendicular part )
e_1  →  ( 1,  τ,  0 | -τ,  1,  0 )
e_2  →  ( τ,  0,  1 |  1,  0, -τ )
e_3  →  ( 0,  1,  τ |  0, -τ,  1 )
e_4  →  (-1,  τ,  0 |  1,  1,  0 )    ...etc.
```

Two facts make this *the* matrix and not an arbitrary one:

1. **Parallel components are an icosahedral star.** `(1, τ, 0)`, `(τ, 0, 1)`, `(0, 1, τ)`, …
   are 6 icosahedron vertices, one per antipodal pair. This guarantees the projected
   reciprocal lattice has icosahedral point symmetry m-3̄-5 (235).

2. **Perpendicular components are the Galois conjugate star.** The map taking the parallel
   star to the perpendicular star is the field automorphism of **Q(√5)** given by
   `√5 → −√5`, i.e. `τ → τ' = −1/τ = 1 − τ`. This conjugation is the algebraic heart of every
   quasicrystal embedding: the same six integers, read with `τ` give the physical vector, read
   with `τ'` give the perpendicular (phason) vector. It is why E∥ and E⊥ are both icosahedral
   but mutually incommensurate.

## The normalization constant

Line 884: `const = 1/np.sqrt(2.0*(2.0+TAU))`. Each parallel column has length²

```
1² + τ² + 0² = 1 + τ² = 1 + (τ+1) = 2 + τ      (using τ² = τ + 1)
```

and the perpendicular column has the same length² = 2 + τ. So each full 6D column has
length² = `2(2+τ)`. Dividing by `√(2(2+τ))` makes every column a **unit** vector, and the
columns are mutually orthogonal — so `m6d` is a genuine 6×6 **orthogonal (rotation) matrix**.
That orthonormality is precisely why it can be a rigid rotation of **R⁶** that lines three
axes up with E∥ and three with E⊥, preserving lengths and the `Z⁶` lattice metric. (In
`Projection6dArrayApproximant` the same thing is done numerically:
`const = 1/np.linalg.norm(self.rmat[0,:])`.)

## The `mmm` matrix — Elser vs. Cahn indexing

The comment on lines 838/873 ("transform Elser's 6D indices to Cahn's 6D indices") explains
the leading `mmm` permutation matrix. Elser and Cahn-Shechtman-Gratias chose **different
orderings/signs of the six basis vectors** (and different normalizations — Elser often uses an
integer/primitive-cubic setting, CSG a body-centred or scaled one). `mmm` is just the signed
permutation `P` that relabels a 6-vector from Elser's convention into Cahn's before the
projection `m6d` is applied:

```
tmp = mmm · v0          # reindex Elser → Cahn  (line 906)
tmp = m6d · tmp         # project to (∥, ⊥)     (line 907)
```

So the data may be indexed in one literature convention while the projection is defined in the
other; `mmm` reconciles them.

## Summary

The matrix is not derived as a rotation by some angle — it is *constructed* by demanding that

- (a) the six 6D basis vectors project in E∥ onto an icosahedral vertex star (giving
  icosahedral symmetry to the diffraction pattern),
- (b) their E⊥ images be the `√5 → −√5` Galois conjugate star (the cut-and-project condition),
- (c) the whole thing be orthonormalized (the `1/√(2(2+τ))` factor) so it is a
  length-preserving rotation of **R⁶**.

That is exactly the Elser / Cahn-Shechtman-Gratias icosahedral indexing scheme, and the
phason-strain matrix refined later is a small linear distortion added to the E⊥ block of this
same construction.

---

## Appendix: Antipodal pairs

"Antipodal pair" means two points diametrically opposite through the centre — like the North
and South poles of a sphere. If one vertex is at position **v**, its antipode is at **−v**
(same line through the centre, opposite direction, same distance).

### In the icosahedron

An icosahedron has **12 vertices** in **6 antipodal pairs**: for every vertex **v** there is
exactly one opposite vertex **−v**. With vertices at the cyclic permutations of `(0, ±1, ±τ)`:

```
(1, τ, 0)   and its antipode   (-1, -τ, 0)
(0, 1, τ)   and its antipode   (0, -1, -τ)
(τ, 0, 1)   and its antipode   (-τ, 0, -1)
   ...                          ...
12 vertices  =  6 pairs of (v, -v)
```

### Why it matters for the basis

A reciprocal vector **Q** and its negative **−Q** are not independent (Friedel pair; as basis
vectors **v** and **−v** span the same line, so they are linearly dependent). To build the 6D
index basis you want six genuinely independent directions, so you pick **one vertex from each
of the 6 antipodal pairs** — giving exactly the rank-6 basis. Choosing the other member of a
pair just flips the sign of that index `n_i → −n_i`; it adds no new degree of freedom. That is
why "6 of the 12 vertices, one per antipodal pair" appears in the construction, and those 6
vectors become the columns of the projection matrix.
