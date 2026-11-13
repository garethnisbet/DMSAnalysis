# Copyright 2014 Diamond Light Source Ltd.123
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dr Gareth Nisbet, gareth.nisbet@diamond.ac.uk Tel: +44 1235 778786
# www.diamond.ac.uk 
# Diamond Light Source, Chilton, Didcot, Oxon, OX11 0DE, U.K.
import numpy as np
from numpy import linalg as LA
from scipy import ndimage
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize, differential_evolution, basinhopping
import copy
#from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
# shapely powers the Kossel-line intersection test used by the tripfit engine;
# imported lazily-friendly (the rest of the module works without it).
try:
    from shapely.geometry.polygon import LinearRing as _LinearRing
except Exception:      # pragma: no cover - shapely is a declared dependency
    _LinearRing = None


TAU =  0.5+0.5*5**0.5
# The rational approximant the slider projects 6D reflections with
# (Projection6dArrayApproximant) and exports as crystal.tau_approx.  Shared so
# the tripfit engine indexes a quasicrystal exactly as the slider does.
TAU_APPROX = 55 / 34.


def eval_number_expr(text):
    '''Evaluate an arithmetic formula such as ``0.5+0.5*5**0.5`` or ``55/34``
    to a float, without ``eval``: numbers, ``+ - * / **``, unary signs,
    parentheses, ``sqrt(x)`` and the constant ``tau`` (the golden ratio) only.
    Raises ``ValueError`` for anything else or a non-finite result.'''
    import ast
    import math
    import operator as op
    bins = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
            ast.Div: op.truediv, ast.Pow: op.pow}
    unas = {ast.UAdd: op.pos, ast.USub: op.neg}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in bins:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 64:
                raise ValueError('exponent too large')
            return bins[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in unas:
            return unas[type(n.op)](ev(n.operand))
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'sqrt' and len(n.args) == 1 and not n.keywords):
            return math.sqrt(ev(n.args[0]))
        if isinstance(n, ast.Name) and n.id == 'tau':
            return TAU
        raise ValueError('not allowed in a formula: %s' % ast.dump(n)[:40])

    try:
        val = ev(ast.parse(str(text).strip(), mode='eval'))
    except (SyntaxError, ZeroDivisionError, OverflowError, TypeError) as e:
        raise ValueError('cannot evaluate %r: %s' % (text, e))
    if not isinstance(val, float) or not math.isfinite(val):
        raise ValueError('%r does not evaluate to a finite number' % (text,))
    return val
###################################

# ── Conventional-crystal symmetry layer ────────────────────────────────────────
# Support for ordinary (non-quasicrystal) crystals indexed with 3-element Miller
# indices.  These helpers are table-driven and shared by the slider, the batch
# fitter and the fit engine so the parameter packing can never drift between
# them.  The lattice is carried in slots [0..5] = [a, b, c, alpha, beta, gamma]
# of the 24-element guess vector (the same slots the icosahedral path reserves);
# the phason slots [15..23] are unused for conventional crystals.

CONVENTIONAL_SYSTEMS = ('cubic', 'tetragonal', 'tetragonal_a', 'tetragonal_b',
                        'orthorhombic', 'monoclinic', 'monoclinic_a', 'monoclinic_c',
                        'rhombohedral', 'hexagonal', 'triclinic')

# Which lattice slots [a,b,c,alpha,beta,gamma] are free (refined) per system.
# Tetragonal defaults to the unique-c setting (a=b); the *_a / *_b variants make
# a or b the unique axis.  Monoclinic defaults to b-unique (beta != 90); the
# *_a / *_c variants make alpha or gamma the non-90 angle.
_LATTICE_FREE_SLOTS = {
    'cubic':         [0],
    'tetragonal':    [0, 2],        # unique c  (a=b)
    'tetragonal_a':  [0, 1],        # unique a  (b=c)
    'tetragonal_b':  [0, 1],        # unique b  (a=c)
    'hexagonal':     [0, 2],
    'orthorhombic':  [0, 1, 2],
    'monoclinic':    [0, 1, 2, 4],  # beta  != 90 (b unique)
    'monoclinic_a':  [0, 1, 2, 3],  # alpha != 90 (a unique)
    'monoclinic_c':  [0, 1, 2, 5],  # gamma != 90 (c unique)
    'rhombohedral':  [0, 3],
    'triclinic':     [0, 1, 2, 3, 4, 5],
}


def lattice_free_slots(system):
    '''Indices into the 6-element lattice [a,b,c,alpha,beta,gamma] that are free
    (refined) for the given crystal system.'''
    try:
        return list(_LATTICE_FREE_SLOTS[system])
    except KeyError:
        raise ValueError('Unknown crystal system: %s' % system)


def expand_lattice(system, six):
    '''Return the full constrained lattice [a,b,c,alpha,beta,gamma] for a crystal
    system, reading only the free slots of ``six`` and enforcing the symmetry
    constraints.  Stale values in constrained slots are therefore harmless.'''
    a, b, c, alpha, beta, gamma = (float(six[0]), float(six[1]), float(six[2]),
                                   float(six[3]), float(six[4]), float(six[5]))
    if system == 'cubic':
        return [a, a, a, 90.0, 90.0, 90.0]
    elif system == 'tetragonal':       # unique c
        return [a, a, c, 90.0, 90.0, 90.0]
    elif system == 'tetragonal_a':     # unique a  (b=c, both = slot 1)
        return [a, b, b, 90.0, 90.0, 90.0]
    elif system == 'tetragonal_b':     # unique b  (a=c)
        return [a, b, a, 90.0, 90.0, 90.0]
    elif system == 'hexagonal':
        return [a, a, c, 90.0, 90.0, 120.0]
    elif system == 'orthorhombic':
        return [a, b, c, 90.0, 90.0, 90.0]
    elif system == 'monoclinic':       # beta unique
        return [a, b, c, 90.0, beta, 90.0]
    elif system == 'monoclinic_a':     # alpha unique
        return [a, b, c, alpha, 90.0, 90.0]
    elif system == 'monoclinic_c':     # gamma unique
        return [a, b, c, 90.0, 90.0, gamma]
    elif system == 'rhombohedral':
        return [a, a, a, alpha, alpha, alpha]
    elif system == 'triclinic':
        return [a, b, c, alpha, beta, gamma]
    raise ValueError('Unknown crystal system: %s' % system)


def reduced_param_indices(system, detopt, energyopt):
    '''Indices into the 24-element guess vector passed to the optimiser for a
    conventional crystal: the free lattice slots, then psicor (6), chicor (7) and
    thetacor (8), then the detector geometry (if ``detopt``) and energy (if
    ``energyopt``).  Slots 7/8 (formerly hcor/kcor) are repurposed as the chi /
    theta corrections; lcor (9) is redundant with the primary hkl and the phason
    slots [15..23] are never included.'''
    idx = lattice_free_slots(system) + [6, 7, 8]
    if detopt:
        idx += [10, 11, 12, 13]
    if energyopt:
        idx += [14]
    return idx


IMCALC_MODES = ('icosahedral', 'icosahedral_fixed_a', 'cubic_no_strain', 'calibrate')

def imcalc_param_indices(system, detopt, energyopt):
    '''Indices into the 24-element guess that make up the reduced vector
    ``dmsfit_ico_hkl.imcalc`` unpacks, for any mode it supports.

    The packing is fixed by imcalc's own branches, so it lives here, next to the
    engine, for the slider and fit.py to share.  For the non-conventional modes:
    ``a`` (unless the lattice is fixed), psicor/chicor/thcor, then slot 9 as a
    placeholder imcalc skips — needed only when something follows it, since the
    detector block, energy and phason block are read at fixed offsets past it —
    then the detector (``detopt``), energy (``energyopt``) and phason block.

    These tables were once written out by hand per mode and drifted from imcalc:
    ``cubic_no_strain`` with the detector refined was one slot short (every
    evaluation raised IndexError, so the slider overlay silently stopped
    updating), and ``icosahedral_fixed_a`` fed the energy in as the detector z
    rotation.'''
    if system in CONVENTIONAL_SYSTEMS:
        return reduced_param_indices(system, detopt, energyopt)
    if system not in IMCALC_MODES:
        raise ValueError('Unknown bravais: %s' % system)
    phason = system in ('icosahedral', 'icosahedral_fixed_a')
    idx = ([0] if system in ('icosahedral', 'cubic_no_strain') else []) + [6, 7, 8]
    tail = ([10, 11, 12, 13] if detopt else []) + ([14] if energyopt else [])
    if tail or phason:
        idx += [9]
    idx += tail
    if phason:
        idx += list(range(15, 24))
    return idx


def hklgen_3d(depth):
    '''All integer Miller indices [h,k,l] in [-depth, depth]^3 minus the origin
    (the 3D analogue of the icosahedral 6D hkl generator).'''
    import itertools as _it
    rng = range(-depth, depth + 1)
    idx = np.array(list(_it.product(rng, repeat=3)))
    return idx[np.any(idx != 0, axis=1)]


# ── Pseudo-cubic re-indexing matrices ──────────────────────────────────────────
# Table 1 of Nisbet et al. (2023), J. Appl. Cryst. 56, 1046-1050
# (https://doi.org/10.1107/S1600576723004120): the 12 transformation matrices
# relating the equivalent indexing choices of a pseudo-cubic crystal.  Matrix 1
# is the identity (indexing unchanged).  A matrix M re-indexes a reflection as
# hkl' = M @ hkl; it must be applied consistently to the primary reflection, the
# azimuthal reference and every reflection in the list.  The matrices are listed
# in the order they appear in the table (row-major: table rows 1-4, three
# matrices per row; the three matrices in a table row give the same truth-table
# vector).  Note: as published, matrix 9 is identical to matrix 2.
PSEUDOCUBIC_TRANSFORMS = tuple(np.array(_m, dtype=int) for _m in (
    # table row 1
    [[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]],   # 1  (identity)
    [[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]],   # 2
    [[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]],   # 3
    # table row 2
    [[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]],   # 4
    [[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]],   # 5
    [[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]],   # 6
    # table row 3
    [[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]],   # 7
    [[ 0,  0,  1], [ 0, -1,  0], [ 1,  0,  0]],   # 8
    [[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]],   # 9  (as published; = matrix 2)
    # table row 4
    [[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]],   # 10
    [[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]],   # 11
    [[ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0]],   # 12
))


def pseudocubic_matrix(index):
    '''The Table-1 pseudo-cubic re-indexing matrix for 1-based ``index`` (1-12);
    index 1 is the identity.'''
    index = int(index)
    if not 1 <= index <= len(PSEUDOCUBIC_TRANSFORMS):
        raise ValueError('pseudocubic transform index must be 1..%d, got %s'
                         % (len(PSEUDOCUBIC_TRANSFORMS), index))
    return PSEUDOCUBIC_TRANSFORMS[index - 1]


def pseudocubic_label(index):
    '''Compact one-line label for a Table-1 matrix, e.g.
    ``'(1 0 0)(0 -1 0)(0 0 -1)'``; index 1 returns ``'identity'``.'''
    if int(index) == 1:
        return 'identity'
    m = pseudocubic_matrix(index)
    return ''.join('(%s)' % ' '.join('%d' % v for v in row) for row in m)


# ── Coset-decomposition derivation of the pseudo-cubic domains ──────────────────
# The 12 hard-coded PSEUDOCUBIC_TRANSFORMS were transcribed from a published
# table, so they carry a transcription risk.  The functions below re-derive the
# underlying group theory from first principles — the pseudo-cubic indexing
# ambiguity is a twinning-by-pseudo-merohedry problem (Flack, H. D. (1987),
# Acta Cryst. A43, 564-568): the distinct indexing choices are the left cosets
# of the crystal's (rhombohedral) point group in the cubic metric holohedry.
# `verify_pseudocubic_transforms` checks the hard-coded table against this
# derivation, so a sign slip or dropped matrix cannot pass silently.

def cubic_proper_rotations():
    '''The 24 proper rotations of the cube (point group 432 / O): every 3x3
    signed permutation matrix with determinant +1.'''
    import itertools as _it
    mats = []
    for perm in _it.permutations(range(3)):
        for signs in _it.product((1, -1), repeat=3):
            M = np.zeros((3, 3), dtype=int)
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if int(round(np.linalg.det(M))) == 1:
                mats.append(M)
    return mats


def _rot180(axis):
    '''Integer matrix of a 180 degrees rotation about ``axis`` (a lattice
    direction); exact for the <100>/<110>/<111> axes of the cube.'''
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    return np.round(2.0 * np.outer(n, n) - np.eye(3)).astype(int)


def rhombohedral_proper_group():
    '''The 6 proper rotations of point group 32 (D3), oriented as the
    rhombohedral subgroup of the cube: the 3-fold about the body diagonal
    [111] together with the three 2-folds about the <1-10> directions
    perpendicular to it.'''
    A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)   # 3-fold [111]
    return [np.eye(3, dtype=int), A, A @ A,
            _rot180([1, -1, 0]), _rot180([0, 1, -1]), _rot180([1, 0, -1])]


def _mat_in(M, group):
    return any(np.array_equal(M, G) for G in group)


def coset_decomposition(group, subgroup):
    '''Left-coset decomposition of ``group`` by ``subgroup``: a list of cosets
    (each a list of matrices) that partitions ``group``.  ``subgroup`` must be a
    genuine subgroup, so the cosets tile the group with no overlap.'''
    cosets, seen = [], set()
    for g in group:
        if tuple(g.flatten()) in seen:
            continue
        cs = [g @ h for h in subgroup]
        cosets.append(cs)
        seen.update(tuple(c.flatten()) for c in cs)
    return cosets


def pseudocubic_domains():
    '''The 4 distinct pseudo-cubic indexing domains, derived as the left cosets
    of the rhombohedral proper group (32, 3-fold along [111]) in the cubic
    proper holohedry (432).  Returns one canonical representative matrix per
    coset (the member with the fewest sign changes, ties broken
    lexicographically), identity domain first.

    These 4 cosets are the group-theoretic content behind the 12
    PSEUDOCUBIC_TRANSFORMS: the tabulated matrices distribute three-per-coset
    across these domains (see `verify_pseudocubic_transforms`).  Inversion is
    not needed — reflection indexing is centrosymmetric (Friedel's law), so the
    proper rotation groups already capture every distinct indexing.'''
    cosets = coset_decomposition(cubic_proper_rotations(),
                                 rhombohedral_proper_group())

    def _key(M):
        return (int((M < 0).sum()), tuple(-M.flatten()))

    reps = [min(cs, key=_key) for cs in cosets]
    eye = np.eye(3, dtype=int)
    reps.sort(key=lambda M: (not np.array_equal(M, eye), _key(M)))
    return reps


def verify_pseudocubic_transforms():
    '''Check the hard-coded PSEUDOCUBIC_TRANSFORMS table against the coset
    decomposition, confirming it is the correct and complete set of pseudo-cubic
    indexing matrices.  Raises AssertionError on any mismatch; returns a summary
    dict on success.

    Verifies that
      * every tabulated matrix is a proper cubic rotation (an element of 432);
      * the cubic proper group splits into exactly 4 cosets of the rhombohedral
        proper group (the 4 distinct indexing domains), each of order 6;
      * every domain is represented by the table (it is complete); and
      * counting the table exactly as published (matrix 9 is a duplicate of
        matrix 2), three tabulated entries fall in each of the 4 domains.'''
    O  = cubic_proper_rotations()
    D3 = rhombohedral_proper_group()
    assert len(O) == 24, 'cubic proper group must have 24 elements'
    assert all(_mat_in(g, O) for g in D3), 'D3 is not a subgroup of O'
    assert all(_mat_in(a @ b, D3) for a in D3 for b in D3), 'D3 not closed'

    cosets = coset_decomposition(O, D3)
    assert len(cosets) == 4 and all(len(c) == 6 for c in cosets), \
        'expected 4 cosets of order 6, got sizes %s' % [len(c) for c in cosets]

    def coset_of(M):
        for i, cs in enumerate(cosets):
            if _mat_in(M, cs):
                return i
        return None

    n = len(PSEUDOCUBIC_TRANSFORMS)
    counts = [0, 0, 0, 0]
    for i in range(1, n + 1):
        M = pseudocubic_matrix(i)
        assert _mat_in(M, O), 'matrix %d is not a proper cubic rotation' % i
        c = coset_of(M)
        assert c is not None, 'matrix %d not located in any coset' % i
        counts[c] += 1
    assert set(coset_of(pseudocubic_matrix(i)) for i in range(1, n + 1)) \
        == {0, 1, 2, 3}, 'not all 4 domains are represented by the table'
    assert all(k == 3 for k in counts), \
        'expected 3 tabulated matrices per domain, got %s' % counts
    return {'n_domains': 4, 'per_domain': counts, 'n_matrices': n}

###################################

class bmatrix(object):
    """ Convert to Cartesian coordinate system. Returns the Bmatrix and the metric tensors in direct and reciprocal spaces"""
    def __init__(self,lattice):#
        self.lattice = lattice
        lattice=self.lattice
        a=lattice[0];
        b=lattice[1];
        c=lattice[2];
        alph = lattice[3];
        bet =  lattice[4];
        gamm = lattice[5];
        alpha1=alph*np.pi/180.0
        alpha2=bet*np.pi/180.0
        alpha3=gamm*np.pi/180.0
        beta1=np.arccos((np.cos(alpha2)*np.cos(alpha3)-np.cos(alpha1))/(np.sin(alpha2)*np.sin(alpha3)))
        beta2=np.arccos((np.cos(alpha1)*np.cos(alpha3)-np.cos(alpha2))/(np.sin(alpha1)*np.sin(alpha3)))
        beta3=np.arccos((np.cos(alpha1)*np.cos(alpha2)-np.cos(alpha3))/(np.sin(alpha1)*np.sin(alpha2)))
        b1=1./(a*np.sin(alpha2)*np.sin(beta3))
        b2=1./(b*np.sin(alpha3)*np.sin(beta1))
        b3=1./(c*np.sin(alpha1)*np.sin(beta2))
        c1= b1*b2*np.cos(beta3);
        c2= b1*b3*np.cos(beta2);
        c3= b2*b3*np.cos(beta1);
        self.bmatrix = np.matrix([[b1,b2*np.cos(beta3),b3*np.cos(beta2)],[0.0,b2*np.sin(beta3),-b3*np.sin(beta2)*np.cos(alpha1)],[0.0, 0.0, 1./c]])
    def bm(self):
        return self.bmatrix
    def ibm(self):
        return self.bmatrix.I
    def mt(self):
        return self.bmatrix.I*self.bmatrix.transpose().I
    def rmt(self):
        mt=self.bmatrix.I*self.bmatrix.transpose().I
        return mt.I
    def volume(self):
        self.vol = np.sqrt(np.linalg.det(self.bmatrix.I*self.bmatrix.transpose().I))
        return self.vol
    
    def reciprocal_parameters(self,lp2=[]):
        if lp2 == []:
            lp = self.lattice
        else:
            lp = lp2
        lp[3:] = np.radians(lp[3:])
#         cell_volume = np.sqrt(LA.det(self.bmatrix.I*self.bmatrix.transpose().I))
        cell_volume = np.sqrt(np.linalg.det(self.bmatrix.I*self.bmatrix.transpose().I))
        rp = np.zeros(6)
        rp[0]=(lp[1]*lp[2]*np.sin(lp[3])/cell_volume)
        rp[1]=(lp[2]*lp[0]*np.sin(lp[4])/cell_volume)
        rp[2]=(lp[0]*lp[1]*np.sin(lp[5])/cell_volume)
        rp[3]=(np.arccos( (np.cos(lp[4])*np.cos(lp[5])-np.cos(lp[3])) / (np.sin(lp[4])*np.sin(lp[5])) ))
        rp[4]=(np.arccos( (np.cos(lp[3])*np.cos(lp[5])-np.cos(lp[4])) / (np.sin(lp[3])*np.sin(lp[5])) ))
        rp[5]=(np.arccos( (np.cos(lp[3])*np.cos(lp[4])-np.cos(lp[5])) / (np.sin(lp[3])*np.sin(lp[4])) ))
        rp[3:] = np.rad2deg(rp[3:])
        return rp
    
    def direct_matrix(self):
        lp = self.lattice
        lp_norm = [1,1,1,lp[3],lp[4],lp[5]]
        rp_norm = self.reciprocal_parameters(lp_norm)
        direct_matrix = np.array([[ lp_norm[0], lp_norm[1]*np.cos(np.radians(lp_norm[5])), lp_norm[2]*np.cos(np.radians(lp_norm[4])) ],
                            [ 0,        lp_norm[1]*np.sin(np.radians(lp_norm[5])), -lp_norm[2]*np.sin(np.radians(lp_norm[4]))*np.cos(np.radians(rp_norm[3])) ],
                            [ 0,        0,                                         1/rp_norm[2] ]])
        return direct_matrix
 


class rotxyz(object):
    """Example p = rotxyz(initial_vector, vectorrotateabout, angle)"""
    def __init__(self,u,angle):
        self.u = u
        self.angle = angle
        u=np.matrix(self.u)/np.linalg.norm(np.matrix(self.u))
        e11=u[0,0]**2+(1-u[0,0]**2)*np.cos(angle*np.pi/180.0)
        e12=u[0,0]*u[0,1]*(1-np.cos(angle*np.pi/180.0))-u[0,2]*np.sin(angle*np.pi/180.0)
        e13=u[0,0]*u[0,2]*(1-np.cos(angle*np.pi/180.0))+u[0,1]*np.sin(angle*np.pi/180.0)
        e21=u[0,0]*u[0,1]*(1-np.cos(angle*np.pi/180.0))+u[0,2]*np.sin(angle*np.pi/180.0)
        e22=u[0,1]**2+(1-u[0,1]**2)*np.cos(angle*np.pi/180.0)
        e23=u[0,1]*u[0,2]*(1-np.cos(angle*np.pi/180.0))-u[0,0]*np.sin(angle*np.pi/180.0)
        e31=u[0,0]*u[0,2]*(1-np.cos(angle*np.pi/180.0))-u[0,1]*np.sin(angle*np.pi/180.0)
        e32=u[0,1]*u[0,2]*(1-np.cos(angle*np.pi/180.0))+u[0,0]*np.sin(angle*np.pi/180.0)
        e33=u[0,2]**2+(1-u[0,2]**2)*np.cos(angle*np.pi/180.0)
        self.rotmat = np.matrix([[e11,e12,e13],[e21,e22,e23],[e31,e32,e33]])
    def rmat(self):
        return self.rotmat

class dhkl(object):
    '''calculate d-spacing for reflection from reciprocal metric tensor
    d = dhkl(lattice,HKL)
    lattice = [a b c alpha beta gamma] (angles in degrees)
    HKL: list of hkl. size(HKL) = n x 3 or 3 x n
    !!! if size(HKL) is 3 x 3, HKL must be in the form: 
    HKL = [h1 k1 l1 ; h2 k2 l2 ; h3 k3 l3]'''
    def __init__(self,lattice,hkl):
        self.lattice = lattice
        self.hkl = np.matrix(hkl)
    def d(self):
        hkl=self.hkl
        if np.shape(hkl)[0] == 3 and np.shape(hkl)[1] != 3:
            hkl=hkl.transpose()
            T=1
        else:
            T=0
        G = bmatrix(self.lattice).mt()
        d = 1./np.sqrt(np.diagonal(hkl*(G.I*hkl.transpose())))
        #d = 1/np.sqrt(hkl*G.I*hkl.T)
        if T==1:
            d = d.transpose()
        return d

class interplanarangle(object):
    def __init__(self,lattice,hkl1,hkl2):
        ''' calculates interplanar angles in degrees for reflections using the metric tensor
        Example interplanarangle(lattice,hkl,hkl2) where hkl and hkl2 must have the same column length
        interplanarangle([3,3,3,90,90,120],[[1,2,3],[1,2,3]],[[1,1,3],[1,2,3]]) '''
        self.lattice = lattice
        if len(hkl1) != len(hkl2):
            hkl1=np.zeros((len(hkl2),3))+hkl1
        self.hkl1=np.matrix(hkl1)
        self.hkl2=np.matrix(hkl2)
    def ang(self):
        G = bmatrix(self.lattice).mt()
        dhkl1 = dhkl(self.lattice,self.hkl1).d()
        dhkl2 = dhkl(self.lattice,self.hkl2).d()
        term1 = np.diagonal(self.hkl1*(G.I*self.hkl2.T))
        term2 = term1*dhkl1*dhkl2
        term2[np.where(term2>1)]=1 # to prevent nans due to rounding errors
        return np.arccos(term2)*180/np.pi
        # return np.arccos(np.multiply((term1*dhkl1),dhkl2))*180/np.pi

class bragg(object):
    def __init__(self,lattice,hkl,energy):
        ''' returns Bragg angle of a reflection
        theta = bragg(lattice,hkl,energy)'''
        self.lattice = lattice
        self.hkl = hkl
        self.energy = energy
    def th(self):
        keV2A = 12.3984187
        wl = keV2A/self.energy
        d = dhkl(self.lattice,self.hkl).d()
#        if wl/2.0/d <= 1:
        theta = 180/np.pi*np.arcsin(wl/2.0/d);
#        else:
#            theta = np.nan;
        return theta

class calcms(object):
    def __init__(self,lattice,hkl,hklint,hkl2,energy,azir,F = [],F2 = []):
        self.F = np.matrix(F)
        self.F2 = np.matrix(F2)
        self.lattice = lattice
        self.hkl = np.matrix(hkl)
        self.hkl2 = np.matrix(hkl2)
        self.hkl3 = hklint-self.hkl2
        self.energy = energy
        self.azir = np.matrix(azir)
        bm = bmatrix(self.lattice).bm()
        #   Convert primary hkl and reduced hkl2 list to orthogonal coordinate system    
        hklnotlist=(bm*self.hkl.transpose()).transpose()
        self.hklrlv=hklnotlist
        azir2=(bm*self.azir.transpose()).transpose()
        zref=np.matrix([[0,0,1]])
        #   Determin transformation to align primary reflection to the z direction
        alignangle=interplanarangle(self.lattice,[0,0,1],self.hkl).ang()
        realvecthkl=(bm*self.hkl2.transpose()).transpose()
        realvecthkl3=(bm*self.hkl3.transpose()).transpose()
        rotvect=np.cross(zref,hklnotlist)
        if np.abs(rotvect[0][0])+np.abs(rotvect[0][1])+np.abs(rotvect[0][2]) >= 0.0001:
            realvecthkl=realvecthkl*rotxyz(rotvect,alignangle[0]).rmat() # multiplication order for rotation towards zref
            self.rmatrix = rotxyz(rotvect,alignangle[0]).rmat()
            self.tvprime = hklnotlist*self.rmatrix
        else:
            self.tvprime = hklnotlist
        #   Build Ewald Sphere
        brag1 = np.empty(self.hkl2.shape[0])*0+1.0*bragg(self.lattice,self.hkl,self.energy).th()
        self.brag1 = brag1
        keV2A = 12.398
        ko=(self.energy/keV2A)
        self.ko = ko
        #   height dependent radius of ewald slice in the hk plane
        rewl=ko*np.cos((np.arcsin(((ko*np.sin(-brag1*np.pi/180.0))+(realvecthkl[:,2]))/ko)*180.0/np.pi)*np.pi/180.0)
        rhk=np.sqrt(np.square(realvecthkl[:,0])+np.square(realvecthkl[:,1]))
        #   Origin of intersecting circle
        orighk = np.empty(self.hkl2.shape[0])*0+(ko*np.cos(brag1[0]*np.pi/180.))
        ####################### MS Calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if np.abs(rotvect[0][0])+np.abs(rotvect[0][1])+np.abs(rotvect[0][2]) > 0.001:
            azir2=azir2*rotxyz(rotvect,alignangle[0]).rmat()
        azirangle=np.arctan2(azir2[0,0],azir2[0,1])*180.0/np.pi
        rhkangle=np.arctan2((realvecthkl[:,0]),(realvecthkl[:,1]))*180.0/np.pi
        yhkintercept=np.divide(np.square(orighk)-np.square(rhk)+np.square(rewl),(2.0*orighk))-orighk
        xintercept=np.sqrt(np.square(rewl)-np.square(np.divide((np.square(orighk)-np.square(rhk)+np.square(rewl)),2.0*orighk)))
        interceptangle1=np.arctan2(xintercept,yhkintercept)*180.0/np.pi
        interceptangle2=np.arctan2(-xintercept,yhkintercept)*180.0/np.pi #with respect to the real space origin
        self.ewpsi1=interceptangle1+rhkangle
        self.ewpsi2=interceptangle2+rhkangle
        psirotate=(interceptangle1+azirangle-rhkangle)
        psirotate2=(interceptangle2+azirangle-rhkangle)
        self.interceptangle1 = interceptangle1-rhkangle
        self.interceptangle2 = interceptangle2-rhkangle        
        self.rhkangle=rhkangle
        ########## return hkl back to original coordinate system ##############
        psi1 = (np.mod(psirotate+180.0,360.0)-180.0)
        psi1 = psi1[:,0]
        psi2 = (np.mod(psirotate2+180.0,360.0)-180.0)
        psi2 = psi2[:,0]
        brag1=np.matrix(brag1).transpose()
        braga=np.array(brag1)[0]
        self.kov1 =np.array((rotxyz([1,0,0],-np.array(braga)[0]).rmat()*np.matrix([[0,self.ko,0]]).T).T)
        self.psi1 = psi1
        self.psi2 = psi2
        self.bragg1 = brag1
        energyl=np.matrix(np.ones(psi1.shape[0])*energy).T
        if len(F) == 0:
            self.fullarray = np.array(np.concatenate((hkl2,psi1,psi2,brag1,energyl),1))
        else:
            self.fullarray = np.array(np.concatenate((hkl2,psi1,psi2,brag1,(self.F).T,energyl),1))
        self.realvecthkl = realvecthkl
        self.realvecthkl3 = realvecthkl3
        self.ko=ko
    def tv(self):
        return self.realvecthkl
    def tvt(self):
        return self.realvecthkl3
    def rhkangle(self):
        return self.rhkangle
    def prlv(self):
        return self.hklrlv
    def kov(self):
        return self.kov1
    def ko(self):
        return self.ko
    def psi(self):
        return np.concatenate((self.psi1[:,0],self.psi2[:,0]),1)
        #return self.psi1, self.psi2
    def ewpsi(self):
        return self.ewpsi1, self.ewpsi2
    def bragg(self):
        return np.array(self.bragg1)
    def full(self):
        ''' returns hkl2,psi1,psi2,brag1,energ '''
        return self.fullarray
    def trv(self):
        ''' returns transformed and rotated vectors. '''
        trvarray=np.array([rotxyz([0,0,1],np.array(self.ewpsi1[i1,:])[0][0]).rmat()*self.realvecthkl[i1,:].T for i1 in range(self.ewpsi1.shape[0])])
        trvarray2=np.array([rotxyz([0,0,1],np.array(self.ewpsi2[i1,:])[0][0]).rmat()*self.realvecthkl[i1,:].T for i1 in range(self.ewpsi2.shape[0])])
        return np.matrix(np.squeeze(trvarray)), np.matrix(np.squeeze(trvarray2))
    def trvt(self):
        ''' returns transformed and rotated tertiary vectors. '''
        trvarrayt=np.array([rotxyz([0,0,1],np.array(self.ewpsi1[i1,:])[0][0]).rmat()*self.realvecthkl3[i1,:].T for i1 in range(self.ewpsi1.shape[0])])
        trvarray2t=np.array([rotxyz([0,0,1],np.array(self.ewpsi2[i1,:])[0][0]).rmat()*self.realvecthkl3[i1,:].T for i1 in range(self.ewpsi2.shape[0])])
        return np.matrix(np.squeeze(trvarrayt)), np.matrix(np.squeeze(trvarray2t))
    def bvects(self):
        ''' returns secondary beam vectors '''
        return self.trv()[0]+self.kov1,self.trv()[1]+self.kov1
    def bvects2(self):
        ''' returns tertiary beam vectors '''
        return self.trvt()[0]+self.bvects()[0],self.trvt()[1]+self.bvects()[1]
    def angs(self):
        ''' Angles between ko and beam vectors '''
        norms1=np.apply_along_axis(np.linalg.norm, 1, self.bvects()[0])
        angs1=np.arccos((np.matrix(-self.kov())*np.matrix(self.bvects()[0]).T)/(LA.norm(self.kov())*norms1))*180.0/np.pi
        norms2=np.apply_along_axis(np.linalg.norm, 1, self.bvects()[1])
        angs2=np.arccos((np.matrix(-self.kov())*np.matrix(self.bvects()[1]).T)/(LA.norm(self.kov())*norms2))*180.0/np.pi
        return angs1, angs2
    def psiplaneang(self):
        ''' Angle required to rotate k1 about ko onto the secondary scattering plane '''
        v1=np.matrix([[1,0,0]]) # determines slice direction of interplanerangle function
        norms1=np.apply_along_axis(np.linalg.norm, 1, self.bvects()[0])
        nbv=(self.bvects()[0].T/norms1).T # normalized beam vectors
        v2=np.cross(-self.kov(),nbv)
        psiangs=interplanarangle([1,1,1,90,90,90],v1,v2).ang()
        return psiangs
    def psiplaneang2(self):
        ''' Angle required to rotate k2 about k1 onto the tertiary scattering plane '''       
        norms1=np.apply_along_axis(np.linalg.norm, 1, self.bvects()[0])
        norms2=np.apply_along_axis(np.linalg.norm, 1, self.bvects2()[0])
        nbv1=np.cross(-self.kov(),(self.bvects()[0].T/norms1).T)
        nbv2=np.cross((self.bvects()[0].T/norms1).T,(self.bvects2()[0].T/norms2).T)
        psiangs2=interplanarangle([1,1,1,90,90,90],nbv1,nbv2).ang()
        return psiangs2
    def pol(self,polv):
        ''' returns hkl2, sig, pi, pfactor   '''
        refs=self.fullarray[:,[0,1,2]]
        braggs=bragg(self.lattice,refs,self.energy).th()
        psiang=self.psiplaneang()
        pmtmpv=np.array(np.squeeze([(np.matrix([[1,0],[0,np.cos(2*braggs[i1]*np.pi/180.0)]])* \
                        np.matrix([[np.cos(psiang[i1]*np.pi/180.0),np.sin(psiang[i1]*np.pi/180.0)], \
                        [-np.sin(psiang[i1]*np.pi/180.0),np.cos(psiang[i1]*np.pi/180.0)]])*np.matrix(polv).T).T \
                        for i1 in range(braggs.shape[0])]))
        sums=np.matrix(np.sum((pmtmpv)**2,1)).T
        return np.concatenate((pmtmpv,sums),1)
#     def pol2(self,polv):
#         ''' returns hkl3, sig, pi, pfactor   '''
#         refs=self.fullarray[:,[0,1,2]]
#         polv2=self.pol(polv)[:,[-3,-2]]
#         brags2=bragg(self.lattice,self.hkl-refs,self.energy).th()
#         psiang2=self.psiplaneang2()
#         pmtmpv2=np.array(np.squeeze([(np.matrix([[1,0],[0,np.cos(2*brags2[i1]*np.pi/180.0)]])* \
#                         np.matrix([[np.cos(psiang2[i1]*np.pi/180.0),np.sin(psiang2[i1]*np.pi/180.0)],\
#                         [-np.sin(psiang2[i1]*np.pi/180.0),np.cos(psiang2[i1]*np.pi/180.0)]])* \
#                         np.matrix(polv2[i1,[0,1]]).T).T for i1 in range(brags2.shape[0])]))
#         sums2=np.matrix(np.sum((pmtmpv2)**2,1)).Tko=(self.energy/keV2A)
#         return np.concatenate((pmtmpv2,sums2),1)
    
    def pol2(self,polv):
        ''' returns hkl3, sig, pi, pfactor   '''
        refs=self.fullarray[:,[0,1,2]]
        brags=bragg(self.lattice,refs,self.energy).th()
        brags2=bragg(self.lattice,self.hkl-refs,self.energy).th()
        psiang=self.psiplaneang()
        psiang2=self.psiplaneang2()
        pmtmpv2=np.array(np.squeeze([(np.matrix([[1,0],[0,np.cos(2*brags2[i1]*np.pi/180.0)]])* \
                        np.matrix([[np.cos(psiang2[i1]*np.pi/180.0),np.sin(psiang2[i1]*np.pi/180.0)], \
                        [-np.sin(psiang2[i1]*np.pi/180.0),np.cos(psiang2[i1]*np.pi/180.0)]])* \
                        np.matrix([[1,0],[0,np.cos(2*brags[i1]*np.pi/180.0)]])* \
                        np.matrix([[np.cos(psiang[i1]*np.pi/180.0),np.sin(psiang[i1]*np.pi/180.0)], \
                        [-np.sin(psiang[i1]*np.pi/180.0),np.cos(psiang[i1]*np.pi/180.0)]])*np.matrix(polv).T).T \
                        for i1 in range(brags2.shape[0])]))
        sums2=np.matrix(np.sum((pmtmpv2)**2,1)).T
        return np.concatenate((pmtmpv2,sums2),1)

#     def polfull(self,polv):
#         ''' returns hkl2,psi1,psi2,brag1,energy, sig, pi, pfactor, pfactor*F   '''
# #         return np.concatenate((self.full(),self.pol(polv)[:,[-3,-2,-1]],(self.pxf(polv)).T),1)
#         return np.concatenate((self.full(),self.pol2(polv),(self.pxf(polv)).T),1)
#     def pol2full(self,polv):
#         ''' returns hkl2,psi1,psi2,brag1,energy, sig, pi, pfactor, pfactor*F  using '''
#         return np.concatenate((self.full(),self.pol(polv)[:,[-3,-2,-1]],(self.pxf(polv)).T),1)
#     def polfull2(self,polv):
#         ''' returns hkl2,psi1,psi2,brag1,energy, sig, pi, pfactor, pfactor*F   '''
# #         return np.concatenate((self.full(),self.pol(polv)[:,[-3,-2,-1]]),1)
    def pv1xsf1(self,polv):
        ampT=np.array(self.F.T)*np.array(self.pol(polv)[:,-1])
        return np.concatenate((self.full(),ampT),1)
    def geometry(self):
        return self.full()
    def polfull(self,polv):
        ampT=np.array(self.F.T)*np.array(self.F2.T)*np.array(self.pol2(polv)[:,-1])
        return np.concatenate((self.full(),ampT),1)
    def polfull2(self,polv):
        ''' returns hkl2,psi1,psi2,brag1,energy, sig, pi, pfactor, pfactor*F   '''
        return np.concatenate((self.full(),self.pol(polv)),1)
    def sfonly(self):
        ampT = np.array(self.F.T)*np.array(self.F2.T)
        return np.concatenate((self.full(),ampT),1)
    def sf1only(self):
        ampT = np.array(self.F.T)
        return np.concatenate((self.full(),ampT),1)
    def pol1only(self,polv):
        ampT=self.pol(polv)[:,-1]
        return np.concatenate((self.full(),ampT),1)
    def pol2only(self,polv):
        ampT=self.pol2(polv)[:,-1]
        return np.concatenate((self.full(),ampT),1)
    def SF(self):
        return self.F
    def SF2(self):
        return self.F2
    def pxf(self,polv):
        return np.array(self.SF())*np.array(self.pol(polv)[:,-1]).T
    def ov(self):
        ''' returns original vector list. '''
        return self.hkl2
    def orig(self):
        ''' Returns reciprocal space origin. '''
        return np.matrix([0,self.ko*np.cos(self.brag1[0]*np.pi/180.0),-self.ko*np.sin(self.brag1[0]*np.pi/180.0)])
    def kv(self):
        return self.ko
    def tvp(self):
        return self.tvprime
    def ppsi(self):
        return np.concatenate((self.interceptangle1,self.interceptangle2),1)
#     def SF2(self):
#         reflist2=vfind
    def th(self):
        return np.arcsin((self.kov()+self.trv())[0][:,2]/self.ko)*180/np.pi

# class hklgen(object):
#     def __init__(self,depth):
#         self.depth=depth
#     def v(self):
#         depth=self.depth
#         reflist=np.zeros((((2*depth)+1)**3)*3).reshape(((((2*depth)+1)**3)*3)/3,3)
#         list1=[x+1 for x in range(-depth-1,depth)]
#         clist=it.cycle(list1)
#         for hh in range(depth,(((2*depth)+1)**3)-depth,(2*depth+1)): #2 times depth +1
#             reflist[[hh+x+1 for x in range(-depth-1,depth)],0]=[x+1 for x in range(-depth-1,depth)]
#         for kk in range(depth,(((2*depth)+1)**3)-depth,(2*depth+1)): #2 times depth +1
#             reflist[[kk+x+1 for x in range(-depth-1,depth)],1]=clist.next()
#         for kk in range(depth,(((2*depth)+1)**3)-depth,(2*depth+1)): #2 times depth +1
#             reflist[[kk+x+1 for x in range(-depth-1,depth)],2]=clist.next()
#         reflist[:,2].sort()
#         return reflist.astype(int)
    
    
class pilkhlrange(object):
    def __init__(self,lattice,hkl,energy,botangle,topangle):
        self.lattice = lattice
        self.hkl = np.matrix(hkl)
        keV2A = 12.3984187
        ko=energy/keV2A
        bm = bmatrix(self.lattice).bm()
        self.invbm = bmatrix(self.lattice).ibm()
        hklnotlist=(bm*self.hkl.transpose()).transpose()
        topscale=2.0*ko*np.sin(topangle*np.pi/180)
        botscale=2.0*ko*np.sin(botangle*np.pi/180)
        normedvect=hklnotlist/LA.norm(hklnotlist)
        self.pildeltarange1=np.array([normedvect*botscale,normedvect*topscale])
        self.hklr = (self.invbm*self.pildeltarange1.transpose()).transpose()
    def hklrange(self):
        return self.hklr
    def hklscan(self,numsteps):
        hklempty=np.zeros((numsteps,3))
        hklempty[:,0]=np.linspace(self.hklr[0,0],self.hklr[1,0],numsteps)
        hklempty[:,1]=np.linspace(self.hklr[0,1],self.hklr[1,1],numsteps)
        hklempty[:,2]=np.linspace(self.hklr[0,2],self.hklr[1,2],numsteps)
        return hklempty


def dms2px(detv1,detv2,o,v):
    ''' usage dms2px(detector vector 1,detector vector2, sample origin as vector, vectors which will be scaled to intersect detector'''
    v=np.array(v)
    n=np.cross(detv1,detv2)
    D=-(n[0]*detv1[0]+n[1]*detv1[1]+n[2]*detv1[2])# scalar equation for plane
    t=-(n[0]*o[0]+n[1]*o[1]+n[2]*o[2]+D)/(n[0]*v[:,0]+n[1]*v[:,1]+n[2]*v[:,2])# scalar from parametric representation of vectors 
    return (t*v.T).T+o # returns intersection coordinates


# ── Detector mounting: which image axis the scattering plane runs along ────────
# The engines work in a detector frame whose component 0 is normal to the
# scattering plane and component 2 lies in it (along 2θ).  How those land on
# the image depends on how the detector sits on the arm:
#
#   vertical   (default) — component 0 → rows, component 2 → columns.  For the
#              Pilatus 100K in vertical scattering, 2θ (δ) runs along the long
#              axis and an azimuthal scan sweeps the DMS lines across the short
#              axis.  Every existing config is in this mode.
#   horizontal — the same detector used in horizontal scattering: component 2
#              → rows (2θ along the short axis), component 0 → columns, so an
#              azimuthal scan sweeps the lines along the long axis.
#
# `scatv` keeps its meaning (bit 0: the sign of the in-plane component), and the
# mounting rides in bit 1, so the existing positional argument tuples — builder
# args, engine args, the joblib workers in fit.py — carry it unchanged.  Only
# `scatv_code` builds the combined value and only the two helpers below read it.
DET_ORIENTATIONS = ('vertical', 'horizontal')
SCATV_HORIZONTAL = 2


def det_orientation(name):
    '''Validate a detector-orientation name (None → 'vertical').'''
    name = 'vertical' if name is None else str(name).strip().lower()
    if name not in DET_ORIENTATIONS:
        raise ValueError('det_orientation must be one of %s, not %r'
                         % (DET_ORIENTATIONS, name))
    return name


def scatv_code(scatv, orientation='vertical'):
    '''The engines' ``scatv``: the sign bit of the config ``scatv`` plus the
    mounting bit for ``orientation``.'''
    code = int(scatv) & 1
    if det_orientation(orientation) == 'horizontal':
        code |= SCATV_HORIZONTAL
    return code


def scatv_orientation(code):
    '''Split an engine ``scatv`` back into (config scatv, orientation).'''
    code = int(code)
    return code & 1, ('horizontal' if code & SCATV_HORIZONTAL else 'vertical')


def det_to_pixels(p0, p2, px, py, scatv):
    '''Detector-frame components (0: normal to the scattering plane, 2: in it)
    → image (rows, cols) about the origin pixel (px row, py col).'''
    code = int(scatv)
    s2 = p2 if code & 1 else -p2
    if code & SCATV_HORIZONTAL:
        return px + s2, py + p0
    return px + p0, py + s2


def pixels_to_det(rows, cols, px, py, scatv):
    '''Inverse of `det_to_pixels`: image (rows, cols) → components (0, 2).'''
    code = int(scatv)
    if code & SCATV_HORIZONTAL:
        p0, s2 = cols - py, rows - px
    else:
        p0, s2 = rows - px, cols - py
    return p0, (s2 if code & 1 else -s2)


def psith2v(psi,th):
    X=np.sin((90.0-th)*np.pi/180.0)*np.cos((90.0+psi)*np.pi/180.0)
    Y=np.sin((90.0-th)*np.pi/180.0)*np.sin((90.0+psi)*np.pi/180.0)
    Z=np.cos((90.0-th)*np.pi/180.0)
    return np.array([X,Y,Z]).T

def makekernel(func,size, sigma,sigma2 = 1):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    x0 = y0 = size // 2
    if func=='gauss':
        return np.exp(-((x-x0)**2 + (y-y0)**2) / sigma**2)
    elif func=='lorentz':
        return np.pi*0.5*sigma/(((x-x0)**2+(0.5*sigma)**2)+((y-y0)**2+(0.5*sigma)**2))
    elif func=='custom1':
        return np.pi*4.0*sigma/(((x-x0)**2+(0.5*sigma)**2)+((y-y0)**2+(0.5*sigma)**2))**0.25
    elif func=='custom2':
        return np.exp(-((x-x0)**2 + (y-y0)**2) / sigma**2)+np.pi*0.5*sigma2/(((x-x0)**2+(0.5*sigma2)**2)+((y-y0)**2+(0.5*sigma2)**2))

def blur_gauss(img, size, sigma):
    '''`ndimage.convolve(img, makekernel('gauss', size, sigma))`, done as two
    1-D passes.  That kernel is exp(-(dx²+dy²)/σ²) = g(dx)·g(dy), so it
    factors exactly, and the separable form gives the same image (to rounding,
    ~1e-15) at ~10x less cost: size+size multiplies per pixel instead of
    size².  Same boundary mode ('reflect') and origin as the 2-D call.'''
    x = np.arange(0, size, 1, float) - size // 2
    g = np.exp(-x**2 / sigma**2)
    return ndimage.convolve1d(ndimage.convolve1d(img, g, axis=0), g, axis=1)

def gauss(x,sigma, intensity,centre, bg):
    return intensity*np.exp(-(((x)-centre)**2/(2*sigma**2)))+bg
    
def gauss2(x,sigma1, sigma2, intensity1, intensity2, centre1, centre2, bg):
    return (intensity1*np.exp(-(((x)-centre1)**2/(2*sigma1**2))))+(intensity2*np.exp(-(((x)-centre2)**2/(2*sigma2**2))))+bg

def fitgauss(xdata,ydata):
    sigma=(xdata[np.gradient(ydata,3).argmax()]-xdata[np.gradient(ydata,3).argmin()])/2.3548
    intensity=ydata.max()-ydata.min()
    centre=xdata[ydata.argmax()]
    bg=ydata.min()
    fitcoeffs, pcov = curve_fit(gauss,xdata,ydata,[sigma,intensity,centre,bg])
    fitpoints=gauss(xdata,fitcoeffs[0],fitcoeffs[1],fitcoeffs[2],fitcoeffs[3])
    return fitcoeffs, pcov, fitpoints
    
def fitgauss1from2(xdata,ydata,sig):
    sigma=(xdata[np.gradient(ydata,2).argmin()]-xdata[np.gradient(ydata,2).argmax()])/2.3548
    intensity=ydata.max()-ydata.min()
    bg=ydata.min()
    if abs(sigma) > sig:
        centre1=((xdata[np.gradient(ydata,2).argmax()]+xdata[np.gradient(ydata,2).argmin()])/2)-sigma
        centre2=centre1+(2*sigma)
        sigma1=sigma2=sigma/2.0
        intensity1=intensity2=intensity/2.0
        fitcoeffs, pcov = curve_fit(gauss2,xdata,ydata,[sigma1,sigma2,intensity1,intensity2,centre1,centre2,bg])
        if fitcoeffs[0]+fitcoeffs[0]>sig*10:
            fitcoeffs, pcov = curve_fit(gauss2,xdata,ydata,[sigma1/2.0,sigma2/2.0,intensity1,intensity2,centre1,centre2,bg])
        if np.abs(fitcoeffs[0]*fitcoeffs[2])>np.abs(fitcoeffs[1]*fitcoeffs[3]):
            fitcoeffs=fitcoeffs[np.r_[:5:2,-1]]
            pcov=pcov[np.r_[:5:2,-1],:4]
        else:
            fitcoeffs=fitcoeffs[np.r_[1:6:2,-1]]
            pcov=pcov[np.r_[1:6:2,-1],:4]
    else:
        centre=xdata[ydata.argmax()]
        fitcoeffs, pcov = curve_fit(gauss,xdata,ydata,[sigma,intensity,centre,bg])
    fitpoints=gauss(xdata,fitcoeffs[0],fitcoeffs[1],fitcoeffs[2],fitcoeffs[3])
    return fitcoeffs, pcov, fitpoints

def centroid(xdata,ydata):
    '''Background-subtracted centre-of-mass peak position, returned in the same
    (coef, pcov, fitpoints) shape as fitgauss/fitgauss1from2 so it can be used
    as a drop-in alternative.  coef = [sigma, intensity, centre, bg] with
    coef[2] the centroid; sigma is the RMS width about the centroid.'''
    xdata=np.asarray(xdata,dtype=float)
    ydata=np.asarray(ydata,dtype=float)
    bg=ydata.min()
    w=ydata-bg
    total=w.sum()
    if total<=0:
        centre=float(xdata[ydata.argmax()])
        sigma=1.0
    else:
        centre=float((xdata*w).sum()/total)
        var=(w*(xdata-centre)**2).sum()/total
        sigma=float(np.sqrt(var)) if var>0 else 1.0
    intensity=float(ydata.max()-bg)
    fitcoeffs=np.array([sigma,intensity,centre,float(bg)])
    pcov=np.zeros((4,4))
    fitpoints=gauss(xdata,sigma,intensity,centre,bg)
    return fitcoeffs, pcov, fitpoints

# `sig` threshold handed to peakfit for the ROI curves: above it, fitgauss1from2
# treats the curve as a blended doublet, fits two Gaussians and keeps the
# stronger.  Deliberately None — automatic doublet splitting is DISABLED.
#
# A doublet in an experimental ROI comes from multiple phases in the sample,
# each contributing its own DMS line.  Picking the stronger component is
# therefore a phase assignment, and the code has no basis for making it: the
# stronger line is not necessarily the phase being refined.  That call belongs
# to the user, who sets the centre for such a ROI by right-clicking it in the
# curve grid (the manual centre override, which survives rebuilds and fits).
#
# The simulation is single-phase and produces one line per ROI, so it has no
# doublet to resolve either way.  Both sides take their `sig` from here — the
# experimental extraction via multiroifit/multiroifit2, the simulated one via
# dmsfit_ico_hkl.peaksig — so the target and the value driven onto it are always
# extracted by the same estimator.
AUTO_DOUBLET_SIG = None

# Multiple of the ROI width charged as the residual for a ROI whose simulated
# peak could not be located.  Must be > 1 so that a failure is always worse than
# the largest miss a located peak can produce (the centre cannot fall outside
# the integrated curve, so that bound is the width itself).
ROI_FAIL_PENALTY_FACTOR = 2.0

def centre_residuals(sim_centres, target_centres, fail_penalty):
    '''Per-ROI centre residuals from already-extracted peak centres.

    The arithmetic behind the image fit's objective, factored out so a caller
    that has already located the simulated peaks (the slider's live readout,
    which integrates the simulated image for plotting anyway) can score them
    without a second imcalc — and, more importantly, cannot drift from what the
    fit actually minimises.  dmsfit_ico_hkl._centre_residuals is the same
    function with the peak extraction attached.

    NaN in `target_centres` means no experimental peak was locatable, so that
    ROI has no target and is dropped; NaN in `sim_centres` means the simulated
    peak was not locatable, which is a real failure of the current parameters
    and is charged `fail_penalty`.

    Returns (residual vector, n_sim_failed, n_no_target).'''
    sim = np.asarray(sim_centres, dtype=float)
    tgt = np.asarray(target_centres, dtype=float)
    has_target = ~np.isnan(tgt)
    sim_failed = has_target & np.isnan(sim)
    resid = np.where(has_target, sim - tgt, 0.0)
    resid = np.where(sim_failed, float(fail_penalty), resid)
    return resid, int(sim_failed.sum()), int((~has_target).sum())

def peakfit(xdata,ydata,method='gauss',sig=None):
    '''Dispatch peak-position extraction by method name: 'centroid' uses the
    centre of mass, anything else uses Gaussian curve fitting (the two-peak
    aware fitgauss1from2 when sig is given, otherwise fitgauss).'''
    if method=='centroid':
        return centroid(xdata,ydata)
    if sig is not None:
        return fitgauss1from2(xdata,ydata,sig)
    return fitgauss(xdata,ydata)

def uniquearray(inarray):
    tup=tuple(map(tuple, inarray))
    reducedtup = list(set(tup))
    return np.array(reducedtup)

def reducebypsirange(mslist,psirange):
        keepindex=np.where([~np.isnan(mslist).any(1)])[1]
        mslist=mslist[keepindex,:]
        mslist1=np.squeeze(mslist[np.where(mslist[:,3] >= psirange[0])[0],:])
        mslist1=np.array(np.squeeze(mslist1[np.where(mslist1[:,3] <= psirange[1])[0],:]))
        mslist2=np.squeeze(mslist[np.where(mslist[:,4] >= psirange[0])[0],:])
        mslist2=np.array(np.squeeze(mslist2[np.where(mslist2[:,4] <= psirange[1])[0],:]))
        mslist1=np.delete(mslist1, 4, axis=1)
        mslist2=np.delete(mslist2, 3, axis=1)
        return np.concatenate((mslist1, mslist2),0)

def cmap():
    return OrderedDict([('GS','Gray Scale'),('pm3d','Traditional pm3d (black-blue-red-yellow)'),
                    ('hot','Hot (black-red-yellow-white)'),('FNS','Film Negative Sqrt'),
                    ('jet','Jet (Blue-Cyan-Green-Yellow-Red)'),('SFN','Squared Film Negative'),
                    ('ocean','Ocean (green-blue-white)'),('NCD','NCD'),
                    ('rainbow','Rainbow (blue-green-yellow-red)'),('afm','AFM hot (black-red-yellow-white)')])

def roi_dedupe_path(pts):
    '''Drop repeat visits to a pixel, keeping the path in first-seen order.

    The engine walks both psi solutions of a reflection and concatenates them.
    In many geometries the two trace the *same* detector line, so the raw index
    is that line twice over: the path then looks like two pieces (or, once
    ordered by a detector axis, like one line with every pixel doubled).  One
    visit per pixel is what a ROI is built from.'''
    pts = np.asarray(pts)
    if len(pts) < 2:
        return pts
    w = int(pts[:, 1].max()) + 1
    key = pts[:, 0].astype(np.int64) * w + pts[:, 1].astype(np.int64)
    _, first = np.unique(key, return_index=True)
    return pts[np.sort(first)]


def roi_split_runs(pts, min_pts=4, gap_factor=6.0, gap_floor=4.0):
    '''Cut one reflection's on-detector locus into its continuous pieces.

    `pts` is the pixel path in *scan order* (as `dmscalc_ico_hkl.roiindex`
    returns it), so consecutive entries are consecutive samples of the same
    line — except where the locus leaves the physical region or the detector and
    comes back, or where the second psi branch starts (the engine concatenates
    both branches).  Those show up as a jump far larger than the sampling step,
    and are the cuts made here.  Returns a list of index arrays, longest first.

    The threshold is 6x the median step, floored at 4 px so a densely sampled
    line whose median step is a fraction of a pixel is not chopped at every
    slight unevenness.'''
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return [np.arange(len(pts))] if len(pts) >= min_pts else []
    d  = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    nz = d[d > 0]
    thr = max(gap_factor * float(np.median(nz)), gap_floor) if nz.size else gap_floor
    runs = np.split(np.arange(len(pts)), np.flatnonzero(d > thr) + 1)
    runs = [r for r in runs if len(r) >= min_pts]
    return sorted(runs, key=len, reverse=True)


def roi_rasterise(path, imshape):
    '''Binary image of a pixel path, consecutive points joined by a straight
    segment so the gaps between scan samples are filled.  Points off the image
    are dropped (the caller's line may run past the plate edge).'''
    im = np.zeros(imshape)
    p = np.asarray(path, dtype=float)
    if len(p) == 0:
        return im
    if len(p) == 1:
        rr, cc = p[:, 0], p[:, 1]
    else:
        a, b = p[:-1], p[1:]
        steps = np.maximum(np.abs(b - a).max(axis=1), 1).astype(int)
        rr = np.concatenate([np.linspace(a[i, 0], b[i, 0], steps[i] + 1)
                             for i in range(len(a))])
        cc = np.concatenate([np.linspace(a[i, 1], b[i, 1], steps[i] + 1)
                             for i in range(len(a))])
    r = np.round(rr).astype(int)
    c = np.round(cc).astype(int)
    ok = (r >= 0) & (r < imshape[0]) & (c >= 0) & (c < imshape[1])
    im[r[ok], c[ok]] = 1
    return im


def roibuilder_ico_hkl(args):
    #builderargs=reflist,hkllist,hklint,1,psirange,100,hkl,detvects,imdata.shape,simsigma,azir,psi,px,py,scatv,detdistancepx,rotx,roty,rotz,energy,ig,reflist2,mtrx2
    #                ref,hkllist,hklint,1,psirange,100,hkl,detvects,emptyim,     simsigma,azir,psi,px,py,scatv,detdistancepx,rotx,roty,rotz,energy,   reflist2,mtrx2
    ''''reflist,hkllistmask,hklint,1,psirange,threshold,hkl,detvects,imshape,0,azir,psi,px,py,scatv'''

    reflist=args[0]    # parallel component of Bragg reflection
    hkllist=args[1]
    hklint=args[2]
    intensity=args[3]
    psirange=args[4]
    threshold=args[5]
    hkl=args[6]
    detvects=args[7]
    imshape=args[8]
    simsigma=args[9]
    azir=args[10]
    psi=args[11]
    px=args[12]
    py=args[13]
    scatv=args[14]
    detdistancepx=args[15]
    rotx=args[16]
    roty=args[17]
    rotz=args[18]
    energy=args[19]
    ig=args[20]
    reflist2=args[21]   # perpemdicular component of Bragg reflection
    mtrx2=args[22]      # 3x3 phason matrix
    # Optional crystal system (conventional crystals only): when set, the ROI
    # geometry uses the constrained full lattice instead of cubic a=b=c.
    crystal_system = args[23] if len(args) > 23 else None

    numrefs=reflist.shape[0]
    reflist2_arr=np.asarray(reflist2)
    # Planes only for the ROIs that are kept, as uint8: a full-frame float64
    # plane per *candidate* ROI (then a fancy-indexed copy of the kept ones)
    # needed ~7 GB on a zoomed Pilatus 2M with ~45 reflections, most of whose
    # lines miss the plate.  Every consumer only tests `> 0`, slices or sums.
    planes = []
    for i1 in range(0,numrefs,1):
        ref=reflist[i1,:]
        # The perpendicular component of *this* reflection, not the whole list.
        # PhasonDistoArray broadcasts v1 + pm@v2 over v2's rows, so handing it
        # all N of them built every ROI from N loci (N copies of the same curve
        # in a conventional crystal, where the perpendicular component is zero;
        # N slightly different ones for a quasicrystal, each carrying another
        # reflection's phason shift).  Everything downstream then worked on an
        # N-fold path.
        ref2=reflist2_arr[i1:i1+1,:] if reflist2_arr.ndim == 2 else reflist2_arr
        emptyim=np.zeros(imshape)
        dmsroi = dmscalc_ico_hkl([ref],hkllist,hklint,1,psirange,100,hkl,detvects,emptyim,simsigma,azir,psi,px,py,scatv,detdistancepx,rotx,roty,rotz,energy,ref2,mtrx2)
        dmsroi.crystal_system = crystal_system

        roiindex=dmsroi.roiindex(ig)

        # Two ROIs per reflection: the locus is cut in half *along itself* and
        # each half becomes a kernel plane.  A rigid shift of the line moves both
        # halves the same way and a rotation moves them oppositely, which is
        # where the fit's sensitivity to the line's orientation comes from.
        #
        # The halves are taken along the path in scan order, on the longest
        # continuous run of it.  The locus is not always one tidy curve: it can
        # leave the physical region or the detector and come back, and the engine
        # concatenates both psi branches, so a reflection's index can hold several
        # far-apart pieces.  Ordering those by a detector axis and cutting at the
        # median (what this did before) interleaves them, and joining the result
        # up drew ROIs with tails shooting hundreds of pixels across the plate —
        # into which msroi then integrated whatever they crossed, and from which
        # it took a meaningless perpendicular direction.  Cutting on the gaps
        # first and keeping one connected arc is what makes a ROI a strip along
        # a line.
        roiindex = roi_dedupe_path(roiindex)
        runs = roi_split_runs(roiindex)
        if runs:
            run = runs[0]
            if len(runs) > 1:
                print('ROI %d: locus is in %d pieces on the detector; using the '
                      'longest (%d of %d points).'
                      % (i1, len(runs), len(run), len(roiindex)))
            half = len(run) // 2
            roi1 = roi_rasterise(roiindex[run[:half]], imshape)
            roi2 = roi_rasterise(roiindex[run[half:]], imshape)
            if roi1.any() and roi2.any():
                planes.append(np.asarray(roi1 > 0, dtype=np.uint8))
                planes.append(np.asarray(roi2 > 0, dtype=np.uint8))
            else:
                print('ROI '+str(i1)+' removed because a half is off the detector.')
        else:
            print('ROI '+str(i1)+' removed because lines miss the detector.')
    if planes:
        return np.stack(planes, axis=2)
    print('No ROIS used!')
    return np.ones((imshape[0], imshape[1], numrefs*2), dtype=np.uint8)
def roi_kernel_indices(kernel):
    '''Pixel indices of every ROI in a (rows, cols, n_roi) kernel, one
    `np.where` tuple per plane — exactly what `msroi` would find.

    The kernel is fixed for the whole of a fit, but searching a full detector
    plane for its few hundred lit pixels costs ~8 ms on a Pilatus 2M, which
    was ~40% of every fit evaluation.  Find them once and pass the tuple to
    `msroi` in place of the plane.'''
    kernel = np.asarray(kernel)
    return [np.where(kernel[:, :, i] > 0) for i in range(kernel.shape[2])]


def msroi(img, kernel, width):
    '''Integrate `img` across one ROI.  `kernel` is the ROI's 2D plane, or its
    pixel indices as `roi_kernel_indices` returns them (same result, no
    full-frame search).'''
    vs_idx = kernel if isinstance(kernel, tuple) else np.where(kernel > 0)
    dv = np.array([[vs_idx[0][-1] - vs_idx[0][0], vs_idx[1][-1] - vs_idx[1][0]]], dtype=float)
    v = (dv @ np.array([[0, 1], [-1, 0]])).flatten()
    v = v / np.linalg.norm(v)
    vs = np.stack([vs_idx[0], vs_idx[1]], axis=1).astype(float)

    irange = np.arange(int(np.round(-width / 2.0)), int(np.round(width / 2.0)))
    offsets = np.outer(irange, v)                                              # (W, 2)
    shifted = np.round(vs[np.newaxis] + offsets[:, np.newaxis]).astype(int)   # (W, N, 2)

    valid = ((shifted[:, :, 0] > 0) & (shifted[:, :, 0] < img.shape[0]) &
             (shifted[:, :, 1] >= 0) & (shifted[:, :, 1] < img.shape[1]))
    r0 = np.clip(shifted[:, :, 0], 0, img.shape[0] - 1)
    r1 = np.clip(shifted[:, :, 1], 0, img.shape[1] - 1)
    vals = np.where(valid, img[r0, r1], 0.0)
    v1 = vals.sum(axis=1, keepdims=True)                                       # (W, 1)
    w_idx, n_idx = np.where(valid)
    v2 = shifted[w_idx, n_idx]                                                 # (M, 2)
    return v1, v2

def msroi2(img, kernel, width):
    ''' Kernel should be 2D array'''
    vs_idx = np.where(kernel > 0)
    dv = np.array([[vs_idx[0][-1] - vs_idx[0][0], vs_idx[1][-1] - vs_idx[1][0]]], dtype=float)
    v = (dv @ np.array([[0, 1], [-1, 0]])).flatten()
    v = v / np.linalg.norm(v)
    vs = np.stack([vs_idx[0], vs_idx[1]], axis=1).astype(float)

    irange = np.arange(int(np.round(-width / 2.0)), int(np.round(width / 2.0)))
    offsets = np.outer(irange, v)                                              # (W, 2)
    shifted = np.round(vs[np.newaxis] + offsets[:, np.newaxis]).astype(int)   # (W, N, 2)

    valid = ((shifted[:, :, 0] > 0) & (shifted[:, :, 0] < img.shape[0]) &
             (shifted[:, :, 1] >= 0) & (shifted[:, :, 1] < img.shape[1]))
    r0 = np.clip(shifted[:, :, 0], 0, img.shape[0] - 1)
    r1 = np.clip(shifted[:, :, 1], 0, img.shape[1] - 1)
    vals = np.where(valid, img[r0, r1], 0.0)
    v1 = vals.sum(axis=1, keepdims=True)                                       # (W, 1)
    w_idx, n_idx = np.where(valid)
    v2 = shifted[w_idx, n_idx]                                                 # (M, 2)
    return v1, v2, v


# ── Sub-pixel ROI integration ──────────────────────────────────────────────────
# `msroi` samples the strip at whole-pixel offsets across the line (a width of
# 21 gives ~21 bins), takes the nearest pixel at each, and uses one direction
# across the line for the whole ROI (normal to the chord from its first to last
# kernel pixel).  The sub-pixel integrator keeps the strip — same ROI path, same
# width, same x coordinate (pixels from the strip edge, so centres, manual
# centres and AUTO_DOUBLET_SIG keep their meaning) — but
#
#   * samples it every `step` px (ROI_BIN_STEP) with bilinear interpolation,
#   * crosses the line along its *local* normal, from a smoothed copy of the
#     path, so a curved line is not cut obliquely toward the ROI's ends,
#   * fixes all of that once per ROI (`roi_sampler`), so integrating an image
#     is one `map_coordinates` call over every ROI (`roi_profiles`).
#
# The geometry depends only on the kernel, so the data and every simulated
# image are integrated through exactly the same samples — the consistency the
# ROI fit rests on.  `step=None` selects the original whole-pixel `msroi`.
ROI_BIN_STEP = 0.25


def roi_path_stride(step):
    '''Path pixels between samples along a ROI at bin step `step`: one sample
    per 1/ROI_BIN_STEP bins' worth of length, i.e. 1 at the default 0.25 and
    `zoomval` when the step is scaled by the zoom (bin_step_per_zoom) — one
    sample per detector pixel either way.  The image between detector pixels
    is interpolation, so sampling it more densely along the line adds cost and
    no information.'''
    return max(1, int(round(float(step) / ROI_BIN_STEP)))


def roi_bin_step(step):
    '''Normalise a bin-step setting: None, 0, a negative value or 'legacy' mean
    the original whole-pixel integrator (returned as None); anything else is a
    positive bin width in pixels.'''
    if step is None or (isinstance(step, str) and step.lower() == 'legacy'):
        return None
    step = float(step)
    return step if step > 0 else None


def roi_bin_from_config(roi_cfg, zoomval=1):
    '''The image-pixel bin step a config's ``roi`` block asks for:
    ``bin_step_per_zoom`` (default ROI_BIN_STEP) times ``zoomval``, like
    ``width_per_zoom``.  null / 0 is the legacy integrator (None).'''
    per = roi_bin_step((roi_cfg or {}).get('bin_step_per_zoom', ROI_BIN_STEP))
    return None if per is None else per * max(float(zoomval), 1e-9)


def roi_sampler(kernel, width, step):
    '''Where one ROI's profile samples the image, at sub-pixel bins.

    `kernel` is the ROI's 2D plane or its `np.where` index tuple.  Returns a
    dict with ``x`` (B,) bin positions in pixels from the strip edge — the same
    coordinate as the legacy integrator's bin index — and ``coords`` (B, N, 2)
    float (row, col) sample points, one row of N along the path per bin, or
    None for a ROI with fewer than two pixels.'''
    vs_idx = kernel if isinstance(kernel, tuple) else np.where(np.asarray(kernel) > 0)
    if len(vs_idx[0]) < 2:
        return None
    # The chord normal, exactly as msroi takes it: it fixes which way across the
    # line x increases, so curves keep the legacy orientation.
    dv = np.array([vs_idx[0][-1] - vs_idx[0][0], vs_idx[1][-1] - vs_idx[1][0]],
                  dtype=float)
    v0 = np.array([-dv[1], dv[0]])
    if np.linalg.norm(v0) == 0:
        v0 = np.array([1.0, 0.0])
    v0 = v0 / np.linalg.norm(v0)

    path = roi_walk_path(np.stack([vs_idx[0], vs_idx[1]], axis=1)).astype(float)
    n = len(path)
    # Smooth the rasterised path (its pixels zigzag ±0.5 px about the line) and
    # take the tangent over the same window.  Up to ±8 px: long enough to
    # average out the staircase, short enough that the sagitta of a curved line
    # over the window stays well under a tenth of a pixel.
    k = int(np.clip(n // 8, 1, 8))
    i = np.arange(n)
    lo, hi = np.clip(i - k, 0, n - 1), np.clip(i + k, 0, n - 1)
    cs = np.vstack([np.zeros((1, 2)), np.cumsum(path, axis=0)])
    centre = (cs[hi + 1] - cs[lo]) / (hi - lo + 1)[:, None]
    tan = path[hi] - path[lo]
    tl = np.linalg.norm(tan, axis=1)
    chord_tan = np.array([v0[1], -v0[0]])
    tan = np.where(tl[:, None] > 0, tan / np.where(tl > 0, tl, 1)[:, None],
                   chord_tan[None, :])
    nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    nrm = np.where((nrm @ v0)[:, None] < 0, -nrm, nrm)

    irange = np.arange(int(np.round(-width / 2.0)), int(np.round(width / 2.0)))
    if irange.size == 0:
        irange = np.array([0])
    span = float(irange[-1] - irange[0])
    x = np.arange(0.0, span + 1e-9, float(step))
    off = irange[0] + x
    stride = roi_path_stride(step)
    centre, nrm = centre[::stride], nrm[::stride]
    coords = centre[None, :, :] + off[:, None, None] * nrm[None, :, :]
    return {'x': x, 'coords': coords}


def roi_sample_set(samplers, shape):
    '''Precompute the integration of every ROI of `samplers` on an image of
    `shape` as one sparse (bins x pixels) matrix of bilinear weights (zero for
    samples off the plate).  Fixed for a fit, so each evaluation integrates
    every ROI with one sparse mat-vec.'''
    h, w = int(shape[0]), int(shape[1])
    nb = [0 if s is None else s['coords'].shape[0] for s in samplers]
    blocks = [s['coords'] for s in samplers if s is not None]
    if not blocks:
        return {'shape': (h, w), 'nbins': nb, 'M': None}
    P = np.concatenate([b.reshape(-1, 2) for b in blocks])
    bins = np.concatenate([
        (np.repeat(np.arange(b.shape[0]), b.shape[1]) + off)
        for b, off in zip(blocks, np.cumsum([0] + [b.shape[0] for b in blocks])[:-1])])
    # Samples off the plate contribute 0, as the legacy integrator's do; one
    # within the outer pixel ring is interpolated from what is on the plate.
    inb = (P[:, 0] >= 0) & (P[:, 0] <= h - 1) & (P[:, 1] >= 0) & (P[:, 1] <= w - 1)
    r = np.clip(P[:, 0], 0, h - 1)
    c = np.clip(P[:, 1], 0, w - 1)
    r0 = np.minimum(np.floor(r).astype(np.int64), h - 2 if h > 1 else 0)
    c0 = np.minimum(np.floor(c).astype(np.int64), w - 2 if w > 1 else 0)
    fr, fc = r - r0, c - c0
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    idx = np.stack([r0 * w + c0, r1 * w + c0, r0 * w + c1, r1 * w + c1])
    wts = np.stack([(1 - fr) * (1 - fc), fr * (1 - fc), (1 - fr) * fc, fr * fc])
    wts = wts * inb[None, :]
    # One sparse (bins x pixels) matrix: every bin's bilinear weights, with the
    # many samples that share a pixel summed once here rather than gathered on
    # every evaluation.  A profile is then a single sparse mat-vec.
    from scipy import sparse
    total = int(sum(nb))
    M = sparse.csr_matrix((wts.ravel(), (np.tile(bins, 4), idx.ravel())),
                          shape=(total, h * w))
    M.sum_duplicates()
    M.eliminate_zeros()
    return {'shape': (h, w), 'nbins': nb, 'M': M}


def roi_profiles(img, samplers, sample_set=None):
    '''Integrate `img` along every ROI of `samplers` (a list of `roi_sampler`
    results, None allowed).  Returns one (B,) profile per ROI (an empty array
    for None): at each bin, the sum along the path of the bilinearly
    interpolated image.  Pass `sample_set` (roi_sample_set for these samplers
    and this image shape) to skip rebuilding the gathers.'''
    img = np.asarray(img, dtype=float)
    ss = sample_set if sample_set is not None else roi_sample_set(samplers, img.shape)
    if tuple(ss['shape']) != tuple(img.shape):
        raise ValueError('sample set was built for a different image shape')
    total = int(sum(ss['nbins']))
    if total == 0:
        return [np.array([]) for _ in samplers]
    prof = ss['M'] @ img.ravel()
    out, start = [], 0
    for n in ss['nbins']:
        out.append(prof[start:start + n] if n else np.array([]))
        start += n
    return out


def roi_sample_pixels(sampler, shape):
    '''The whole pixels a sub-pixel ROI reads, as an (M, 2) int array — the
    analogue of msroi's second return value, for drawing the ROI mask.'''
    if sampler is None:
        return np.empty((0, 2), int)
    P = np.round(sampler['coords'].reshape(-1, 2)).astype(int)
    ok = ((P[:, 0] >= 0) & (P[:, 0] < shape[0]) &
          (P[:, 1] >= 0) & (P[:, 1] < shape[1]))
    return np.unique(P[ok], axis=0)


def roi_curve(img, kernel, width, step=None):
    '''One ROI's integrated curve as (x, y, roi pixels).  `step` None is the
    legacy whole-pixel `msroi` (x the bin index); otherwise the sub-pixel
    integrator at that bin width (x in pixels from the strip edge).'''
    if step is None:
        sumvals, roi = msroi(img, kernel, width)
        return np.arange(len(sumvals)), sumvals[:, 0], roi
    s = roi_sampler(kernel, width, step)
    if s is None:
        raise ValueError('ROI has fewer than two pixels')
    return s['x'], roi_profiles(img, [s])[0], roi_sample_pixels(s, np.shape(img))


def roi_walk_path(vs):
    '''Put a kernel plane's pixels back into path order by walking them.

    `np.where` returns them row by row, which for anything but a steep straight
    line is not the order the line goes in.  Sorting by the dominant detector
    axis fixes the easy cases but not a curved arc, which doubles back in that
    axis and comes out as a zigzag.  Walking 8-connected neighbours from an end
    of the path follows any shape.  A pixel with no unvisited neighbour ends a
    leg; the walk then jumps to the nearest unvisited pixel and carries on, so
    every pixel is ordered even if the kernel is not one connected piece.'''
    vs = np.asarray(vs, dtype=int)
    n = len(vs)
    if n < 3:
        return vs
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(vs)}
    nbrs = [[] for _ in range(n)]
    for (r, c), i in pos.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    j = pos.get((r + dr, c + dc))
                    if j is not None:
                        nbrs[i].append(j)
    # Start from an end of the path (fewest neighbours); a closed loop has none,
    # and any pixel will do there.
    start = int(np.argmin([len(b) for b in nbrs]))
    seen = np.zeros(n, bool)
    order = []
    cur = start
    while cur is not None:
        order.append(cur)
        seen[cur] = True
        nxt = next((j for j in nbrs[cur] if not seen[j]), None)
        if nxt is None:
            left = np.flatnonzero(~seen)
            if len(left) == 0:
                break
            d = np.linalg.norm(vs[left] - vs[cur], axis=1)
            nxt = int(left[np.argmin(d)])
        cur = nxt
    return vs[order]


def roi_outline(kernel, width, step=None):
    '''Closed (rows, cols) outline of the strip `msroi` integrates for one ROI.
    With a sub-pixel `step` it is the strip `roi_sampler` integrates instead:
    the smoothed path offset along its local normals.

    `roibuilder_ico_hkl` lays a one-pixel-wide path along half of a DMS line;
    `msroi` then sums the image along that path at every perpendicular offset in
    `[-width/2, width/2)`, so the region actually integrated is that path swept
    sideways.  This returns its boundary, for drawing the ROI on the detector
    image.

    The perpendicular direction is taken exactly as `msroi` takes it (one fixed
    direction per ROI, from the first and last kernel pixel in row-major order),
    so the drawn box is the region the curve came from and not an idealisation of
    it.  The path is put back into walk order first (`roi_walk_path`), without
    which a curved arc's two edges are drawn as zigzags.'''
    kernel = np.asarray(kernel)
    if step is not None:
        smp = roi_sampler(kernel, width, step)
        if smp is None:
            return np.array([]), np.array([])
        e1, e2 = smp['coords'][0], smp['coords'][-1]
        ring = np.concatenate([e1, e2[::-1], e1[:1]])
        return ring[:, 0], ring[:, 1]
    vs_idx = np.where(kernel > 0)
    if len(vs_idx[0]) < 2:
        return np.array([]), np.array([])
    dv = np.array([[vs_idx[0][-1] - vs_idx[0][0], vs_idx[1][-1] - vs_idx[1][0]]],
                  dtype=float)
    v = (dv @ np.array([[0, 1], [-1, 0]])).flatten()
    nv = np.linalg.norm(v)
    if nv == 0:
        return np.array([]), np.array([])
    v = v / nv
    vs = roi_walk_path(np.stack([vs_idx[0], vs_idx[1]], axis=1)).astype(float)

    irange = np.arange(int(np.round(-width / 2.0)), int(np.round(width / 2.0)))
    if irange.size == 0:
        irange = np.array([0])
    e1 = vs + irange[0] * v
    e2 = vs + irange[-1] * v
    # Down the first edge, back up the second, closed: one polyline the caller
    # can hand straight to a plot item.
    ring = np.concatenate([e1, e2[::-1], e1[:1]])
    # Deliberately not clipped to the image.  A ROI whose strip overhangs the
    # plate edge is integrated only up to the border (msroi drops the offsets
    # that fall outside), but clamping the outline there would fold both edges
    # onto the border and stop it being a simple polygon; the overhang is at
    # most half the width, so it is left visible instead.
    return ring[:, 0], ring[:, 1]


def _centre_in_strip(coef, xdata):
    '''A located centre outside the strip it was integrated from is not a
    measurement — a Gaussian dragged off the curve by a slope or a neighbour.
    Mark it not found (NaN): for the data that drops the ROI's target, for the
    simulation it is charged the fail penalty, as a line that left its ROI.'''
    coef = np.array(coef, dtype=float)
    if len(xdata) and not (xdata[0] <= coef[2] <= xdata[-1]):
        coef[2] = np.nan
    return coef


def _multiroi_subpixel(img, kernel, width, step, method, sig):
    '''multiroifit / multiroifit2 at a sub-pixel bin step: every ROI integrated
    in one batch (roi_profiles), then each curve located with `peakfit`.'''
    n = kernel.shape[2]
    samplers = [roi_sampler(kernel[:, :, i], width, step) for i in range(n)]
    profiles = roi_profiles(img, samplers)
    v1 = np.array([[]]*4).T
    vx, vy, v3, pcovlist = [], [], [], []
    v4 = np.zeros((img.shape[0], img.shape[1], n), dtype=np.uint8)
    for i1, (smp, ydata) in enumerate(zip(samplers, profiles)):
        xdata = smp['x'] if smp is not None else np.array([])
        try:
            coef, pcov, fitpoints = peakfit(xdata, ydata, method, sig)
            coef = _centre_in_strip(coef, xdata)
        except Exception:
            print('Fit not possible for _' + str(i1))
            coef = np.array([0, 0, np.nan, 0])   # NaN centre: no usable peak here
            pcov = np.zeros((4, 4))
            fitpoints = ydata
        v1 = np.vstack([v1, coef])
        vx.append(xdata)
        vy.append(ydata)
        v3.append(fitpoints)
        roi = roi_sample_pixels(smp, img.shape)
        v4[roi[:, 0], roi[:, 1], i1] = 1
        pcovlist.append(pcov)
    return v1, np.array(vx), np.array(vy), np.array(v3), v4, pcovlist


def multiroifit(img,kernel,width,percentileval,method='gauss',sig=None,step=None):
    '''Integrate and locate every ROI of `kernel` in `img`.  `step` is the bin
    width (ROI_BIN_STEP for the sub-pixel integrator); None is the original
    whole-pixel `msroi`.'''
    if step is not None:
        return _multiroi_subpixel(img, kernel, width, step, method, sig)
    v1=np.array([[]]*4).T
    vx=[]
    vy=[]
    v3=[]
    pcovlist=[]
    v4=np.zeros((img.shape[0],img.shape[1],kernel.shape[2]),dtype=np.uint8)
    for i1 in range(kernel.shape[2]):
        sumvals,roi = msroi(img,kernel[:,:,i1],width)
        xdata=np.arange(len(sumvals))
        ydata=sumvals[:,0]
        # xdata=xdata[ydata>np.percentile(ydata, percentileval)]
        # ydata=ydata[ydata>np.percentile(ydata, percentileval)]
#         ydata[ydata<np.percentile(ydata, 10)]=np.percentile(ydata, 10)
        try:
            coef, pcov,fitpoints = peakfit(xdata,ydata,method,sig)
        except Exception:
            print('Fit not possible for _'+str(i1))
            coef = np.array([0,0,np.nan,0])   # NaN centre: no usable peak here
            pcov= np.zeros((4,4))
            fitpoints = ydata
        v1=np.vstack([v1,coef])
        vx.append(xdata)
        vy.append(ydata)
        v3.append(fitpoints)
        v4[roi[:,0].astype(int),roi[:,1].astype(int),i1]=1
        pcovlist.append(pcov)
    return v1, np.array(vx),np.array(vy), np.array(v3), v4, pcovlist
def _multiroifit2_one(img, kernel_slice, width, sig, idx, method='gauss'):
    sumvals, roi, transvect0 = msroi2(img, kernel_slice, width)
    xdata = np.arange(len(sumvals))
    ydata = sumvals[:, 0]
    try:
        coef, pcov, fitpoints = peakfit(xdata, ydata, method, sig)
    except Exception:
        print('Fit not possible for _' + str(idx))
        coef = np.array([0, 0, np.nan, 0])   # NaN centre: no usable peak here
        pcov = np.zeros((4, 4))
        fitpoints = ydata
    return coef, xdata, ydata, fitpoints, roi, pcov

def multiroifit2(img,kernel,width,percentileval,sig,method='gauss',step=None):
    '''The experimental extraction: as multiroifit, ROIs fitted in parallel on
    the legacy path.  `step` as in multiroifit.'''
    if step is not None:
        return _multiroi_subpixel(img, kernel, width, step, method, sig)
    n = kernel.shape[2]
    results = Parallel(n_jobs=-1)(
        delayed(_multiroifit2_one)(img, kernel[:, :, i1], width, sig, i1, method)
        for i1 in range(n)
    )
    v4 = np.zeros((img.shape[0], img.shape[1], n), dtype=np.uint8)
    v1 = np.array([[]]*4).T
    vx, vy, v3, pcovlist = [], [], [], []
    for i1, (coef, xdata, ydata, fitpoints, roi, pcov) in enumerate(results):
        v1 = np.vstack([v1, coef])
        vx.append(xdata)
        vy.append(ydata)
        v3.append(fitpoints)
        v4[roi[:, 0].astype(int), roi[:, 1].astype(int), i1] = 1
        pcovlist.append(pcov)
    return v1, np.array(vx), np.array(vy), np.array(v3), v4, pcovlist
class res(object):
    def __init__(self,x):
        self.x=x
    def x(self):
        return self.x
minimizers = {'Differential Evolution' : 'GA',
             'Nelder-Mead' : 'Nelder-Mead',
             'Newton_CG' : 'Newton-CG',
             'Swarm' : 'SW',
             'Basin Hopping' : 'BH',
             'SLSQP' : 'SLSQP',
             'Powell' :'Powell',
             'CG': 'CG',
             'BFGS':'BFGS',
             'L_BFGS_B' : 'L-BFGS-B',
             'TNC' : 'TNC',
             'dogleg' : 'dogleg',
             'trust_ncg' : 'trust-ncg',
             'SW' : 'SW',
             }


DE_Strategy = {'best1bin':'best1bin',
               'best1exp':'best1exp',
               'rand1exp':'rand1exp',
               'randtobest1exp':'randtobest1exp',
               'best2exp':'best2exp',
               'rand2exp':'rand2exp',
               'randtobest1bin':'randtobest1bin',
               'best2bin':'best2bin',
               'rand2bin':'rand2bin',
               'rand1bin':'rand1bin'}
def im2rgb(*arg):
    if len(arg) > 3:
        print('You can only use one image per channel')
    else:
        imempty=np.zeros((arg[0].shape[0],arg[0].shape[1],3))
        for ii in range(len(arg)):
            imempty[:,:,ii]=arg[ii]
        return imempty
class PhasonDisto(object):
    '''Modifies reflection list according to the phason strain matrix'''    
    def __init__(self,reflist_parallel,reflist_perpendicular,matrix_phason):
        self.reflist_1 = reflist_parallel
        self.reflist_2 = reflist_perpendicular
        self.matrix_phason = matrix_phason
        self.pmatrix = np.array(matrix_phason).reshape(3,3)
    def pm(self):
        return self.pmatrix   
    def qe0(self):
        return self.reflist_1
    def qe0(self):
        return self.reflist_2
    def qe1(self):
        pm=self.pmatrix
        v0=np.empty([0,3])
        for i in range(len(self.reflist_1)):
            v1=np.array(self.reflist_1[i]).T
            v2=np.array(self.reflist_2[i])
            v3=v1+np.dot(pm,v2.T)
            v0 = np.append(v0,v3.tolist())
        v0 = np.array(v0)
        return np.array(v0.reshape(len(self.reflist_1),3))


class PhasonDistoArray(object):
    '''Modifies reflection list according to the phason strain matrix'''
    def __init__(self,reflist_parallel,reflist_perpendicular,matrix_phason):
        self.reflist_1 = reflist_parallel
        self.reflist_2 = reflist_perpendicular
        self.matrix_phason = matrix_phason
        m = self.matrix_phason
        self.pmatrix = np.array([[m[0],m[1],m[2]],[m[3],m[4],m[5]],[m[6],m[7],m[8]]])
    def pm(self):
        return self.pmatrix
    def qe1(self):
        pm=self.pmatrix
        v1=self.reflist_1
        v2=self.reflist_2
        v3=v1+(np.dot(pm,v2.T)).T
        return v3
class Projection6dArrayApproximant(object):
    def __init__(self,ref,tau):
        self.ref=ref
        self.tau=tau
    
    def reflection_6d(self):
        #ref=self.ref
        # This matrix transform Elser's 6D indices to Cahn's 6D indices.
        self.mmm=np.matrix([
            [ 1., 0., 0., 0., 0., 0.],
            [ 0., 1., 0., 0., 0., 0.],
            [ 0., 0., 0., 0., 0., 1.],
            [ 0., 0., 0., 0., 1., 0.],
            [ 0., 0., 1., 0., 0., 0.],
            [ 0., 0., 0.,-1., 0., 0.],
            ])
        # 6 x 6 matrix for the projection onto the reciprocal 
        # 3D parallel and the 3D perpendicular spaces.
        #self.const=1.0/np.sqrt(2.0*(2.0+self.tau))   # (r.l.u)
        self.rmat=np.matrix([
            [   1.,  self.tau,   0.,  -1.,  self.tau,   0.],
            [  self.tau,   0.,   1.,  self.tau,   0.,  -1.],
            [   0.,   1.,  self.tau,   0.,  -1.,  self.tau],
            [ -self.tau,   1.,   0.,  self.tau,   1.,   0.],
            [   1.,   0., -self.tau,   1.,   0.,  self.tau],
            [   0.,  -self.tau,   1.,   0.,  self.tau,   1.],
            ])
            
        self.const = 1/np.linalg.norm(self.rmat[0,:])
        self.m6d = self.const*self.rmat
        refs = (self.m6d*(self.mmm*self.ref.T)).T
        ref_par = np.array(refs[:,:3])
        ref_perp = np.array(refs[:,3:])
        return ref_par, ref_perp

class Projection6d(object):
    
    def __init__(self,ref):
        self.ref=ref
    
    def reflection_6d(self):
        #ref=self.ref
        # This matrix transform Else's 6D indices to Cahn's 6D indices.
        mmm=np.matrix([
            [ 1., 0., 0., 0., 0., 0.],
            [ 0., 1., 0., 0., 0., 0.],
            [ 0., 0., 0., 0., 0., 1.],
            [ 0., 0., 0., 0., 1., 0.],
            [ 0., 0., 1., 0., 0., 0.],
            [ 0., 0., 0.,-1., 0., 0.],
            ])
        # 6 x 6 matrix for the projection onto the reciprocal 
        # 3D parallel and the 3D perpendicular spaces.
        const=1.0/np.sqrt(2.0*(2.0+TAU))   # (r.l.u)
        m6d=const*np.matrix([
            [   1.,  TAU,   0.,  -1.,  TAU,   0.],
            [  TAU,   0.,   1.,  TAU,   0.,  -1.],
            [   0.,   1.,  TAU,   0.,  -1.,  TAU],
            [ -TAU,   1.,   0.,  TAU,   1.,   0.],
            [   1.,   0., -TAU,   1.,   0.,  TAU],
            [   0.,  -TAU,   1.,   0.,  TAU,   1.],
            ])
#         m6d=const*np.matrix([
#             [   1.,  TAU,   0.,  -1.,  TAU,   0.],
#             [  TAU,   0.,   1.,  TAU,   0.,  -1.],
#             [   0.,   1.,  TAU,   0.,  -1.,  TAU],
#             [ -TAU,   1.,   0.,  TAU,   1.,   0.],
#             [   1.,   0., -TAU,   1.,   0.,  TAU],
#             [   0.,  -TAU,   1.,   0.,  1,   TAU],
#             ]) # from Cahn's paper 
        ref1=np.empty((0,3))
        ref2=np.empty((0,3))
        for k in range(len(self.ref)):
            n=self.ref[k]
            v0=np.array([n[0],n[1],n[2],n[3],n[4],n[5]])
            tmp=np.dot(mmm,v0.T)
            tmp=np.dot(m6d,tmp.T)
            ref_par=tmp[0:3].T
            ref_perp=tmp[3:6].T     
            ref1=np.append(ref1,ref_par)
            ref2=np.append(ref2,ref_perp)
        
        ref1=ref1.reshape(len(self.ref),3) # Parallel component (r.l.u)
        ref2=ref2.reshape(len(self.ref),3) # Perpendicular component (r.l.u)
        
        return ref1,ref2
            
##################################
       
       # TY modified as following;
       # calcms  -> calcms_ico
       # dmscalc -> dmscalc_ico
       # dmsfit -> dmsfit_ico
       
class calcms_ico(object):
    def __init__(self,lattice,hkl,hklint,hkl2,energy,azir,hkl4,mtrx2,F = [],F2 = []):   
        self.F = np.matrix(F)
        self.F2 = np.matrix(F2)
        self.lattice = lattice
        self.hkl = np.matrix(hkl)
        self.hkl2 = np.matrix(hkl2)
        self.hkl3 = hklint-self.hkl2
        self.energy = energy
        self.azir = np.matrix(azir)
        bm = bmatrix(self.lattice).bm()
        self.hkl4 = np.matrix(hkl4)
        self.mtrx2 = mtrx2
        #print self.hkl4
        #print self.mtrx2
        
##############  Modified by TY  ##############  
# hkl reflections after a distortion by a phason distortion
        self.hkl002 = PhasonDisto(self.hkl2,self.hkl4,self.mtrx2).qe1()
        self.hkl003 = hklint-self.hkl002
##############################################

#####   Convert primary hkl and reduced hkl2 list to orthogonal coordinate system    
        hklnotlist=(bm*self.hkl.transpose()).transpose()
        self.hklrlv=hklnotlist
        azir2=(bm*self.azir.transpose()).transpose()
#         zref=(bm*np.matrix([0,0,1]).transpose()).transpose()
        zref=np.matrix([[0,0,1]])
#   Determin transformation to align primary reflection to the z direction
        alignangle=interplanarangle(self.lattice,[0,0,1],self.hkl).ang()
        #realvecthkl=(bm*self.hkl2.transpose()).transpose()
        #realvecthkl3=(bm*self.hkl3.transpose()).transpose()
##############  Modified by TY  ##############
        realvecthkl=(bm*self.hkl002.transpose()).transpose()
        realvecthkl3=(bm*self.hkl003.transpose()).transpose()
##############################################

        rotvect=np.cross(zref,hklnotlist)
        if np.abs(rotvect[0][0])+np.abs(rotvect[0][1])+np.abs(rotvect[0][2]) >= 0.0001:
            realvecthkl=realvecthkl*rotxyz(rotvect,alignangle[0]).rmat() # multiplication order for rotation towards zref
#             self.tvprime = hklnotlist*rotxyz(rotvect,alignangle[0]).rmat()
            self.rmatrix = rotxyz(rotvect,alignangle[0]).rmat()
            self.tvprime = hklnotlist*self.rmatrix
        else:
            self.tvprime = hklnotlist
#   Build Ewald Sphere
        brag1 = np.empty(self.hkl2.shape[0])*0+1.0*bragg(self.lattice,self.hkl,self.energy).th()
        self.brag1 = brag1
        keV2A = 12.398
        ko=(self.energy/keV2A)
        self.ko = ko
        
#   height dependent radius of ewald slice in the hk plane
        rewl=ko*np.cos((np.arcsin(((ko*np.sin(-brag1*np.pi/180.0))+(realvecthkl[:,2]))/ko)*180.0/np.pi)*np.pi/180.0)
        rhk=np.sqrt(np.square(realvecthkl[:,0])+np.square(realvecthkl[:,1]))
        
#   Origin of intersecting circle
        #orighk = np.empty(self.hkl2.shape[0])*0+(ko*np.cos(brag1[0]*np.pi/180.))
##############  Modified by TY  ##############
        orighk = np.empty(self.hkl002.shape[0])*0+(ko*np.cos(brag1[0]*np.pi/180.))
##############################################

        ####################### MS Calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if np.abs(rotvect[0][0])+np.abs(rotvect[0][1])+np.abs(rotvect[0][2]) > 0.001:
            azir2=azir2*rotxyz(rotvect,alignangle[0]).rmat()

        azirangle=np.arctan2(azir2[0,0],azir2[0,1])*180.0/np.pi
        rhkangle=np.arctan2((realvecthkl[:,0]),(realvecthkl[:,1]))*180.0/np.pi
        yhkintercept=np.divide(np.square(orighk)-np.square(rhk)+np.square(rewl),(2.0*orighk))-orighk
        xintercept=np.sqrt(np.square(rewl)-np.square(np.divide((np.square(orighk)-np.square(rhk)+np.square(rewl)),2.0*orighk)))
#        realindex1 = np.where(yhkintercept.imag!=0)
#        realindex2 = np.where(xintercept.imag!=0)
#        realindex=[realindex1,realindex2]
        interceptangle1=np.arctan2(xintercept,yhkintercept)*180.0/np.pi
        interceptangle2=np.arctan2(-xintercept,yhkintercept)*180.0/np.pi #with respect to the real space origin
#        self.ewpsi1=np.arctan2(xintercept,yhkintercept-orighk)*180.0/np.pi
        self.ewpsi1=interceptangle1+rhkangle
        self.ewpsi2=interceptangle2+rhkangle
        psirotate=(interceptangle1+azirangle-rhkangle)
        psirotate2=(interceptangle2+azirangle-rhkangle)
        self.interceptangle1 = interceptangle1-rhkangle
        self.interceptangle2 = interceptangle2-rhkangle        
        self.rhkangle=rhkangle
        
        ########## return hkl back to original coordinate system ##############
        psi1 = (np.mod(psirotate+180.0,360.0)-180.0)
        psi1 = psi1[:,0]
        psi2 = (np.mod(psirotate2+180.0,360.0)-180.0)
        psi2 = psi2[:,0]
        brag1=np.matrix(brag1).transpose()
        braga=np.array(brag1)[0]
        self.kov1 =np.array((rotxyz([1,0,0],-np.array(braga)[0]).rmat()*np.matrix([[0,self.ko,0]]).T).T)
        self.psi1 = psi1
        self.psi2 = psi2
        self.bragg1 = brag1
        energyl=np.matrix(np.ones(psi1.shape[0])*energy).T        
  
##############################################
        if len(F) == 0:
            #self.fullarray = np.array(np.concatenate((hkl2,psi1,psi2,brag1,energyl),1))
##############  Modified by TY  ##############
            self.fullarray = np.array(np.concatenate((self.hkl002,psi1,psi2,brag1,energyl),1))
##############################################
        else:
            #self.fullarray = np.array(np.concatenate((hkl2,psi1,psi2,brag1,(self.F).T,energyl),1))
##############  Modified by TY  ##############
            self.fullarray = np.array(np.concatenate((self.hkl002,psi1,psi2,brag1,(self.F).T,energyl),1))
##############################################
        self.realvecthkl = realvecthkl
        self.realvecthkl3 = realvecthkl3
        self.ko=ko
    def tv(self):
        return self.realvecthkl
    def tvt(self):
        return self.realvecthkl3
    def rhkangle(self):
        return self.rhkangle
    def prlv(self):
        return self.hklrlv
    def kov(self):
        return self.kov1
    def ko(self):
        return self.ko
    def psi(self):
        return np.concatenate((self.psi1[:,0],self.psi2[:,0]),1)
        #return self.psi1, self.psi2
    def ewpsi(self):
        return self.ewpsi1, self.ewpsi2
    def bragg(self):
        return np.array(self.bragg1)
    def full(self):
        ''' returns hkl2,psi1,psi2,brag1,energ '''
        # MEMO by TY
        # hkl2 is replace by hkl002 in self.fullarray defined above.
        return self.fullarray
    def trv(self):
        ''' returns transformed and rotated vectors. '''
        trvarray=np.array([rotxyz([0,0,1],np.array(self.ewpsi1[i1,:])[0][0]).rmat()*self.realvecthkl[i1,:].T for i1 in range(self.ewpsi1.shape[0])])
        trvarray2=np.array([rotxyz([0,0,1],np.array(self.ewpsi2[i1,:])[0][0]).rmat()*self.realvecthkl[i1,:].T for i1 in range(self.ewpsi2.shape[0])])
        return np.matrix(np.squeeze(trvarray)), np.matrix(np.squeeze(trvarray2))
    def trvt(self):
        ''' returns transformed and rotated tertiary vectors. '''
        trvarrayt=np.array([rotxyz([0,0,1],np.array(self.ewpsi1[i1,:])[0][0]).rmat()*self.realvecthkl3[i1,:].T for i1 in range(self.ewpsi1.shape[0])])
        trvarray2t=np.array([rotxyz([0,0,1],np.array(self.ewpsi2[i1,:])[0][0]).rmat()*self.realvecthkl3[i1,:].T for i1 in range(self.ewpsi2.shape[0])])
        return np.matrix(np.squeeze(trvarrayt)), np.matrix(np.squeeze(trvarray2t))
    def bvects(self):
        ''' returns secondary beam vectors '''
        return self.trv()[0]+self.kov1,self.trv()[1]+self.kov1
    def bvects2(self):
        ''' returns tertiary beam vectors '''
        return self.trvt()[0]+self.bvects()[0],self.trvt()[1]+self.bvects()[1]
    def angs(self):
        ''' Angles between ko and beam vectors '''
        norms1=np.apply_along_axis(np.linalg.norm, 1, self.bvects()[0])
        angs1=np.arccos((np.matrix(-self.kov())*np.matrix(self.bvects()[0]).T)/(LA.norm(self.kov())*norms1))*180.0/np.pi
        norms2=np.apply_along_axis(np.linalg.norm, 1, self.bvects()[1])
        angs2=np.arccos((np.matrix(-self.kov())*np.matrix(self.bvects()[1]).T)/(LA.norm(self.kov())*norms2))*180.0/np.pi
        return angs1, angs2
    def psiplaneang(self):
        ''' Angle required to rotate k1 about ko onto the secondary scattering plane '''
        v1=np.matrix([[1,0,0]]) # determines slice direction of interplanerangle function
        norms1=np.apply_along_axis(np.linalg.norm, 1, self.bvects()[0])
        nbv=(self.bvects()[0].T/norms1).T # normalized beam vectors
        v2=np.cross(-self.kov(),nbv)
        psiangs=interplanarangle([1,1,1,90,90,90],v1,v2).ang()
        return psiangs
    def psiplaneang2(self):
        ''' Angle required to rotate k2 about k1 onto the tertiary scattering plane '''       
        norms1=np.apply_along_axis(np.linalg.norm, 1, self.bvects()[0])
        norms2=np.apply_along_axis(np.linalg.norm, 1, self.bvects2()[0])
        nbv1=np.cross(-self.kov(),(self.bvects()[0].T/norms1).T)
        nbv2=np.cross((self.bvects()[0].T/norms1).T,(self.bvects2()[0].T/norms2).T)
        psiangs2=interplanarangle([1,1,1,90,90,90],nbv1,nbv2).ang()
        return psiangs2
    def pol(self,polv):
        ''' returns hkl2, sig, pi, pfactor   '''
        refs=self.fullarray[:,[0,1,2]]
        braggs=bragg(self.lattice,refs,self.energy).th()
        psiang=self.psiplaneang()
        pmtmpv=np.array(np.squeeze([(np.matrix([[1,0],[0,np.cos(2*braggs[i1]*np.pi/180.0)]])* \
                        np.matrix([[np.cos(psiang[i1]*np.pi/180.0),np.sin(psiang[i1]*np.pi/180.0)], \
                        [-np.sin(psiang[i1]*np.pi/180.0),np.cos(psiang[i1]*np.pi/180.0)]])*np.matrix(polv).T).T \
                        for i1 in range(braggs.shape[0])]))
        sums=np.matrix(np.sum((pmtmpv)**2,1)).T
        return np.concatenate((pmtmpv,sums),1)
#     def pol2(self,polv):
#         ''' returns hkl3, sig, pi, pfactor   '''
#         refs=self.fullarray[:,[0,1,2]]
#         polv2=self.pol(polv)[:,[-3,-2]]
#         brags2=bragg(self.lattice,self.hkl-refs,self.energy).th()
#         psiang2=self.psiplaneang2()
#         pmtmpv2=np.array(np.squeeze([(np.matrix([[1,0],[0,np.cos(2*brags2[i1]*np.pi/180.0)]])* \
#                         np.matrix([[np.cos(psiang2[i1]*np.pi/180.0),np.sin(psiang2[i1]*np.pi/180.0)],\
#                         [-np.sin(psiang2[i1]*np.pi/180.0),np.cos(psiang2[i1]*np.pi/180.0)]])* \
#                         np.matrix(polv2[i1,[0,1]]).T).T for i1 in range(brags2.shape[0])]))
#         sums2=np.matrix(np.sum((pmtmpv2)**2,1)).Tko=(self.energy/keV2A)
#         return np.concatenate((pmtmpv2,sums2),1)
    
    def pol2(self,polv):
        ''' returns hkl3, sig, pi, pfactor   '''
        refs=self.fullarray[:,[0,1,2]]
        brags=bragg(self.lattice,refs,self.energy).th()
        brags2=bragg(self.lattice,self.hkl-refs,self.energy).th()
        psiang=self.psiplaneang()
        psiang2=self.psiplaneang2()
        pmtmpv2=np.array(np.squeeze([(np.matrix([[1,0],[0,np.cos(2*brags2[i1]*np.pi/180.0)]])* \
                        np.matrix([[np.cos(psiang2[i1]*np.pi/180.0),np.sin(psiang2[i1]*np.pi/180.0)], \
                        [-np.sin(psiang2[i1]*np.pi/180.0),np.cos(psiang2[i1]*np.pi/180.0)]])* \
                        np.matrix([[1,0],[0,np.cos(2*brags[i1]*np.pi/180.0)]])* \
                        np.matrix([[np.cos(psiang[i1]*np.pi/180.0),np.sin(psiang[i1]*np.pi/180.0)], \
                        [-np.sin(psiang[i1]*np.pi/180.0),np.cos(psiang[i1]*np.pi/180.0)]])*np.matrix(polv).T).T \
                        for i1 in range(brags2.shape[0])]))
        sums2=np.matrix(np.sum((pmtmpv2)**2,1)).T
        return np.concatenate((pmtmpv2,sums2),1)

#     def polfull(self,polv):
#         ''' returns hkl2,psi1,psi2,brag1,energy, sig, pi, pfactor, pfactor*F   '''
# #         return np.concatenate((self.full(),self.pol(polv)[:,[-3,-2,-1]],(self.pxf(polv)).T),1)
#         return np.concatenate((self.full(),self.pol2(polv),(self.pxf(polv)).T),1)
#     def pol2full(self,polv):
#         ''' returns hkl2,psi1,psi2,brag1,energy, sig, pi, pfactor, pfactor*F  using '''
#         return np.concatenate((self.full(),self.pol(polv)[:,[-3,-2,-1]],(self.pxf(polv)).T),1)
#     def polfull2(self,polv):
#         ''' returns hkl2,psi1,psi2,brag1,energy, sig, pi, pfactor, pfactor*F   '''
# #         return np.concatenate((self.full(),self.pol(polv)[:,[-3,-2,-1]]),1)
    def pv1xsf1(self,polv):
        ampT=np.array(self.F.T)*np.array(self.pol(polv)[:,-1])
        return np.concatenate((self.full(),ampT),1)
    def geometry(self):
        return self.full()
    def polfull(self,polv):
        ampT=np.array(self.F.T)*np.array(self.F2.T)*np.array(self.pol2(polv)[:,-1])
        return np.concatenate((self.full(),ampT),1)
    def polfull2(self,polv):
        ''' returns hkl2,psi1,psi2,brag1,energy, sig, pi, pfactor, pfactor*F   '''
        return np.concatenate((self.full(),self.pol(polv)),1)
    def sfonly(self):
        ampT = np.array(self.F.T)*np.array(self.F2.T)
        return np.concatenate((self.full(),ampT),1)
    def sf1only(self):
        ampT = np.array(self.F.T)
        return np.concatenate((self.full(),ampT),1)
    def pol1only(self,polv):
        ampT=self.pol(polv)[:,-1]
        return np.concatenate((self.full(),ampT),1)
    def pol2only(self,polv):
        ampT=self.pol2(polv)[:,-1]
        return np.concatenate((self.full(),ampT),1)
    def SF(self):
        return self.F
    def SF2(self):
        return self.F2
    def pxf(self,polv):
        return np.array(self.SF())*np.array(self.pol(polv)[:,-1]).T
    def ov(self):
        ''' returns original vector list. '''
        #return self.hkl2
        return self.hkl002
    def orig(self):
        ''' Returns reciprocal space origin. '''
        return np.matrix([0,self.ko*np.cos(self.brag1[0]*np.pi/180.0),-self.ko*np.sin(self.brag1[0]*np.pi/180.0)])
    def kv(self):
        return self.ko
    def tvp(self):
        return self.tvprime
    def ppsi(self):
        return np.concatenate((self.interceptangle1,self.interceptangle2),1)
#     def SF2(self):
#         reflist2=vfind
    def th(self):
        return np.arcsin((self.kov()+self.trv())[0][:,2]/self.ko)*180/np.pi
    def getref(self):
        return self.hkl002
        
                
class dmscalc_ico(object):
    def __init__(self,*args):
        self.reflist=args[0]    # parallel component of Bragg reflection
        self.hkllist=args[1]
        self.hklint=args[2]
        self.intensity=args[3]
        self.psirange=args[4]
        self.threshold=args[5]
        self.hkl=args[6]
        self.detvects=args[7]
        self.imdata=args[8]
        self.simsigma=args[9]
        self.azir=args[10]
        self.psi=args[11]
        self.px=args[12]
        self.py=args[13]
        self.scatv=args[14]
        if len(args) > 15:
            self.detdistancepx=args[15]
            self.detxrot=args[16]
            self.detyrot=args[17]
            self.detzrot=args[18]
            self.energy=args[19]
######## TY added #########
            if len(args) > 20:
                self.reflist2=args[20]   # perpemdicular component of Bragg reflection
                self.mtrx2=args[21]      # 3x3 phason matrix
###########################
        self.imsim=None
    def sethkl(self,hkl):
        self.hkl = hkl
    def sethkllist(self,hkllist):
        self.hkllist=hkllist
    def imcalc(self,*inputs):
        inputs=inputs[0]
        a,b,c,alpha,beta,gamma=inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5]
        psicorrection,thetacorrection, chicorrection=inputs[6],inputs[7],inputs[8]
        if len(inputs) > 9:
            detdistancepx,detxrot,detyrot,detzrot=inputs[9],inputs[10],inputs[11],inputs[12]
            energy=inputs[13]
            mtrx2=[inputs[14],inputs[15],inputs[16],inputs[17],inputs[18],inputs[19],inputs[20],inputs[21],inputs[22]]
        else:
            detdistancepx,detxrot,detyrot,detzrot=self.detdistancepx,self.detxrot,self.detyrot,self.detzrot
            energy=self.energy
            mtrx2=self.mtrx2

        lattice = [a,b,c,alpha,beta,gamma]
        keV2A_ko   = 12.398
        keV2A_bragg= 12.3984187
        ko  = energy / keV2A_ko
        wl  = keV2A_bragg / energy

        # ── Detector setup (unchanged) ───────────────────────────────────────
        thb=bragg(lattice,self.hkl,energy).th()[0]
        self.bragg = thb
        detvs=np.array(self.detvects*rotxyz([0,0,1],-detzrot).rmat()*rotxyz([0,1,0],-detyrot).rmat()*rotxyz([1,0,0],-detxrot-thb).rmat())
        irmat=rotxyz([1,0,0],detxrot+thb).rmat()*rotxyz([0,1,0],detyrot).rmat()*rotxyz([0,0,1],detzrot).rmat()
        chiaxis = (rotxyz(np.cross((rotxyz(self.hkl,self.psi).rmat()*np.array([self.azir]).T).T, np.array([self.hkl])),90).rmat()*np.array([self.hkl]).T).T
        hkllist = np.array((rotxyz(chiaxis, -chicorrection).rmat()*np.array(self.hkllist).T).T)  # (N_steps, 3)
        N_steps = hkllist.shape[0]

        # ── Constants independent of scan hkl ───────────────────────────────
        bm = np.array(bmatrix(lattice).bm())                    # (3,3)
        hkl002 = PhasonDistoArray(
            np.array(self.reflist), np.array(self.reflist2), mtrx2
        ).qe1()                                                  # (N_refs, 3)
        hkl002_cart = hkl002 @ bm.T                             # (N_refs, 3) — reflist in Cartesian
        azir_cart0  = np.array(self.azir).reshape(3) @ bm.T     # (3,)        — azir in Cartesian
        N_refs = hkl002.shape[0]

        # ── Per-step: scan hkl → Cartesian, Bragg angle ─────────────────────
        hklnotlist = hkllist @ bm.T                              # (N_steps, 3)
        hklnotlist_norms = np.linalg.norm(hklnotlist, axis=1)   # (N_steps,)
        safe_norms = np.maximum(hklnotlist_norms, 1e-12)
        brag1_all = 180/np.pi * np.arcsin(wl * safe_norms / 2.0)  # (N_steps,)  — Bragg: sin(θ)=λ/(2d)=λ|G|/2

        # ── Per-step: rotation axis & angle to align scan hkl → z-axis ─────
        rotvect_all = np.cross([0.0,0.0,1.0], hklnotlist)       # (N_steps, 3)
        rotvect_l1  = np.sum(np.abs(rotvect_all), axis=1)       # (N_steps,)

        # interplanarangle([0,0,1], hkllist[i]) via dot in Cartesian
        zref_cart      = np.array([0.0,0.0,1.0]) @ bm.T         # (3,) = bm[2,:]
        zref_cart_norm = np.linalg.norm(zref_cart)
        cos_align = np.clip(
            (hklnotlist @ zref_cart) / (safe_norms * zref_cart_norm), -1.0, 1.0
        )                                                         # (N_steps,)
        alignangle_all = np.arccos(cos_align) * 180/np.pi        # (N_steps,) degrees

        # ── Batch Rodrigues rotation matrices (N_steps, 3, 3) ───────────────
        u_all = rotvect_all / np.maximum(
            np.linalg.norm(rotvect_all, axis=1, keepdims=True), 1e-12
        )                                                         # (N_steps, 3) normalised axes
        t_rad = alignangle_all * np.pi / 180
        c_t = np.cos(t_rad);  s_t = np.sin(t_rad)               # (N_steps,)
        ux, uy, uz = u_all[:,0], u_all[:,1], u_all[:,2]

        R = np.zeros((N_steps, 3, 3))
        R[:,0,0] = c_t + ux*ux*(1-c_t);  R[:,0,1] = ux*uy*(1-c_t) - uz*s_t;  R[:,0,2] = ux*uz*(1-c_t) + uy*s_t
        R[:,1,0] = uy*ux*(1-c_t) + uz*s_t;  R[:,1,1] = c_t + uy*uy*(1-c_t);  R[:,1,2] = uy*uz*(1-c_t) - ux*s_t
        R[:,2,0] = uz*ux*(1-c_t) - uy*s_t;  R[:,2,1] = uz*uy*(1-c_t) + ux*s_t;  R[:,2,2] = c_t + uz*uz*(1-c_t)
        R[rotvect_l1 < 0.0001] = np.eye(3)                      # identity for near-[0,0,1] steps

        # ── Apply rotations to reflist and azir ─────────────────────────────
        # realvecthkl[i,j,:] = hkl002_cart[j,:] @ R[i]  →  (N_steps, N_refs, 3)
        realvecthkl = np.einsum('jr,irs->ijs', hkl002_cart, R)

        azir_rot = np.einsum('r,irs->is', azir_cart0, R)        # (N_steps, 3)
        azir_rot[rotvect_l1 < 0.001] = azir_cart0
        azirangle_all = np.arctan2(azir_rot[:,0], azir_rot[:,1]) * 180/np.pi  # (N_steps,)

        # ── Ewald sphere intersection (vectorised over steps × refs) ────────
        b1       = brag1_all[:,np.newaxis]                       # (N_steps, 1)
        orighk   = ko * np.cos(b1 * np.pi/180)                  # (N_steps, 1)
        raw_sin  = (ko*np.sin(-b1*np.pi/180) + realvecthkl[:,:,2]) / ko
        valid    = np.abs(raw_sin) <= 1.0                        # physical Ewald condition
        sin_arg  = np.clip(raw_sin, -1.0, 1.0)
        rewl     = ko * np.cos(np.arcsin(sin_arg))               # (N_steps, N_refs)
        rhk      = np.sqrt(realvecthkl[:,:,0]**2 + realvecthkl[:,:,1]**2)
        rhkangle = np.arctan2(realvecthkl[:,:,0], realvecthkl[:,:,1]) * 180/np.pi

        numer      = orighk**2 - rhk**2 + rewl**2               # (N_steps, N_refs)
        half_n     = numer / (2*orighk)
        disc       = rewl**2 - half_n**2
        valid     &= disc >= 0                                    # real intersection exists
        xint       = np.sqrt(np.maximum(disc, 0))

        ia1 = np.arctan2( xint, half_n - orighk) * 180/np.pi
        ia2 = np.arctan2(-xint, half_n - orighk) * 180/np.pi

        az = azirangle_all[:,np.newaxis]                         # (N_steps, 1)
        psi1 = np.mod(ia1 + az - rhkangle + 180, 360) - 180     # (N_steps, N_refs)
        psi2 = np.mod(ia2 + az - rhkangle + 180, 360) - 180

        # ── Build mslist: (N_steps*N_refs + 1, 7) ──────────────────────────
        # columns: hkl002[0:3], psi1, psi2, brag1, energy  (matches calcms_ico fullarray)
        mslist = np.empty((N_steps * N_refs + 1, 7))
        mslist[0] = np.nan
        flat = mslist[1:].reshape(N_steps, N_refs, 7)           # view into mslist
        flat[:,:,0:3] = hkl002[np.newaxis,:,:]                  # same reflist for every step
        flat[:,:,3]   = psi1
        flat[:,:,4]   = psi2
        flat[:,:,5]   = brag1_all[:,np.newaxis]
        flat[:,:,6]   = energy
        flat[~valid, 3:5] = np.nan                               # kill non-physical solutions

        # ── Pixel projection (unchanged) ─────────────────────────────────────
        vecs1=psith2v(self.psi-mslist[:,3]-psicorrection,mslist[:,5]+thetacorrection)
        vecs2=psith2v(self.psi-mslist[:,4]-psicorrection,mslist[:,5]+thetacorrection)
        vecs=np.concatenate((vecs1,vecs2),0)
        centralv=-psith2v(0,thb)*detdistancepx
        prepxvec=dms2px(detvs[0,:],detvs[1,:],centralv,vecs)
        valid_px = ~np.isnan(prepxvec).any(axis=1)
        pxvec=np.array(np.round(prepxvec[valid_px]*irmat).astype(int))
        imsim=np.zeros(np.shape(self.imdata))
        self.vecs = vecs

        # Track which reflection index each projected pixel belongs to.
        # mslist index i maps to ref (i-1) % N_refs for i >= 1 (sentinel at 0).
        total = N_steps * N_refs + 1
        ref_idx_half = np.full(total, -1, dtype=int)
        ref_idx_half[1:] = np.tile(np.arange(N_refs), N_steps)
        ref_idx_all = np.concatenate([ref_idx_half, ref_idx_half])
        ref_idx_valid = ref_idx_all[valid_px]

        pxv2d=np.array(pxvec[:,[0,2]])
        _r, _c = det_to_pixels(pxv2d[:,0], pxv2d[:,1], self.px, self.py, self.scatv)
        pxv2d[:,0], pxv2d[:,1] = _r, _c
        try:
            m0 = pxv2d[:,0] > -1
            pxv2d = pxv2d[m0]; ref_idx_valid = ref_idx_valid[m0]
            m1 = pxv2d[:,0] < imsim.shape[0]
            pxv2d = pxv2d[m1]; ref_idx_valid = ref_idx_valid[m1]
            m2 = pxv2d[:,1] > -1
            pxv2d = pxv2d[m2]; ref_idx_valid = ref_idx_valid[m2]
            m3 = pxv2d[:,1] < imsim.shape[1]
            pxv2d = pxv2d[m3]; ref_idx_valid = ref_idx_valid[m3]
            self.dmsindex=tuple([pxv2d[:,0],pxv2d[:,1]])
            self.pxv2d_all = pxv2d
            self.pxv2d_refidx = ref_idx_valid
            if self.simsigma != 0:
                imsim[self.dmsindex]=self.imdata.max()
                self.imsim=ndimage.gaussian_filter(imsim, sigma=(self.simsigma), order=0)
            else:
                imsim[self.dmsindex]=1
                self.imsim=imsim
        except:
            self.dmsindex=[]
            self.pxv2d_all = np.empty((0, 2), dtype=int)
            self.pxv2d_refidx = np.empty(0, dtype=int)

    def full(self,inputs):
        try:
            self.imcalc(inputs)# adding attribute
            numabovethresh=len(np.where(self.imdata+self.imsim > self.threshold)[0])
            return -np.sum(self.imsim*self.imdata/numabovethresh), self.imsim, self.dmsindex, self.imdata
        except:
            print('Index empty')
            return 500,  self.imdata*10100, np.array([[],[]]), self.imdata*10100
    def roiindex(self,inputs):
        self.imcalc(inputs)# adding attribute
        dmsindex=np.array(self.dmsindex).T
        return dmsindex
    def getref(self):
        return self.ms.getref()
    def vecs(self,inputs):
        return self.vecs
        
class dmscalc_ico_hkl(object):
    def __init__(self,*args):
        self.reflist=args[0]    # parallel component of Bragg reflection
        self.hkllist=args[1]
        self.hklint=args[2]
        self.intensity=args[3]
        self.psirange=args[4]
        self.threshold=args[5]
        self.hkl=args[6]
        self.detvects=args[7]
        self.imdata=args[8]
        self.simsigma=args[9]
        self.azir=args[10]
        self.psi=args[11]
        self.px=args[12]
        self.py=args[13]
        self.scatv=args[14]
        if len(args) > 15:
            self.detdistancepx=args[15]
            self.detxrot=args[16]
            self.detyrot=args[17]
            self.detzrot=args[18]
            self.energy=args[19]
######## TY added #########
            if len(args) > 20:
                self.reflist2=args[20]   # perpemdicular component of Bragg reflection
                self.mtrx2=args[21]      # 3x3 phason matrix
###########################
        self.imsim=None
    def sethkl(self,hkl):
        self.hkl = hkl
    def sethkllist(self,hkllist):
        self.hkllist=hkllist
    def imcalc(self,*inputs):
        inputs=inputs[0]
        # Conventional crystals carry the full lattice in slots [0:6]; the
        # icosahedral path keeps the cubic a=b=c, 90,90,90 constraint.
        # The ROI builder always receives the full 24-element guess, so the
        # corrections sit at fixed slots: psicor(6), chi(7), theta(8).  The hkl
        # corrections are unused (redundant with the primary hkl).
        if getattr(self, 'crystal_system', None) in CONVENTIONAL_SYSTEMS:
            a,b,c,alpha,beta,gamma = expand_lattice(self.crystal_system, inputs[0:6])
        else:
            a,b,c,alpha,beta,gamma=inputs[0],inputs[0],inputs[0],90, 90, 90
        psicorrection   = inputs[6]
        chicorrection   = inputs[7]
        thetacorrection = inputs[8]
        h_correction = k_correction = l_correction = 0.0

        if len(inputs) > 10:
            detdistancepx,detxrot,detyrot,detzrot=inputs[10],inputs[11],inputs[12],inputs[13]
            energy=inputs[14]
            # 3x3 phason matrix, mtrx2
            mtrx2=[inputs[15],inputs[16],inputs[17],inputs[18],inputs[19],inputs[20],inputs[21],inputs[22],inputs[23]]

###########################
        else:
            detdistancepx,detxrot,detyrot,detzrot=self.detdistancepx,self.detxrot,self.detyrot,self.detzrot
            energy=self.energy
        lattice = [a,b,c,alpha,beta,gamma]
        thb=bragg(lattice,self.hkl,energy).th()[0]
        self.bragg = thb
        detvs=np.array(self.detvects*rotxyz([0,0,1],-detzrot).rmat()*rotxyz([0,1,0],-detyrot).rmat()*rotxyz([1,0,0],-detxrot-thb).rmat())
        irmat=rotxyz([1,0,0],detxrot+thb).rmat()*rotxyz([0,1,0],detyrot).rmat()*rotxyz([0,0,1],detzrot).rmat()

        # ── Vectorised Ewald sphere calculation ─────────────────────────────
        keV2A_ko    = 12.398
        keV2A_bragg = 12.3984187
        ko  = energy / keV2A_ko
        wl  = keV2A_bragg / energy

        bm          = np.array(bmatrix(lattice).bm())
        hkl002      = PhasonDistoArray(
            np.array(self.reflist), np.array(self.reflist2), mtrx2
        ).qe1()                                                          # (N_refs, 3)
        hkl002_cart = hkl002 @ bm.T
        azir_cart0  = np.array(self.azir).reshape(3) @ bm.T
        N_refs      = hkl002.shape[0]
        hkllist_arr = np.array(self.hkllist)
        if chicorrection != 0:
            chiaxis = (rotxyz(np.cross((rotxyz(self.hkl,self.psi).rmat()*np.array([self.azir]).T).T, np.array([self.hkl])),90).rmat()*np.array([self.hkl]).T).T
            hkllist_arr = np.array((rotxyz(chiaxis, -chicorrection).rmat()*hkllist_arr.T).T)
        N_steps     = hkllist_arr.shape[0]

        hklnotlist       = hkllist_arr @ bm.T                           # (N_steps, 3)
        hklnotlist_norms = np.linalg.norm(hklnotlist, axis=1)
        safe_norms       = np.maximum(hklnotlist_norms, 1e-12)
        brag1_all        = 180/np.pi * np.arcsin(wl * safe_norms / 2.0)

        rotvect_all = np.cross([0.0, 0.0, 1.0], hklnotlist)
        rotvect_l1  = np.sum(np.abs(rotvect_all), axis=1)

        zref_cart      = np.array([0.0, 0.0, 1.0]) @ bm.T
        zref_cart_norm = np.linalg.norm(zref_cart)
        cos_align = np.clip(
            (hklnotlist @ zref_cart) / (safe_norms * zref_cart_norm), -1.0, 1.0
        )
        alignangle_all = np.arccos(cos_align) * 180/np.pi

        u_all = rotvect_all / np.maximum(
            np.linalg.norm(rotvect_all, axis=1, keepdims=True), 1e-12
        )
        t_rad = alignangle_all * np.pi / 180
        c_t = np.cos(t_rad);  s_t = np.sin(t_rad)
        ux, uy, uz = u_all[:,0], u_all[:,1], u_all[:,2]

        R = np.zeros((N_steps, 3, 3))
        R[:,0,0] = c_t + ux*ux*(1-c_t);  R[:,0,1] = ux*uy*(1-c_t) - uz*s_t;  R[:,0,2] = ux*uz*(1-c_t) + uy*s_t
        R[:,1,0] = uy*ux*(1-c_t) + uz*s_t;  R[:,1,1] = c_t + uy*uy*(1-c_t);  R[:,1,2] = uy*uz*(1-c_t) - ux*s_t
        R[:,2,0] = uz*ux*(1-c_t) - uy*s_t;  R[:,2,1] = uz*uy*(1-c_t) + ux*s_t;  R[:,2,2] = c_t + uz*uz*(1-c_t)
        R[rotvect_l1 < 0.0001] = np.eye(3)

        realvecthkl   = np.einsum('jr,irs->ijs', hkl002_cart, R)       # (N_steps, N_refs, 3)
        azir_rot      = np.einsum('r,irs->is', azir_cart0, R)
        azir_rot[rotvect_l1 < 0.001] = azir_cart0
        azirangle_all = np.arctan2(azir_rot[:,0], azir_rot[:,1]) * 180/np.pi

        b1      = brag1_all[:,np.newaxis]
        orighk  = ko * np.cos(b1 * np.pi/180)
        # Physical-solution mask, exactly as dmsfit_ico_hkl.imcalc computes it.
        # Without it the clamps below turn every *non*-physical step into a
        # solution rather than dropping it: |sin| > 1 gets clipped to the
        # horizon, and a negative discriminant (no Ewald intersection) gets
        # clamped to zero, which makes ia1 = ia2 = a degenerate angle.  Those
        # invented points project to a perfectly smooth curve somewhere else on
        # the detector, so the ROI builder — the one caller of this engine —
        # could lay a ROI along a line that does not exist and that the overlay
        # (which uses the fit engine) never draws.  On the TestExample session
        # reflection [-1 1 1] drew a vertical arc and got a horizontal ROI 835 px
        # away, built entirely from these.
        raw_sin = (ko*np.sin(-b1*np.pi/180) + realvecthkl[:,:,2]) / ko
        valid   = np.abs(raw_sin) <= 1.0                                  # physical Ewald condition
        sin_arg = np.clip(raw_sin, -1.0, 1.0)
        rewl     = ko * np.cos(np.arcsin(sin_arg))
        rhk      = np.sqrt(realvecthkl[:,:,0]**2 + realvecthkl[:,:,1]**2)
        rhkangle = np.arctan2(realvecthkl[:,:,0], realvecthkl[:,:,1]) * 180/np.pi

        numer  = orighk**2 - rhk**2 + rewl**2
        half_n = numer / (2*orighk)
        disc   = rewl**2 - half_n**2
        valid &= disc >= 0                                                 # real intersection exists
        xint   = np.sqrt(np.maximum(disc, 0))

        ia1 = np.arctan2( xint, half_n - orighk) * 180/np.pi
        ia2 = np.arctan2(-xint, half_n - orighk) * 180/np.pi

        az   = azirangle_all[:,np.newaxis]
        psi1 = np.mod(ia1 + az - rhkangle + 180, 360) - 180
        psi2 = np.mod(ia2 + az - rhkangle + 180, 360) - 180

        mslist = np.empty((N_steps * N_refs + 1, 7))
        mslist[0] = np.nan
        flat = mslist[1:].reshape(N_steps, N_refs, 7)
        flat[:,:,0:3] = hkl002[np.newaxis,:,:]
        flat[:,:,3]   = psi1
        flat[:,:,4]   = psi2
        flat[:,:,5]   = brag1_all[:,np.newaxis]
        flat[:,:,6]   = energy
        flat[~valid, 3:5] = np.nan                                        # kill non-physical solutions
        vecs1=psith2v(self.psi-mslist[:,3]-psicorrection,mslist[:,5]+thetacorrection)
        vecs2=psith2v(self.psi-mslist[:,4]-psicorrection,mslist[:,5]+thetacorrection)
        vecs=np.concatenate((vecs1,vecs2),0)
        centralv=-psith2v(0,thb)*detdistancepx
        prepxvec=dms2px(detvs[0,:],detvs[1,:],centralv,vecs)
        # Drop the NaN rows before the integer cast, as the fit engine does.
        # Casting NaN to int is undefined (it lands on INT_MIN here, and warns),
        # and leaving it to the range filters below to remove is one platform
        # away from being wrong.
        prepxvec = prepxvec[~np.isnan(np.asarray(prepxvec)).any(axis=1)]
        pxvec=np.array(np.round(prepxvec*irmat).astype(int)) #build reverse matrix for detector
        imsim=np.zeros(np.shape(self.imdata))
        self.vecs = vecs
        #########  Shift vectors to non negative coordinates   ######################
        pxv2d=np.array(pxvec[:,[0,2]])
        _r, _c = det_to_pixels(pxv2d[:,0], pxv2d[:,1], self.px, self.py, self.scatv)
        pxv2d[:,0], pxv2d[:,1] = _r, _c
        try:
            pxv2d=pxv2d[np.where(pxv2d[:,0]>-1)]
            pxv2d=pxv2d[np.where(pxv2d[:,0]< imsim.shape[0])]
            pxv2d=pxv2d[np.where(pxv2d[:,1]>-1)]
            pxv2d=pxv2d[np.where(pxv2d[:,1]< imsim.shape[1])]
            self.dmsindex=tuple([pxv2d[:,0],pxv2d[:,1]])
           
            if self.simsigma != 0:
                imsim[self.dmsindex]=self.imdata.max()
                self.imsim=ndimage.gaussian_filter(imsim, sigma=(self.simsigma), order=0)
#                 self.imsim=ndimage.convolve(imsim,makekernel('custom2',15,self.simsigma,0.5))
            else:
                imsim[self.dmsindex]=1
                self.imsim=imsim
        except:
            self.dmsindex=[]

    def full(self,inputs):
        try:
            self.imcalc(inputs)# adding attribute
            numabovethresh=len(np.where(self.imdata+self.imsim > self.threshold)[0])
            return -np.sum(self.imsim*self.imdata/numabovethresh), self.imsim, self.dmsindex, self.imdata
        except:
            print('Index empty')
            return 500,  self.imdata*10100, np.array([[],[]]), self.imdata*10100
    def roiindex(self,inputs):
        self.imcalc(inputs)# adding attribute
        dmsindex=np.array(self.dmsindex).T
        return dmsindex
    def getref(self):
        return self.ms.getref()
    def vecs(self,inputs):
        return self.vecs
# ── DMS curves as the circles they analytically are ───────────────────────────
# A DMS (Kossel) line comes from one secondary reflection: the doubly-diffracted
# radiation leaves the sample along a *cone* of directions, all satisfying that
# plane's diffraction condition (k_out.Ghat = |G|/2 for the exit wavevector).  A
# cone of unit vectors is a circle on the sphere, so each locus is *exactly* a
# circle in exit-direction space — the theta-scan of `imcalc` only samples it.
#
# That gives a second way to compute the same curves: run the scan coarsely, fit
# the circle each continuous run lies on (three points determine it; the fit is
# exact, not an approximation), and re-sample the arc at whatever resolution the
# detector deserves.  `numsteps` then only decides how precisely the *ends* of
# each arc are located, not how smooth the curve is, so a fraction of the points
# gives a better line.  Ported from the sibling ReciprocalSpaceVisualisation
# project (`dms_compute.py`), which does the same thing in reciprocal space.
#
# Measured against this engine the circle is exact to ~1e-15 (both lattice modes,
# any psi, with or without phason strain or a chi correction) provided each run
# is cut where the geometry changes — see `dms_split_runs`.  The one real
# exception is a non-zero theta correction, which offsets the exit polar angle
# *after* the azimuth was solved at the uncorrected angle and so shears the locus
# slightly off-plane: first order in thetacor, ~2e-4 rad at 1 deg.  Every helper
# below returns its worst deviation so a caller can report that rather than hide
# it.

DMS_CURVE_METHODS = ('sweep', 'circle')

# Fit deviation (in exit-direction units, i.e. radians of arc) past which a run
# is not accepted as a circle and the sampled points are drawn instead.  1e-3 is
# ~3 px at a typical 3000 px detector distance — far above the ~1e-15 a genuine
# arc achieves and above the theta-correction shear, but small enough that a run
# straddling a geometry change can never be drawn as a bogus circle.
DMS_CIRCLE_TOL = 1e-3


def dms_curve_method(name):
    '''Normalise a DMS curve-method name to one of `DMS_CURVE_METHODS`.

    'sweep'  — the sampled theta-scan points are the curve (the original).
    'circle' — each continuous run is reduced to the circle it lies on and
               re-sampled at detector resolution.'''
    key = str(name or 'sweep').strip().lower()
    if key in ('circle', 'circles', 'analytic', 'analytical'):
        return 'circle'
    if key in ('sweep', 'sampled', 'scan', 'theta-sweep'):
        return 'sweep'
    raise ValueError('unknown DMS curve method %r (expected one of %s)'
                     % (name, ', '.join(DMS_CURVE_METHODS)))


def dms_plane_basis(n):
    '''Deterministic in-plane axis for a circle of normal `n` (n x xhat, falling
    back to n x yhat when n is along xhat).  The arc angles are measured in this
    basis, so every producer and consumer of an arc must derive it the same way.
    Accepts a single normal or an (N,3) stack.'''
    n = np.atleast_2d(np.asarray(n, dtype=float))
    u = np.stack([np.zeros(len(n)), n[:, 2], -n[:, 1]], axis=1)
    small = (u * u).sum(1) < 1e-18
    if small.any():
        u[small] = np.stack([-n[small, 2], np.zeros(int(small.sum())), n[small, 0]],
                            axis=1)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def dms_split_runs(pts, idx, seg=None, gap_factor=6.0, min_pts=1):
    '''Split one reflection's sampled sweep into the continuous arcs it contains,
    returning a list of index arrays into `pts`.

    A DMS locus is not one connected sweep.  Three things break it, and all three
    have to be cut or a circle gets fitted across the join:

    * the Ewald construction drops out wherever the intersection is non-physical,
      leaving gaps in the scan (`idx`, the surviving scan-step numbers);
    * a large jump between consecutive surviving points, i.e. the locus leaving
      and re-entering the physical region between two steps;
    * a change of `seg`, an arbitrary per-point label for "the geometry the scan
      is in".  `imcalc` passes the sign of the scan vector along the primary:
      a theta range spanning zero (the slider's default [thb-27, thb+10] does
      whenever thb < 27 deg) reverses the scan vector, which re-aligns the
      crystal and puts the rest of the sweep on a *different* circle.  Without
      this cut such a run fits a circle wrong by ~0.2 rad instead of ~1e-15.'''
    pts = np.asarray(pts, dtype=float)
    idx = np.asarray(idx)
    if len(pts) < min_pts:
        return []
    d  = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    nz = d[d > 0]
    brk = (d > gap_factor * np.median(nz)) if nz.size else np.zeros(len(d), bool)
    brk |= (np.diff(idx) != 1)
    if seg is not None:
        brk |= (np.diff(np.asarray(seg)) != 0)
    runs = np.split(np.arange(len(pts)), np.flatnonzero(brk) + 1)
    return [r for r in runs if len(r) >= min_pts]


def dms_fit_arcs(P, runs):
    '''Fit the circle every run lies on, in one batched pass.

    `P` is an (M,3) array of points and `runs` a list of index arrays into it,
    each in sweep order.  Returns `(centres, radii, normals, a0, a1, resid)`:
    the arc runs a0 -> a1 the way the sweep traversed it (angles measured in the
    `dms_plane_basis` frame of its normal, unwrapped within the run so an arc of
    any length has an unambiguous span), and `resid` is the run's worst deviation
    from the fitted circle — radial or out-of-plane, whichever is larger.

    Batched rather than one fit per run because a discovery slice holds hundreds
    of runs and the per-call numpy overhead of a scalar fit dominates everything
    else.  Each quantity is a `reduceat` over the concatenated runs followed by a
    batched 3x3 `eigh` / `solve`.'''
    lens = np.array([len(r) for r in runs])
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    rid  = np.repeat(np.arange(len(runs)), lens)
    Q    = np.asarray(P, dtype=float)[np.concatenate(runs)]

    c0 = np.add.reduceat(Q, offs, axis=0) / lens[:, None]
    # Scatter matrix per run: sum p(x)p - n.c(x)c.  Its least eigenvector is the
    # plane normal.
    S = (np.add.reduceat(Q[:, :, None] * Q[:, None, :], offs, axis=0)
         - lens[:, None, None] * c0[:, :, None] * c0[:, None, :])
    n = np.linalg.eigh(S)[1][:, :, 0]
    u = dms_plane_basis(n)
    v = np.cross(n, u)

    X = Q - c0[rid]
    x = np.einsum('ij,ij->i', X, u[rid])
    y = np.einsum('ij,ij->i', X, v[rid])
    b = x * x + y * y
    red = lambda a: np.add.reduceat(a, offs)
    # 3x3 normal equations of |p|^2 = 2.cx.x + 2.cy.y + (r^2 - |c|^2), per run.
    AtA = np.empty((len(runs), 3, 3))
    AtA[:, 0, 0] = red(4 * x * x); AtA[:, 0, 1] = AtA[:, 1, 0] = red(4 * x * y)
    AtA[:, 1, 1] = red(4 * y * y); AtA[:, 0, 2] = AtA[:, 2, 0] = red(2 * x)
    AtA[:, 1, 2] = AtA[:, 2, 1] = red(2 * y); AtA[:, 2, 2] = lens
    Atb = np.stack([red(2 * x * b), red(2 * y * b), red(b)], axis=1)
    # A degenerate run (collinear points) has a singular normal matrix, which
    # would take the whole batch down with a LinAlgError; leave those NaN and let
    # the residual check reject them.
    sol = np.full((len(runs), 3), np.nan)
    det = np.linalg.det(AtA)
    solvable = np.isfinite(det) & (np.abs(det) > 0)
    if solvable.any():
        # b as a stack of column vectors: numpy >= 2 reads a bare 2-D b as one
        # matrix.
        sol[solvable] = np.linalg.solve(AtA[solvable],
                                        Atb[solvable][:, :, None])[:, :, 0]
    cx, cy = sol[:, 0], sol[:, 1]
    with np.errstate(invalid='ignore'):
        r = np.sqrt(np.maximum(sol[:, 2] + cx * cx + cy * cy, 0.0))

    dx, dy = x - cx[rid], y - cy[rid]
    resid = np.maximum(
        np.maximum.reduceat(np.abs(np.sqrt(dx * dx + dy * dy) - r[rid]), offs),
        np.maximum.reduceat(np.abs(np.einsum('ij,ij->i', X, n[rid])), offs))

    # Unwrap the angle within each run (never across a run boundary), so a0 -> a1
    # spans the arc the way the sweep traversed it, for arcs of any length.
    ang = np.arctan2(dy, dx)
    d = np.diff(ang, prepend=ang[0])
    d = (d + np.pi) % (2 * np.pi) - np.pi
    d[offs] = 0.0
    cum = np.cumsum(d)
    a0 = ang[offs]
    a1 = a0 + (cum[offs + lens - 1] - cum[offs])
    return c0 + cx[:, None] * u + cy[:, None] * v, r, n, a0, a1, resid


def dms_arc_points(centres, radii, normals, a0, a1, counts):
    '''Tessellate arcs: `counts[i]` points along arc `i`, from a0 to a1.

    Returns `(points, counts)` with the points of every arc concatenated in
    order, so the caller can project them all in one pass and split them again
    with the counts.'''
    counts = np.maximum(np.asarray(counts, dtype=int), 2)
    total  = int(counts.sum())
    offs   = np.concatenate([[0], np.cumsum(counts)[:-1]])
    aid    = np.repeat(np.arange(len(counts)), counts)
    t      = (np.arange(total) - np.repeat(offs, counts)) / (np.repeat(counts, counts) - 1.0)
    ang    = np.repeat(a0, counts) + t * np.repeat(np.asarray(a1) - np.asarray(a0), counts)
    u = dms_plane_basis(normals)
    v = np.cross(normals, u)
    pts = (np.asarray(centres)[aid]
           + (np.asarray(radii)[aid] * np.cos(ang))[:, None] * u[aid]
           + (np.asarray(radii)[aid] * np.sin(ang))[:, None] * v[aid])
    return pts, counts


def _on_image_length(r0, c0, r1, c1, shape):
    '''Length of the part of each segment (r0,c0)->(r1,c1) inside the image.

    Liang-Barsky clipping, vectorised.  Testing the *endpoints* instead is not
    enough: a locus can cross the whole plate between two coarse samples, and
    both ends being off the image would score that stretch zero — precisely the
    fast-moving curve that most needs the points.  The result is capped at the
    image diagonal, which also disposes of the infinities the gnomonic
    projection produces where a locus grazes the detector plane.'''
    H, W = float(shape[0]), float(shape[1])
    dr, dc = r1 - r0, c1 - c0
    t0 = np.zeros_like(dr); t1 = np.ones_like(dr)
    inside = np.ones(dr.shape, dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        for p, q in ((-dr, r0), (dr, H - r0), (-dc, c0), (dc, W - c0)):
            par = (p == 0)
            inside &= ~(par & (q < 0))
            t = np.where(par, 0.0, q / p)
            t0 = np.where(~par & (p < 0), np.maximum(t0, t), t0)
            t1 = np.where(~par & (p > 0), np.minimum(t1, t), t1)
        out = np.hypot(dr, dc) * np.clip(t1 - t0, 0.0, None)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.where(inside, np.minimum(out, np.hypot(H, W)), 0.0)


def dms_arc_points_even(centres, radii, normals, a0, a1, project,
                        spacing_px=0.5, min_points=32, max_points=4000,
                        shape=None, probe=256):
    '''Tessellate arcs at even *pixel* spacing along the projected curve.

    Spreading points evenly in *angle* is the obvious thing and the wrong one:
    the projection to the detector is far from uniform, so an arc that crosses
    the plate quickly at one end comes out sampled several pixels apart there
    while the rest of it is sampled many times per pixel.  Closing the sparse
    part by raising the count alone costs an order of magnitude more points than
    the curve needs (76k to close a 2 px gap, in the case that prompted this).

    So the arc is measured first at a coarse `probe` resolution, its projected
    on-image path length is accumulated, and the final angles are placed at even
    intervals of *that* — which is one `searchsorted` over all arcs at once, not
    a loop.  Points the projection sends off the image accumulate no length, so
    they cost nothing and the budget goes where the curve can be seen.

    The probe runs **twice**: most of an arc is usually off the image, so a
    single pass spends its resolution where nothing is drawn and leaves the
    visible window coarse (3 px gaps at probe=256 in the case that prompted
    this; 1024 fixes it at 35% more cost). The first pass only locates the
    window; the second re-probes *that*, which buys the resolution of a
    ~10x finer probe for the price of one more coarse one.

    Returns `(pts, counts)` like `dms_arc_points`.'''
    n_arcs = len(radii)
    if n_arcs == 0:
        return np.empty((0, 3)), np.empty(0, int)
    spacing_px = max(spacing_px, 1e-3)
    probe = max(int(probe), 8)
    a0 = np.asarray(a0, dtype=float); a1 = np.asarray(a1, dtype=float)

    def _measure(lo, hi, n, sel=None):
        """Projected on-image length of each probe segment over [lo, hi].

        `sel` restricts the work to a subset of the arcs — most of a scene's
        arcs never reach the plate, and probing those is the single largest
        cost in this whole path."""
        idx = slice(None) if sel is None else sel
        m = n_arcs if sel is None else len(np.atleast_1d(sel))
        pp, _ = dms_arc_points(np.asarray(centres)[idx], np.asarray(radii)[idx],
                               np.asarray(normals)[idx], lo, hi, np.full(m, n))
        rows, cols = project(pp)
        rows = np.asarray(rows, dtype=float).reshape(m, n)
        cols = np.asarray(cols, dtype=float).reshape(m, n)
        if shape is None:
            seg = np.hypot(np.diff(rows, axis=1), np.diff(cols, axis=1))
        else:
            seg = _on_image_length(rows[:, :-1], cols[:, :-1],
                                   rows[:, 1:],  cols[:, 1:], shape)
        return np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

    # Pass 1: narrow each arc to the angular window that reaches the image, plus
    # one probe step of margin either side.  A quarter of the resolution is
    # plenty to *find* the window — `_on_image_length` measures the chord, so a
    # crossing between two samples is still detected, not missed.
    coarse = max(probe // 4, 8)
    seg = _measure(a0, a1, coarse)
    hit = seg > 0
    vis = hit.any(1)
    step = 1.0 / (coarse - 1.0)
    f0 = np.clip(np.argmax(hit, axis=1) * step - step, 0.0, 1.0)
    f1 = np.clip((coarse - 1 - np.argmax(hit[:, ::-1], axis=1)) * step + step,
                 0.0, 1.0)
    span = a1 - a0
    a0 = np.where(vis, a0 + f0 * span, a0)
    a1 = np.where(vis, a0 + (f1 - f0) * span, a1)

    # Pass 2: the measurement the point placement is made from — run only on the
    # arcs that reach the image.  The rest carry zero length and fall through to
    # a uniform tessellation of `min_points` below, which is all an invisible
    # arc needs.
    seg = np.zeros((n_arcs, probe - 1))
    if vis.any():
        seen = np.flatnonzero(vis)
        seg[seen] = _measure(a0[seen], a1[seen], probe, seen)
    cum = np.concatenate([np.zeros((n_arcs, 1)), np.cumsum(seg, axis=1)], axis=1)
    length = cum[:, -1]

    counts = np.clip(np.ceil(length / spacing_px),
                     min_points, max_points).astype(int)
    offs   = np.concatenate([[0], np.cumsum(counts)[:-1]])
    total  = int(counts.sum())
    t = ((np.arange(total) - np.repeat(offs, counts))
         / (np.repeat(counts, counts) - 1.0))

    # Invert the cumulative length, every arc in one pass: offsetting arc i's
    # curve by i*big makes the concatenation globally monotonic, so a single
    # searchsorted brackets every target in its own arc.
    big  = float(length.max()) * 2.0 + 1.0
    base = (np.arange(n_arcs) * big)
    j = np.searchsorted(cum.ravel() + np.repeat(base, probe),
                        t * np.repeat(length, counts) + np.repeat(base, counts))
    lo = np.repeat(np.arange(n_arcs) * probe, counts) + 1
    j  = np.clip(j, lo, lo + (probe - 2))
    flat = cum.ravel()
    c0, c1 = flat[j - 1], flat[j]
    w = np.where(c1 > c0,
                 (t * np.repeat(length, counts) - c0) / np.maximum(c1 - c0, 1e-12),
                 0.0)
    # probe index within the arc, as a fraction of the span
    frac = ((j - lo + w) / (probe - 1.0))
    # An arc the probe never saw on the image has no length to spread points
    # along; keep it uniform in angle rather than collapsing it to a point, since
    # a fast locus can cross the image between two probe samples.
    frac = np.where(np.repeat(length, counts) > 0, frac, t)

    ang = np.repeat(a0, counts) + frac * np.repeat(np.asarray(a1) - np.asarray(a0),
                                                   counts)
    aid = np.repeat(np.arange(n_arcs), counts)
    u = dms_plane_basis(normals)
    v = np.cross(normals, u)
    pts = (np.asarray(centres)[aid]
           + (np.asarray(radii)[aid] * np.cos(ang))[:, None] * u[aid]
           + (np.asarray(radii)[aid] * np.sin(ang))[:, None] * v[aid])
    return pts, counts


def dms_circle_runs(vecs, n_steps, n_refs, segment=None, min_fit_pts=4):
    '''Cut the engine's exit-direction array into the continuous runs a circle
    can be fitted to.

    `vecs` is laid out as `dms_circle_curves` describes (two psi branches, each
    `n_steps * n_refs + 1` rows with a leading sentinel).  Returns
    `(P, runs, owner, fallback)`: the run points concatenated, index arrays into
    `P` (one per run, in sweep order), the reflection each run belongs to, and
    the `(reflection, points)` runs too short to fit.  Shared by the drawn
    curves and `ridgefit`, which scores observed points against these circles.'''
    vecs = np.asarray(vecs, dtype=float)
    half = n_steps * n_refs + 1
    owner, pts_all, fallback = [], [], []
    if n_steps < 1 or n_refs < 1 or vecs.shape[0] < 2 * half:
        return np.empty((0, 3)), [], np.empty(0, int), fallback
    seg = None if segment is None else np.asarray(segment).ravel()
    for b in (0, 1):
        blk = vecs[b * half:(b + 1) * half][1:].reshape(n_steps, n_refs, 3)
        for j in range(n_refs):
            p  = blk[:, j, :]
            idx = np.flatnonzero(np.isfinite(p).all(axis=1))
            if idx.size == 0:
                continue
            pp = p[idx]
            for run in dms_split_runs(pp, idx,
                                      None if seg is None else seg[idx]):
                if len(run) < min_fit_pts:
                    # Too few points to determine a circle from — a locus that
                    # only survives a step or two of this scan.  Kept as sampled
                    # rather than dropped: at a coarse scan these are exactly the
                    # fast-moving loci, and losing them would make the analytic
                    # method show *less* than the sweep it is meant to reproduce.
                    fallback.append((j, pp[run]))
                    continue
                owner.append(j)
                pts_all.append(pp[run])
    if not pts_all:
        return np.empty((0, 3)), [], np.empty(0, int), fallback
    P     = np.concatenate(pts_all)
    lens  = np.array([len(p) for p in pts_all])
    offs  = np.concatenate([[0], np.cumsum(lens)[:-1]])
    runs  = [np.arange(o, o + l) for o, l in zip(offs, lens)]
    return P, runs, np.asarray(owner, dtype=int), fallback


def dms_circle_curves(vecs, n_steps, n_refs, project, segment=None, shape=None,
                      spacing_px=0.5, min_points=32, max_points=4000,
                      min_fit_pts=4, tol=DMS_CIRCLE_TOL):
    '''Per-reflection DMS curves built from the circles the sweep lies on.

    `vecs` is the engine's full exit-direction array — the two solution branches
    stacked, each `n_steps * n_refs + 1` rows with a leading sentinel, ordered
    step-major then reflection (exactly what `imcalc` builds).  `project` maps an
    (N,3) block of directions to float `(rows, cols)` detector pixels; it is
    called here on the sampled points, to size each arc's tessellation from the
    pixel path it actually covers.  `shape` is the image `(rows, cols)`, used
    only for that sizing (points off it are not worth resolving).

    Returns `(pts, counts, owner, resid, fallback)`:
      pts       (M,3) tessellated arc points, all arcs concatenated
      counts    points per arc
      owner     reflection index per arc
      resid     worst circle-fit deviation over the accepted arcs (rad)
      fallback  list of (reflection index, point array) for runs kept as the
                points the sweep sampled — too short to fit a circle to, or a
                fit the tolerance rejected.  Every sampled point ends up in one
                or the other, so switching method can never lose a curve.'''
    empty = (np.empty((0, 3)), np.empty(0, int), np.empty(0, int))
    P, runs, owner, fallback = dms_circle_runs(vecs, n_steps, n_refs, segment,
                                               min_fit_pts)
    if not runs:
        return empty + (0.0, fallback)

    c, r, n, a0, a1, resid = dms_fit_arcs(P, runs)

    # How finely to re-sample each arc: from the pixel path its own sampled
    # points cover, so an arc that crosses the detector gets sub-pixel spacing
    # and one that barely grazes it does not pay for points it cannot show.
    good = np.isfinite(r) & (r > 0) & np.isfinite(resid) & (resid <= tol)
    fallback += [(int(owner[i]), P[runs[i]]) for i in np.flatnonzero(~good)]
    worst = float(np.max(resid[good])) if good.any() else 0.0
    if not good.any():
        return empty + (worst, fallback)

    # Re-sampled at even pixel spacing along the projected curve, so the budget
    # goes where the curve is visible and no part of it is left coarse.
    pts, counts = dms_arc_points_even(
        c[good], r[good], n[good], a0[good], a1[good], project,
        spacing_px=spacing_px, min_points=min_points, max_points=max_points,
        shape=shape)
    return pts, counts, owner[good], worst, fallback


def _pre_components(pre, irmat):
    '''Plate-plane intersection points -> unrounded detector-frame components
    (0: normal to the scattering plane, 2: in it), NaN rows kept.'''
    pv = np.asarray(pre, dtype=float) @ np.asarray(irmat, dtype=float)
    return pv[:, 0], pv[:, 2]


def _spacing_weights(P):
    '''Per-point line length (px) of one ordered (n, 2) run, capped at 1: the
    mean distance to its neighbours, ignoring NaN gaps.  A point whose
    neighbours are a pixel or more away — a coarse sweep — weighs 1, as each
    point did in the binary image; an over-sampled arc shares its pixel.'''
    n = len(P)
    if n == 0:
        return np.zeros(0)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1) if n > 1 else np.zeros(0)
    fwd = np.concatenate([d, [np.nan]])
    bwd = np.concatenate([[np.nan], d])
    both = np.stack([fwd, bwd])
    cnt = np.isfinite(both).sum(0)
    s = np.where(cnt > 0, np.nansum(both, 0) / np.maximum(cnt, 1), 1.0)
    return np.minimum(s, 1.0)


def _block_point_weights(rc, lens):
    '''_spacing_weights over consecutive blocks of `rc` of the given lengths.'''
    w = np.ones(len(rc))
    start = 0
    for ln in lens:
        ln = int(ln)
        w[start:start + ln] = _spacing_weights(rc[start:start + ln])
        start += ln
    return w


def _sweep_point_weights(rc, n_steps, n_refs):
    '''_spacing_weights for the sweep layout imcalc builds: two psi branches of
    [one leading point, then n_steps x n_refs step-major], each reflection's
    column a run.  Any other layout falls back to weight 1 per point.'''
    w = np.ones(len(rc))
    n_half = n_steps * n_refs + 1
    if n_steps < 1 or n_refs < 1 or len(rc) != 2 * n_half:
        return w
    for h in range(2):
        base = h * n_half + 1
        P = rc[base:base + n_steps * n_refs].reshape(n_steps, n_refs, 2)
        W = np.stack([_spacing_weights(P[:, j]) for j in range(n_refs)], axis=1)
        w[base:base + n_steps * n_refs] = W.reshape(-1)
    return w


def splat_points(shape, rows, cols, weights=None):
    '''Image of `shape` with each point's weight shared bilinearly over the four
    pixels around it (pixel centres at integer coordinates).  The anti-aliased
    counterpart of setting the nearest pixel to 1: the image changes smoothly
    as a point moves.  Non-finite points and off-plate shares are dropped.'''
    h, w = int(shape[0]), int(shape[1])
    r = np.asarray(rows, dtype=float)
    c = np.asarray(cols, dtype=float)
    wt = np.ones(len(r)) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(r) & np.isfinite(c) & np.isfinite(wt)
    r, c, wt = r[ok], c[ok], wt[ok]
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    fr, fc = r - r0, c - c0
    idx, val = [], []
    for dr, dc, f in ((0, 0, (1 - fr) * (1 - fc)), (1, 0, fr * (1 - fc)),
                      (0, 1, (1 - fr) * fc), (1, 1, fr * fc)):
        rr, cc = r0 + dr, c0 + dc
        m = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        idx.append(rr[m] * w + cc[m])
        val.append((wt * f)[m])
    if not idx:
        return np.zeros((h, w))
    img = np.bincount(np.concatenate(idx), weights=np.concatenate(val),
                      minlength=h * w)
    return img.reshape(h, w)


def _fit_roi_gauss(imsim, kernel_slice, width):
    sumvals, roi = msroi(imsim, kernel_slice, width)
    xdata = np.arange(len(sumvals))
    ydata = sumvals[:, 0]
    try:
        coef, pcov, fitpoints = fitgauss(xdata, ydata)
        return coef
    except:
        return np.array([100, 100, 100, 100])

from types import SimpleNamespace as _Geometry


class dmsfit_ico_hkl(object):
    '''
    If Bravais is set to icosahedral\n 
    The lattice parameter (a) aswell as the phason
    strain matrix will be optimised.\n
     
    If Bravais is set to icosahedral_fixed_a, only the phason strain matrix will be optimised.\n
    
    If Bravais is set to cubic_no_strain, only (a) will be optimised. 
    
    If Bravais is set to callibrate, only the experimental geometry will be optimised.
    Only the phason strain matrix will be optimised.\n
    '''

    def __init__(self,*args):
        self.reflist=args[0]
        self.hkllistrange=args[1]
        self.hklint=args[2]
        self.psirange=args[3]
        self.width=args[4] # intensity > width
        self.centres=args[5]
        self.kernel=args[6] # threshold > kernel
        self.hkl=args[7]
        self.detvects=args[8]
        self.imdata=args[9]
        self.simsigma=args[10]
        self.azir=args[11]
        self.psi=args[12]
        self.px=args[13]
        self.py=args[14]
        self.scatv=args[15]
        self.bravais=args[16]
        self.detopt=args[17]
        self.energyopt=args[18]
        self.detdistancepx=args[19]
        self.detxrot=args[20]
        self.detyrot=args[21]
        self.detzrot=args[22]
        self.energy=args[23]
        self.reflist2=args[24]
        self.mtrx = args[25]
        self.a=args[26]
        self.calibration_lattice = [5.43075,5.43075,5.43075,90.0,90.0,90.0]
        # Peak-position method for the simulated ROI curves: 'gauss' (curve fit)
        # or 'centroid' (centre of mass).  Set via setPeakMethod.  `peaksig` is
        # the doublet threshold and must match the one the experimental centres
        # in self.centres were extracted with — see AUTO_DOUBLET_SIG.
        self.peakmethod = 'gauss'
        self.peaksig = AUTO_DOUBLET_SIG
        # How the DMS curves are computed from the theta scan: 'sweep' (the
        # sampled points are the curve) or 'circle' (each continuous run is
        # reduced to the circle it analytically lies on and re-sampled at
        # detector resolution).  See the DMS_CURVE_METHODS section above and
        # setCurveMethod.  `circle_residual` is the worst deviation of any run
        # from its fitted circle in the last imcalc (radians of arc; None in
        # sweep mode), so a caller can report when the analytic form has stopped
        # being exact instead of hiding it.
        self.curvemethod    = 'sweep'
        self.curvespacing   = 0.5      # target pixel spacing of a tessellated arc
        self.curvemaxpoints = 4000     # cap on the points spent on one arc
        self.circle_residual = None
        # Failure bookkeeping from the last _centre_residuals call, so a caller
        # can tell the user which ROIs are contributing nothing (no experimental
        # target) or contributing a penalty (simulated peak not locatable).
        self.n_sim_failed = 0
        self.n_no_target  = 0
        # Full 24-element guess vector — used by the conventional-crystal branch
        # of imcalc to fill the non-refined parameters around the reduced
        # optimiser vector.  Defaults to args[26] (the lattice 'a') in slot 0.
        self.ig_full = None
        # Optional replacement objective: a callable mapping the reduced vector
        # to a residual vector.  When set, fit / residuals / full score it
        # instead of the ROI centres, and fit / residuals skip imcalc (the
        # simulated image is not needed to score it; full still draws one for
        # display).  ridgefit installs its ridge + cone residual here, so every
        # optimiser that drives this engine works on it unchanged.
        self.objective = None
        # ROI bin step (px): None is the legacy whole-pixel integrator and the
        # binary simulated image; a value selects the sub-pixel integrator and
        # the splatted image (setRoiBinStep).
        self.roibin = None
        self._roi_samplers = None
    def setObjective(self, fn):
        self.objective = fn
    def setCalLattice(self, cal_lattice):
        self.calibration_lattice = cal_lattice
    def setIGFull(self, ig24):
        '''Store the full 24-element guess vector (conventional crystals only).
        The optimiser passes a reduced subset to imcalc; the remaining slots
        (constrained lattice params, detector/energy when not refined, and the
        always-zero phason block) are read back from this template.'''
        self.ig_full = np.asarray(ig24, dtype=float).copy()
    @property
    def kernel(self):
        return self._kernel
    @kernel.setter
    def kernel(self, kernel):
        # The ROI pixels are found once here, not on every evaluation (see
        # roi_kernel_indices).  A property, so reassigning the kernel — as
        # ridgefit does — can never leave stale indices behind.
        self._kernel = kernel
        self._kernel_idx = roi_kernel_indices(kernel)
        self._roi_samplers = None
    def setLattice(self, lattice):
        self.lattice = lattice
    def setCurveMethod(self, method, spacing_px=None, max_points=None):
        '''Choose how the DMS curves are computed from the theta scan.

        'sweep'  — the sampled scan points are the curve (the original path):
                   `hkllistrange[2]` sets both where each arc ends *and* how
                   smooth it is, so a coarse scan gives faceted lines.
        'circle' — each continuous run is reduced to the circle it exactly lies
                   on and re-sampled at `spacing_px` pixel spacing (capped at
                   `max_points` per arc).  The scan then only locates the *ends*
                   of each arc, so it can be run far coarser for the same — in
                   fact smoother — curves and simulated image.'''
        self.curvemethod = dms_curve_method(method)
        if spacing_px is not None:
            self.curvespacing = float(spacing_px)
        if max_points is not None:
            self.curvemaxpoints = int(max_points)

    def setRoiBinStep(self, step):
        '''Bin width (px) across each ROI for the simulated curves — must be the
        step the experimental curves were extracted with.  None (or 0) is the
        legacy whole-pixel integrator with a binary simulated image; a value
        integrates at sub-pixel bins along the local normal (roi_sampler) and
        splats the simulated lines bilinearly, so the simulated centres move
        continuously with the parameters instead of in whole-pixel steps.'''
        self.roibin = roi_bin_step(step)
        self._roi_samplers = None

    def _samplers(self):
        '''(samplers, sample set) for the current kernel, width and bin step,
        built once and reused by every evaluation.'''
        key = (float(self.width), self.roibin, tuple(np.shape(self.imdata)))
        if self._roi_samplers is None or self._roi_samplers[0] != key:
            smp = [roi_sampler(idx, self.width, self.roibin)
                   for idx in self._kernel_idx]
            self._roi_samplers = (key, smp,
                                  roi_sample_set(smp, np.shape(self.imdata)))
        return self._roi_samplers[1], self._roi_samplers[2]

    def setPeakMethod(self, method, sig=AUTO_DOUBLET_SIG):
        '''Set how peak positions are located in the simulated ROI curves.
        `sig` must be the value the experimental centres were extracted with,
        so the residual compares like with like (see AUTO_DOUBLET_SIG).'''
        self.peakmethod = method
        self.peaksig = sig
    def _simcoeffs(self):
        '''Per-ROI peak coefficients of the current simulated image, using the
        selected peak-position method.  v1[:,2] is the centre per ROI.'''
        v1=np.array([[]]*4).T
        if getattr(self, 'roibin', None) is not None:
            samplers, sset = self._samplers()
            for smp, ydata in zip(samplers,
                                  roi_profiles(self.imsim, samplers, sset)):
                try:
                    coef, pcov, fitpoints = peakfit(smp['x'], ydata,
                                                    self.peakmethod, self.peaksig)
                    v1 = np.vstack([v1, _centre_in_strip(coef, smp['x'])])
                except Exception:
                    v1 = np.vstack([v1, [0, 0, np.nan, 0]])
            return v1
        for i1 in range(len(self._kernel_idx)):
            sumvals,roi = msroi(self.imsim,self._kernel_idx[i1],self.width)
            xdata=np.arange(len(sumvals))
            ydata=sumvals[:,0]
            try:
                coef, pcov,fitpoints = peakfit(xdata,ydata,self.peakmethod,self.peaksig)
                v1=np.vstack([v1,coef])
            except Exception:
                # NaN centre, not a plausible-looking pixel index: a fixed
                # constant here can sit near a real target and read as a good
                # match.  Converted to an explicit penalty in _centre_residuals.
                v1=np.vstack([v1,[0,0,np.nan,0]])
        return v1
        
###########################
    def _geometry(self, inputs):
        '''The engine's geometry for one parameter vector, without drawing
        anything: every reflection's exit directions over the theta sweep (both
        psi branches, as `imcalc` stacks them) and the detector map they are
        projected through.  `imcalc` renders the simulated image from this;
        `ridgefit` scores observed line points against the cones it holds,
        which needs no image at all.'''
        chicorrection = 0.0    # chi-axis correction; only conventional crystals set it
        thetacorrection = 0.0  # theta (Bragg-angle) correction; conventional only
        if self.bravais in CONVENTIONAL_SYSTEMS:
            # Conventional crystal: scatter the reduced optimiser vector back into
            # a full 24-element guess, apply the crystal-system lattice
            # constraint, and leave the phason block at zero.
            full = (self.ig_full.copy() if self.ig_full is not None
                    else np.zeros(24))
            full[reduced_param_indices(self.bravais, self.detopt, self.energyopt)] = inputs
            a, b, c, alpha, beta, gamma = expand_lattice(self.bravais, full[:6])
            psicorrection = full[6]
            chicorrection = full[7]    # slot 7 (formerly hcor) repurposed for chi
            thetacorrection = full[8]  # slot 8 (formerly kcor) repurposed for theta
            h_correction = k_correction = l_correction = 0.0
            detdistancepx, detxrot, detyrot, detzrot = full[10], full[11], full[12], full[13]
            energy = full[14] if self.energyopt else self.energy
            self.a11,self.a12,self.a13,self.a21,self.a22,self.a23,self.a31,self.a32,self.a33 = 0,0,0, 0,0,0, 0,0,0
        elif self.bravais == 'icosahedral':
            a,b,c,alpha,beta,gamma=inputs[0],inputs[0],inputs[0],90.0,90.0,90.0
            psicorrection   = inputs[1]
            chicorrection   = inputs[2]   # slot 7 → chi correction
            thetacorrection = inputs[3]   # slot 8 → theta correction
            h_correction = k_correction = l_correction = 0.0
            martix_indices = list(-np.r_[1:10])
            martix_indices.reverse()
            self.a11,self.a12,self.a13,self.a21,self.a22,self.a23,self.a31,self.a32,self.a33 = inputs[martix_indices] 
            if self.detopt:
                detdistancepx,detxrot,detyrot,detzrot =inputs[5],inputs[6],inputs[7],inputs[8]
                if self.energyopt:
                    energy = inputs[9]
                else:
                    energy = self.energy
            else:
                detdistancepx,detxrot,detyrot,detzrot =self.detdistancepx,self.detxrot,self.detyrot,self.detzrot
                if self.energyopt:
                    energy = inputs[5]
                else:
                    energy = self.energy

        elif self.bravais == 'icosahedral_fixed_a':
            a,b,c,alpha,beta,gamma=self.lattice
            psicorrection   = inputs[0]
            chicorrection   = inputs[1]   # slot 7 → chi correction
            thetacorrection = inputs[2]   # slot 8 → theta correction
            h_correction = k_correction = l_correction = 0.0
            martix_indices = list(-np.r_[1:10])
            martix_indices.reverse()
            self.a11,self.a12,self.a13,self.a21,self.a22,self.a23,self.a31,self.a32,self.a33 = inputs[martix_indices] 
            if self.detopt:
                detdistancepx,detxrot,detyrot,detzrot =inputs[4],inputs[5],inputs[6],inputs[7]
                if self.energyopt:
                    energy = inputs[8]
                else:
                    energy = self.energy
            else:
                detdistancepx,detxrot,detyrot,detzrot =self.detdistancepx,self.detxrot,self.detyrot,self.detzrot
                if self.energyopt:
                    energy = inputs[4]
                else:
                    energy = self.energy                  
                    
        elif self.bravais == 'cubic_no_strain':
            a,b,c,alpha,beta,gamma=inputs[0],inputs[0],inputs[0],90.0,90.0,90.0
            psicorrection   = inputs[1]
            chicorrection   = inputs[2]   # slot 7 → chi correction
            thetacorrection = inputs[3]   # slot 8 → theta correction
            h_correction = k_correction = l_correction = 0.0
            self.a11,self.a12,self.a13,self.a21,self.a22,self.a23,self.a31,self.a32,self.a33 = 0,0,0, 0,0,0, 0,0,0
            if self.detopt:
                detdistancepx,detxrot,detyrot,detzrot =inputs[5],inputs[6],inputs[7],inputs[8]
                self.a11,self.a12,self.a13,self.a21,self.a22,self.a23,self.a31,self.a32,self.a33 = 0,0,0, 0,0,0, 0,0,0
                if self.energyopt:
                    energy = inputs[9]
                else:
                    energy = self.energy
            else:
                detdistancepx,detxrot,detyrot,detzrot =self.detdistancepx,self.detxrot,self.detyrot,self.detzrot
                if self.energyopt:
                    energy = inputs[5]
                else:
                    energy = self.energy
                    
                    
        elif self.bravais == 'calibrate':
                a = self.calibration_lattice[0]
                b = self.calibration_lattice[1]
                c = self.calibration_lattice[2]
                alpha = self.calibration_lattice[3]
                beta = self.calibration_lattice[4]
                gamma = self.calibration_lattice[5]
                psicorrection   = inputs[0]
                chicorrection   = inputs[1]   # slot 7 → chi correction
                thetacorrection = inputs[2]   # slot 8 → theta correction
                h_correction = k_correction = l_correction = 0.0
                self.a11,self.a12,self.a13,self.a21,self.a22,self.a23,self.a31,self.a32,self.a33 = 0,0,0, 0,0,0, 0,0,0
                if self.detopt:
                    detdistancepx,detxrot,detyrot,detzrot =inputs[4],inputs[5],inputs[6],inputs[7]
                    if self.energyopt:
                        energy = inputs[8]
                    else:
                        energy = self.energy
                else:
                    detdistancepx,detxrot,detyrot,detzrot =self.detdistancepx,self.detxrot,self.detyrot,self.detzrot
                    if self.energyopt:
                        energy = inputs[4]
                    else:
                        energy = self.energy

        else:
            print('Choose Bravais')

        lattice = [a,b,c,alpha,beta,gamma]
        hkl = [self.hkl[0]+h_correction,self.hkl[1]+k_correction,self.hkl[2]+l_correction]
        thb=bragg(lattice,self.hkl,energy).th()[0]
        detvs=np.array(self.detvects*rotxyz([0,0,1],-detzrot).rmat()*rotxyz([0,1,0],-detyrot).rmat()*rotxyz([1,0,0],-detxrot-thb).rmat())
        irmat=rotxyz([1,0,0],detxrot+thb).rmat()*rotxyz([0,1,0],detyrot).rmat()*rotxyz([0,0,1],detzrot).rmat()
        hkllist = pilkhlrange(lattice,hkl,energy,self.hkllistrange[0],self.hkllistrange[1]).hklscan(self.hkllistrange[2])
        if chicorrection != 0:
            # Rotate the scan list about the chi axis (perpendicular to the
            # primary reflection and its azimuthal reference) — same construction
            # as the reference dmscalc.
            chiaxis = (rotxyz(np.cross((rotxyz(self.hkl,self.psi).rmat()*np.array([self.azir]).T).T, np.array([self.hkl])),90).rmat()*np.array([self.hkl]).T).T
            hkllist = np.array((rotxyz(chiaxis, -chicorrection).rmat()*np.array(hkllist).T).T)
        mtrx2=[self.a11,self.a12,self.a13,self.a21,self.a22,self.a23,self.a31,self.a32,self.a33]

        # ── Vectorised Ewald sphere calculation ─────────────────────────────
        keV2A_ko    = 12.398
        keV2A_bragg = 12.3984187
        ko  = energy / keV2A_ko
        wl  = keV2A_bragg / energy

        bm          = np.array(bmatrix(lattice).bm())                   # (3,3)
        hkl002      = PhasonDistoArray(
            np.array(self.reflist), np.array(self.reflist2), mtrx2
        ).qe1()                                                          # (N_refs, 3)
        hkl002_cart = hkl002 @ bm.T                                     # (N_refs, 3)
        azir_cart0  = np.array(self.azir).reshape(3) @ bm.T             # (3,)
        N_refs  = hkl002.shape[0]
        N_steps = hkllist.shape[0]

        hklnotlist       = hkllist @ bm.T                                # (N_steps, 3)
        hklnotlist_norms = np.linalg.norm(hklnotlist, axis=1)
        safe_norms       = np.maximum(hklnotlist_norms, 1e-12)
        brag1_all        = 180/np.pi * np.arcsin(wl * safe_norms / 2.0) # (N_steps,)

        rotvect_all = np.cross([0.0, 0.0, 1.0], hklnotlist)             # (N_steps, 3)
        rotvect_l1  = np.sum(np.abs(rotvect_all), axis=1)

        zref_cart      = np.array([0.0, 0.0, 1.0]) @ bm.T
        zref_cart_norm = np.linalg.norm(zref_cart)
        cos_align = np.clip(
            (hklnotlist @ zref_cart) / (safe_norms * zref_cart_norm), -1.0, 1.0
        )
        alignangle_all = np.arccos(cos_align) * 180/np.pi               # (N_steps,)

        u_all = rotvect_all / np.maximum(
            np.linalg.norm(rotvect_all, axis=1, keepdims=True), 1e-12
        )
        t_rad = alignangle_all * np.pi / 180
        c_t = np.cos(t_rad);  s_t = np.sin(t_rad)
        ux, uy, uz = u_all[:,0], u_all[:,1], u_all[:,2]

        R = np.zeros((N_steps, 3, 3))
        R[:,0,0] = c_t + ux*ux*(1-c_t);  R[:,0,1] = ux*uy*(1-c_t) - uz*s_t;  R[:,0,2] = ux*uz*(1-c_t) + uy*s_t
        R[:,1,0] = uy*ux*(1-c_t) + uz*s_t;  R[:,1,1] = c_t + uy*uy*(1-c_t);  R[:,1,2] = uy*uz*(1-c_t) - ux*s_t
        R[:,2,0] = uz*ux*(1-c_t) - uy*s_t;  R[:,2,1] = uz*uy*(1-c_t) + ux*s_t;  R[:,2,2] = c_t + uz*uz*(1-c_t)
        R[rotvect_l1 < 0.0001] = np.eye(3)

        realvecthkl  = np.einsum('jr,irs->ijs', hkl002_cart, R)         # (N_steps, N_refs, 3)
        azir_rot     = np.einsum('r,irs->is', azir_cart0, R)            # (N_steps, 3)
        azir_rot[rotvect_l1 < 0.001] = azir_cart0
        azirangle_all = np.arctan2(azir_rot[:,0], azir_rot[:,1]) * 180/np.pi

        b1      = brag1_all[:,np.newaxis]
        orighk  = ko * np.cos(b1 * np.pi/180)
        raw_sin = (ko*np.sin(-b1*np.pi/180) + realvecthkl[:,:,2]) / ko
        valid   = np.abs(raw_sin) <= 1.0                                  # physical Ewald condition
        sin_arg = np.clip(raw_sin, -1.0, 1.0)
        rewl     = ko * np.cos(np.arcsin(sin_arg))
        rhk      = np.sqrt(realvecthkl[:,:,0]**2 + realvecthkl[:,:,1]**2)
        rhkangle = np.arctan2(realvecthkl[:,:,0], realvecthkl[:,:,1]) * 180/np.pi

        numer  = orighk**2 - rhk**2 + rewl**2
        half_n = numer / (2*orighk)
        disc   = rewl**2 - half_n**2
        valid &= disc >= 0                                                 # real intersection exists
        xint   = np.sqrt(np.maximum(disc, 0))

        ia1 = np.arctan2( xint, half_n - orighk) * 180/np.pi
        ia2 = np.arctan2(-xint, half_n - orighk) * 180/np.pi

        az   = azirangle_all[:,np.newaxis]
        psi1 = np.mod(ia1 + az - rhkangle + 180, 360) - 180             # (N_steps, N_refs)
        psi2 = np.mod(ia2 + az - rhkangle + 180, 360) - 180

        mslist = np.empty((N_steps * N_refs + 1, 7))
        mslist[0] = np.nan
        flat = mslist[1:].reshape(N_steps, N_refs, 7)
        flat[:,:,0:3] = hkl002[np.newaxis,:,:]
        flat[:,:,3]   = psi1
        flat[:,:,4]   = psi2
        flat[:,:,5]   = brag1_all[:,np.newaxis]
        flat[:,:,6]   = energy
        flat[~valid, 3:5] = np.nan                                        # kill non-physical solutions

        # ── Pixel projection ─────────────────────────────────────────────────
        vecs1=psith2v(self.psi-mslist[:,3]-psicorrection,mslist[:,5]+thetacorrection)
        vecs2=psith2v(self.psi-mslist[:,4]-psicorrection,mslist[:,5]+thetacorrection)
        vecs=np.concatenate((vecs1,vecs2),0)
        centralv=-psith2v(0,thb)*detdistancepx
        return _Geometry(
            lattice=lattice, hkl=hkl, energy=energy, thb=thb, detvs=detvs,
            irmat=irmat, centralv=centralv, hkllist=hkllist, vecs=vecs,
            N_steps=N_steps, N_refs=N_refs, psicorrection=psicorrection,
            chicorrection=chicorrection, thetacorrection=thetacorrection,
            l_correction=l_correction, detdistancepx=detdistancepx,
            detxrot=detxrot, detyrot=detyrot, detzrot=detzrot)

    def imcalc(self,*inputs):
        g = self._geometry(inputs[0])
        a,b,c,alpha,beta,gamma = g.lattice
        hkl, energy, thb, detvs, irmat = g.hkl, g.energy, g.thb, g.detvs, g.irmat
        centralv, hkllist, vecs = g.centralv, g.hkllist, g.vecs
        N_steps, N_refs = g.N_steps, g.N_refs
        psicorrection, chicorrection = g.psicorrection, g.chicorrection
        thetacorrection, l_correction = g.thetacorrection, g.l_correction
        detdistancepx, detxrot, detyrot, detzrot = (
            g.detdistancepx, g.detxrot, g.detyrot, g.detzrot)

        def _to_pixels(v):
            '''Exit-direction vectors -> float (row, col) detector pixels.  The
            one map both curve methods project through, so an analytic curve and
            the sampled one it was fitted to cannot land a pixel apart.

            Not rounded: the overlay is drawn at sub-pixel positions, so a line
            sits on the cone the ridge objective scores (`ridgefit.ConeModel.
            project` is this map).  Only the simulated image's pixel index is
            rounded, below, where whole pixels are needed.'''
            pre = dms2px(detvs[0,:], detvs[1,:], centralv, np.asarray(v, dtype=float))
            pv  = np.asarray(pre * irmat, dtype=float)
            return det_to_pixels(pv[:,0], pv[:,2], float(self.px), float(self.py),
                                 self.scatv)

        # ── Curve method: the sampled sweep, or the circles it lies on ────────
        self.curvemethod = dms_curve_method(getattr(self, 'curvemethod', 'sweep'))
        arcs = None
        self.circle_residual = None
        if self.curvemethod == 'circle' and N_steps > 0 and N_refs > 0:
            # Where the scan vector reverses (any theta range spanning zero) the
            # crystal alignment flips and the sweep continues on a *different*
            # circle, so the runs are cut on the sign of the scan vector along
            # the primary.  Without that cut a straddling run fits a circle
            # wrong by ~0.2 rad instead of ~1e-15.
            seg = (np.asarray(hkllist, dtype=float)
                   @ np.asarray(hkl, dtype=float)) >= 0
            # Deliberately not guarded: a fit is scored on the simulated image,
            # so quietly dropping back to the sweep for one evaluation would move
            # the objective's footing mid-run.  Either every evaluation uses the
            # circles or the caller sees the failure.
            arcs = dms_circle_curves(
                vecs, N_steps, N_refs, _to_pixels, seg,
                shape=np.shape(self.imdata),
                spacing_px=self.curvespacing,
                max_points=self.curvemaxpoints)
            self.circle_residual = arcs[3]

        prepxvec=dms2px(detvs[0,:],detvs[1,:],centralv,vecs)
        if arcs is None:
            index_src = prepxvec
        else:
            # The simulated image is drawn from the same points as the curves:
            # the tessellated arcs, plus any run whose circle was rejected (kept
            # as sampled, so no reflection silently loses its line).
            arc_pts = [arcs[0]] + [p for _, p in arcs[4]]
            index_src = dms2px(detvs[0,:], detvs[1,:], centralv,
                               np.concatenate(arc_pts) if arc_pts else np.empty((0,3)))
        valid_px = ~np.isnan(index_src).any(axis=1)
        pxvec=np.array(np.round(index_src[valid_px]*irmat).astype(int))
        imsim=np.zeros(np.shape(self.imdata))
        pxv2d=np.array(pxvec[:,[0,2]])
        self.bragg = thb
        _r, _c = det_to_pixels(pxv2d[:,0], pxv2d[:,1], self.px, self.py, self.scatv)
        pxv2d[:,0], pxv2d[:,1] = _r, _c
        pxv2d=pxv2d[np.where(pxv2d[:,0]>-1)]
        pxv2d=pxv2d[np.where(pxv2d[:,0]< imsim.shape[0])]
        pxv2d=pxv2d[np.where(pxv2d[:,1]>-1)]
        pxv2d=pxv2d[np.where(pxv2d[:,1]< imsim.shape[1])]
        if arcs is not None and pxv2d.shape[0]:
            # The arcs are deliberately over-sampled (~2 points per pixel) so no
            # pixel of a line can be skipped; collapsing the duplicates leaves
            # the index — and the overlay scatter drawn from it — the size the
            # sampled path produces, for an identical simulated image.  Packed
            # into one integer per pixel first: np.unique(axis=0) on the pairs
            # sorts a structured view and costs an order of magnitude more than
            # everything else here put together.
            w_im = int(imsim.shape[1])
            key = np.unique(pxv2d[:,0].astype(np.int64) * w_im + pxv2d[:,1])
            pxv2d = np.stack([key // w_im, key % w_im], axis=1)
        self.dmsindex=tuple([pxv2d[:,0],pxv2d[:,1]])
        # Slots 7/8 carry the chi / theta corrections (they were formerly
        # hcor/kcor — see the branches above, every one of which reads them from
        # the guess and holds h/k/l_correction at zero).  Writing the h/k
        # corrections here instead zeroed the two refined values on the way out,
        # so a fit's own result, read back through inputarray, described a
        # geometry the fit had never evaluated.
        self.inputarray = np.array([a,b,c,alpha,beta,gamma,psicorrection,chicorrection,thetacorrection,l_correction,detdistancepx,detxrot,detyrot,detzrot,energy,self.a11,self.a12,self.a13,self.a21,self.a22,self.a23,self.a31,self.a32,self.a33])
        if getattr(self, 'roibin', None) is not None:
            # Sub-pixel: the lines are splatted at their exact positions, so the
            # image (and every simulated ROI centre) moves continuously with the
            # parameters.  dmsindex stays the whole-pixel raster, for display.
            rc = det_to_pixels(*_pre_components(index_src, irmat),
                               float(self.px), float(self.py), self.scatv)
            rc = np.stack(rc, axis=1)
            if arcs is None:
                wts = _sweep_point_weights(rc, N_steps, N_refs)
            else:
                lens = list(arcs[1]) + [len(p) for _, p in arcs[4]]
                wts = _block_point_weights(rc, lens)
            imsim = splat_points(imsim.shape, rc[:, 0], rc[:, 1], wts)
        else:
            imsim[self.dmsindex]=1
        if self.simsigma != 0:
            self.imsim=blur_gauss(imsim,15,self.simsigma)
        else:
            self.imsim=imsim

        # ── Per-reflection line data for visualisation ────────────────────────
        if arcs is not None:
            self.dmslines = self._arc_dmslines(arcs, N_refs, _to_pixels,
                                               imsim.shape)
        elif N_steps > 0 and N_refs > 0:
            n_half = N_steps * N_refs + 1
            def _prepx_lines(pp_half):
                # Unrounded, as in _to_pixels: sub-pixel overlay lines.
                px = np.asarray(pp_half[1:] * irmat, dtype=float)  # (N_steps*N_refs, 3)
                px = px.reshape(N_steps, N_refs, 3)
                r, c = det_to_pixels(px[:,:,0], px[:,:,2], float(self.px),
                                     float(self.py), self.scatv)
                h_im, w_im = imsim.shape
                # Unrounded points: pixel i spans [i-0.5, i+0.5), so the plate is
                # [-0.5, size-0.5) — the same points the old rounding kept.
                oob = ((r < -0.5) | (r >= h_im - 0.5) | (c < -0.5) | (c >= w_im - 0.5)
                       | np.isnan(r) | np.isnan(c))
                r[oob] = np.nan; c[oob] = np.nan
                return r, c
            r1, c1 = _prepx_lines(prepxvec[:n_half])
            r2, c2 = _prepx_lines(prepxvec[n_half:])
            self.dmslines = [
                (np.concatenate([c1[:,j], [np.nan], c2[:,j]]),
                 np.concatenate([r1[:,j], [np.nan], r2[:,j]]))
                for j in range(N_refs)
            ]
        else:
            self.dmslines = []

    def _arc_dmslines(self, arcs, n_refs, to_pixels, shape):
        '''Per-reflection (x=cols, y=rows) overlay lines from tessellated arcs.

        Same contract as the sweep path: one entry per reflection, arcs within a
        reflection separated by a NaN, and any point off the image NaN'd, so a
        consumer cannot tell which method produced the curve apart from its
        smoothness.'''
        pts, counts, owner, _, fall = arcs
        segs = [[] for _ in range(n_refs)]
        blocks = [pts] + [p for _, p in fall]
        lens   = list(counts) + [len(p) for _, p in fall]
        owners = list(owner)   + [j for j, _ in fall]
        if not len(lens):
            return [(np.array([]), np.array([])) for _ in range(n_refs)]
        rows, cols = to_pixels(np.concatenate(blocks))
        h_im, w_im = shape
        # Plate edges for unrounded points, as in the sweep path above.
        oob = ((rows < -0.5) | (rows >= h_im - 0.5) | (cols < -0.5)
               | (cols >= w_im - 0.5)
               | np.isnan(rows) | np.isnan(cols))
        rows = np.where(oob, np.nan, rows); cols = np.where(oob, np.nan, cols)
        start = 0
        for j, ln in zip(owners, lens):
            segs[j].append((cols[start:start + ln], rows[start:start + ln]))
            start += ln
        nan = np.array([np.nan])
        out = []
        for s in segs:
            if not s:
                out.append((np.array([]), np.array([])))
                continue
            xs = [v for pair in s for v in (pair[0], nan)][:-1]
            ys = [v for pair in s for v in (pair[1], nan)][:-1]
            out.append((np.concatenate(xs), np.concatenate(ys)))
        return out

    def _roi_fail_penalty(self):
        '''Per-ROI residual charged for a ROI whose simulated peak could not be
        located.  The centre can only ever lie inside the integrated curve, so
        the largest residual a *successfully* located peak can produce is the
        ROI width; charging a multiple of it makes a failure strictly worse
        than any real miss, so the optimiser is never rewarded for driving
        lines out of their ROIs.  Scales with the ROI, unlike a bare constant.'''
        return ROI_FAIL_PENALTY_FACTOR * float(self.width)

    def _centre_residuals(self, inputs):
        '''Per-ROI centre residuals (simulated centre - target centre), the one
        primitive behind fit/residuals/full so the three cannot disagree.

        Two failure modes, deliberately treated differently:

        * The **experimental** peak could not be located (NaN in self.centres).
          That ROI has no target, so it carries no information and is dropped
          from the residual entirely — scoring it against a made-up number
          would have the fit chase a value with no physical meaning.  The user
          can supply one by right-clicking the ROI (manual centre override),
          which re-includes it.
        * The **simulated** peak could not be located (NaN from _simcoeffs).
          That ROI does have a target, and the simulation missing it is a real
          failure of the current parameters, so it is charged the full penalty.

        Returns (residual vector, n_sim_failed, n_no_target).'''
        self.imcalc(inputs)                       # adding attribute
        resid, n_fail, n_none = centre_residuals(
            self._simcoeffs()[:, 2],
            np.asarray(self.centres, dtype=float)[:, 0],
            self._roi_fail_penalty())
        self.n_sim_failed = n_fail
        self.n_no_target  = n_none
        return resid, n_fail, n_none

    def _total_failure_score(self):
        '''Objective value for an evaluation that raised before any ROI could be
        scored.  Equivalent to "every scorable ROI failed", so it is always at
        least as bad as the worst evaluation that did complete — a crash can
        never look better than a valid geometry.  The old flat 500 could be
        beaten by a merely mediocre fit, which actively rewarded the optimiser
        for walking into parameter regions that throw.'''
        n = 1
        try:
            n = max(int(np.sum(~np.isnan(np.asarray(self.centres, dtype=float)[:, 0]))), 1)
        except Exception:
            pass
        return float(n) * self._roi_fail_penalty() ** 2

    def fit(self,inputs):
        if self.objective is not None:
            return float(np.sum(np.asarray(self.objective(inputs))**2))
        try:
            resid, _, _ = self._centre_residuals(inputs)
            return float(np.sum(resid**2))
        except Exception:
            return self._total_failure_score()

    def residuals(self,inputs):
        """Per-ROI centre residual vector, for scipy.optimize.least_squares.
        `fit` is exactly np.sum(residuals**2).  A robust loss (soft_l1 / huber)
        then downweights the penalty rows produced by failed ROIs."""
        if self.objective is not None:
            return np.asarray(self.objective(inputs), dtype=float)
        try:
            resid, _, _ = self._centre_residuals(inputs)
            return resid
        except Exception:
            return np.full(self.centres.shape[0], self._roi_fail_penalty())

    def stats(self,inputs):
        self.imcalc(inputs) # adding attribute
        v1=self._simcoeffs()
        pcovarray=np.array(self.pcov)
        return np.abs((v1[:,2]-self.coefs[:,2])),pcovarray[:,2,2]

    def full(self,inputs):
        try:
            if self.objective is not None:
                self.imcalc(inputs)            # the image and lines, for display
                result = self.fit(inputs)
                return result,self.imsim, self.dmsindex, self.imdata, self.inputarray
            resid, _, _ = self._centre_residuals(inputs)
            result = float(np.sum(resid**2))
            return result,self.imsim, self.dmsindex, self.imdata, self.inputarray
        except Exception:
            return (self._total_failure_score(), np.zeros(self.imdata.shape),
                    np.array([[],[]]), self.imdata, self.inputarray)


# ── Multiple-intersection (Renninger triple-intersection) lattice fitting ───────
# Ported from calcms/ts_light.py so the tripfit workflow lives in the package
# alongside the image-based fit.  A "triple intersection" is the coincidence, on
# the stereographic projection, of the three Kossel lines of three secondary
# reflections that share a primary reflection — the multiple-diffraction geometry
# that is highly sensitive to small (e.g. pseudo-cubic) lattice distortions.
# Refining the lattice so the three lines meet at a single point (per group)
# measures those distortions without needing the detector image.

def kosscalc(lattice, energy, ref1, ref2, azir, startval, endval, steps):
    '''Kossel-line locus for secondary reflections ``ref2`` (N x 3) about primary
    ``ref1``, swept over azimuth [startval, endval] in ``steps`` points.  Returns
    an (N*steps) x 5 array of [x, y, z, psi_angle, theta_angle] grouped by
    reflection (row block ``i*steps:(i+1)*steps`` is reflection ``i``); the
    leading 3 columns are the (unnormalised) direction, used by the tripfit
    projection.  The whole reflection list is handled in one vectorised sweep so
    a triple needs a single call rather than one per Kossel line.'''
    c, e, h = 299792458, 1.602176487e-19, 6.62606896e-34
    ko = (1e7 * h * c / e) / energy          # wavelength (A); keV2A / energy
    azir = np.array(azir); ref1 = np.array(ref1); ref2 = np.array(ref2)
    bragglist = bragg(lattice, ref2, energy).th()
    bm = bmatrix(lattice).bm()
    azirc = (bm @ azir.T)
    ref1c = (bm @ ref1.T).T
    vectz = np.array([[0, 0, 1]])
    vectzc = (bm @ vectz.T).T
    angprimsec = interplanarangle(lattice, ref1, vectz).ang()
    vecttorotang = np.cross(ref1c, vectzc)
    # Rotate everything into the frame with the primary reflection along z.
    if abs(angprimsec) >= 0.001:
        ref1c = ref1c @ rotxyz(vecttorotang, angprimsec[0]).rmat()
        ref2c = (bm @ ref2.T).T @ rotxyz(vecttorotang, angprimsec[0]).rmat()
        azirc = azirc @ rotxyz(vecttorotang, angprimsec[0]).rmat()
    else:
        ref2c = (bm @ ref2.T).T
    azirangle = (np.arctan2(azirc[0, 0], azirc[0, 1]) * 180 / np.pi)
    ref2c = np.asarray(ref2c)                                    # (N, 3)
    bl = np.asarray(bragglist).ravel()
    # Per-reflection Kossel-cone base direction and sweep axis, computed row-wise
    # so a single call covers the whole reflection list.
    vectnorms = np.linalg.norm(ref2c, axis=1, keepdims=True)
    v1norm = ref2c / vectnorms * ko
    # np.roll is a no-op on the original np.matrix, so the historical expression
    # reduces to v1norm @ [[0,-1,0],[1,0,0],[0,0,1]] == [v_y, -v_x, v_z].
    rolled = np.column_stack((v1norm[:, 1], -v1norm[:, 0], v1norm[:, 2]))
    perpvects = np.cross(rolled, ref2c)
    base = np.vstack([np.asarray(np.matrix(v1norm[i])
                                 * rotxyz(perpvects[i], 90 - bl[i]).rmat())
                      for i in range(ref2.shape[0])])            # (N, 3)
    # Rotation matrix for every azimuth about each reflection's own axis, built
    # in one shot (same entries as rotxyz) and applied with a single einsum
    # instead of an inner Python loop + per-step vstack.
    axes = ref2c / np.linalg.norm(ref2c, axis=1, keepdims=True)
    angs = np.linspace(startval, endval, steps) * np.pi / 180.0
    ux = axes[:, 0:1]; uy = axes[:, 1:2]; uz = axes[:, 2:3]      # (N, 1)
    cc = np.cos(angs)[None, :]; ss = np.sin(angs)[None, :]       # (1, steps)
    e11 = ux**2 + (1 - ux**2)*cc; e12 = ux*uy*(1-cc) - uz*ss; e13 = ux*uz*(1-cc) + uy*ss
    e21 = ux*uy*(1-cc) + uz*ss; e22 = uy**2 + (1 - uy**2)*cc; e23 = uy*uz*(1-cc) - ux*ss
    e31 = ux*uz*(1-cc) - uy*ss; e32 = uy*uz*(1-cc) + ux*ss; e33 = uz**2 + (1 - uz**2)*cc
    rmats = np.stack((np.stack((e11, e12, e13), -1),
                      np.stack((e21, e22, e23), -1),
                      np.stack((e31, e32, e33), -1)), -2)        # (N, steps, 3, 3)
    v1 = np.einsum('ij,isjk->isk', base, rmats).reshape(-1, 3)   # grouped by reflection
    v1 = np.array(v1 @ rotxyz([0, 0, 1], -azirangle).rmat())
    psangles = np.reshape(np.arctan2(v1[:, 0], v1[:, 1]) * 180 / np.pi, (v1.shape[0], 1))
    thangles = np.reshape(np.arctan2(v1[:, 2], np.sqrt(v1[:, 0]**2 + v1[:, 1]**2))
                          * 180 / np.pi, (v1.shape[0], 1))
    return np.concatenate((v1, psangles, thangles), 1)


def stereoproj(vin):
    '''Stereographic projection of unit vectors ``vin`` (N x 3) onto the plane,
    returned as a 2 x N array of [x, y].  Column slices rather than bare column
    indices, so a plain ndarray projects as well as the np.matrix the sweep
    hands over, and both come back in the type they went in as.'''
    vin = np.asanyarray(vin, dtype=float)
    return np.concatenate((vin[:, 0:1] / (1. - vin[:, 2:3]),
                           vin[:, 1:2] / (1. - vin[:, 2:3])), 1).T


def triple_spread(pts):
    '''Scalar spread of three stereographic points ``pts`` (3x2): the summed
    *squared* pairwise distance |v1-v2|^2 + |v2-v3|^2 + |v1-v3|^2.  Zero only
    when all three coincide, and every term is non-negative, so a wide triple
    can never score low through cancellation.

    Squared (rather than plain) distances make this a least-squares objective:
    smooth and quadratic near the minimum instead of kinked, which is what the
    gradient-based optimisers need.'''
    v1, v2, v3 = np.asarray(pts, dtype=float)
    d12 = v1 - v2
    d23 = v2 - v3
    d13 = v1 - v3
    return float(d12 @ d12 + d23 @ d23 + d13 @ d13)


# ── Kossel lines as the circles they analytically are ──────────────────────────
# `kosscalc` sweeps each secondary reflection's exit direction a full 360 deg
# about that reflection's own axis, so the locus is a cone of unit vectors:
# *exactly* a circle on the sphere, {v : |v| = 1, v.n = c}, which the sweep only
# samples.  Crossing the sampled polylines (`intersections`, via shapely) is
# therefore crossing chords rather than arcs: the crossing sits off the true one
# by about the chord sagitta (~1/steps^2), and `triple_spread` — a *squared*
# distance — is wrong by ~1/steps^4.
#
# That made the residual a function of `resolution` rather than of the lattice,
# and left an optimiser free to reach machine zero by tuning the cell to the
# polygon instead of closing the triple: on the rhombohedral example, res=1000
# scored 8e-17 for a triple whose crossings are really 2.2e-11 apart (the
# sequence over 100/400/1000/4000 steps runs 1.0e-6, 6.3e-10, 8.1e-17, 1.2e-11).
# The GUI showed the same fit two ways for the same reason: the status bar's
# number came from the fit resolution and the control panel's from the live one.
#
# So the pair crossings are solved on the sphere instead — fit each locus's
# plane, intersect the two cones analytically, project the directions — and the
# residual is the same number at every resolution, which now only sets how
# finely the lines are drawn.

# Worst deviation from the fitted plane past which a locus is not accepted as a
# circle and the sampled polylines are crossed instead.  A genuine Kossel locus
# sits at ~5e-16 here (any resolution, both lattice families), so this is orders
# of magnitude of headroom rather than a tuned threshold.
KOSSEL_CIRCLE_TOL = 1e-9


def sphere_circle_plane(v):
    '''The plane of the circle that unit vectors ``v`` (N x 3, N >= 3) lie on:
    ``(n, c, resid)`` for ``v . n = c`` with ``n`` a unit normal, and ``resid``
    the worst deviation of any point from that plane.

    The sign is arbitrary — negating both ``n`` and ``c`` describes the same
    circle — so a locus's pair is only ever meaningful used together.  ``resid``
    is the evidence that the locus *is* a circle; a caller that depends on it
    must check it (``KOSSEL_CIRCLE_TOL``).  Raises ValueError when the points do
    not determine a plane at all (all coincident, or a circle of zero radius).'''
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
        raise ValueError('sphere_circle_plane needs an N x 3 array, N >= 3')
    # [v | 1] @ [n; -c] = 0.  The plane is the null vector of that homogeneous
    # system, taken over every sampled point in one SVD rather than from a
    # chosen three of them.  The economy SVD of a bare 3-point system returns
    # only 3 right singular vectors — dropping the very null vector wanted —
    # so that one case asks for the full basis; at N >= 4 the economy form
    # already spans it, and is what keeps a 4000-point locus cheap.
    _u, s, vt = np.linalg.svd(np.column_stack((v, np.ones(len(v)))),
                              full_matrices=len(v) < 4)
    # The null space must be one-dimensional: the system is 4 columns wide, so
    # three non-zero singular values mean exactly one plane.  Fewer and the
    # points are coincident or collinear and determine no circle.
    if s[0] <= 0 or s[2] <= 1e-10 * s[0]:
        raise ValueError('the locus does not determine a circle')
    w = vt[-1]
    nrm = np.linalg.norm(w[:3])
    if nrm < 1e-12:                       # null vector is [0,0,0,1]: no plane
        raise ValueError('the locus does not determine a circle')
    n = w[:3] / nrm
    c = float(-w[3] / nrm)
    return n, c, float(np.abs(v @ n - c).max())


def cone_pair_directions(n1, c1, n2, c2):
    '''The unit directions where two Kossel cones meet: the solutions of
    ``v . n1 = c1``, ``v . n2 = c2``, ``|v| = 1``, as a (k, 3) array with k of
    2 (the general case), 1 (tangent cones) or 0 (they do not meet).

    Writing ``v = a.n1 + b.n2 + t.(n1 x n2)`` makes the two plane conditions a
    2x2 solve for (a, b) and ``|v| = 1`` a quadratic in t, so this is closed
    form.  Raises ValueError for parallel axes, where the cones either coincide
    or never meet and there is no isolated crossing to return.'''
    n1 = np.asarray(n1, dtype=float).ravel()
    n2 = np.asarray(n2, dtype=float).ravel()
    d = float(n1 @ n2)
    det = 1.0 - d * d                     # |n1 x n2|^2
    if det <= 1e-12:
        raise ValueError('the two Kossel cones are coaxial')
    a = (c1 - c2 * d) / det
    b = (c2 - c1 * d) / det
    t2 = (1.0 - (a * a + b * b + 2.0 * a * b * d)) / det
    if t2 < -1e-12:
        return np.zeros((0, 3))
    base = a * n1 + b * n2
    if t2 <= 1e-12:                       # tangent: the two roots coincide
        out = base[None, :]
    else:
        out = np.vstack((base + np.sqrt(t2) * np.cross(n1, n2),
                         base - np.sqrt(t2) * np.cross(n1, n2)))
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def intersections(a, b):
    '''Intersection points of two closed stereographic loci ``a`` and ``b`` (each
    a sequence of 2D points).  Returns (xs, ys, ring_a, ring_b).'''
    if _LinearRing is None:
        raise ImportError('shapely is required for tripfit intersections')
    ea = _LinearRing(a)
    eb = _LinearRing(b)
    mp = ea.intersection(eb)
    geoms = getattr(mp, 'geoms', mp)      # shapely >=2 needs .geoms to iterate
    x = [p.x for p in geoms]
    y = [p.y for p in geoms]
    return x, y, ea, eb


# ── quasicrystal modes for the tripfit engine ───────────────────────────────────
# The three icosahedral modes of the image fit (dmsfit_ico_hkl) and the slider,
# handled the slider's way: 6D reflection indices are projected into parallel
# and perpendicular components with TAU_APPROX, the cell is cubic
# [a, a, a, 90, 90, 90], and each reflection's physical vector is par + M·perp
# for the phason strain matrix M (PhasonDistoArray).  The tripfit parameter
# vector grows from the 6-element lattice to 15 elements,
#     [a, b, c, alpha, beta, gamma, a11, a12, a13, a21, a22, a23, a31, a32, a33]
# so a conventional system keeps exactly the slots, and the reduced vectors, it
# had before.
QUASI_SYSTEMS = ('icosahedral', 'icosahedral_fixed_a', 'cubic_no_strain')
TRIPFIT_SYSTEMS = QUASI_SYSTEMS + CONVENTIONAL_SYSTEMS
TRIPFIT_NPARAMS = 15
TRIPFIT_PHASON_SLOTS = list(range(6, 15))
_TRIPFIT_QUASI_FREE = {
    'icosahedral':         [0] + TRIPFIT_PHASON_SLOTS,   # a + phason
    'icosahedral_fixed_a': list(TRIPFIT_PHASON_SLOTS),   # phason; a held
    'cubic_no_strain':     [0],                          # a; phason held at 0
}


def tripfit_free_slots(system):
    '''Indices into the 15-element tripfit parameter vector that are refined for
    ``system``.  For a conventional system these are its free lattice slots
    (``lattice_free_slots``), so its reduced vector is unchanged.'''
    if system in _TRIPFIT_QUASI_FREE:
        return list(_TRIPFIT_QUASI_FREE[system])
    return lattice_free_slots(system)


def tripfit_params(params):
    '''A 15-element float copy of a tripfit parameter vector.  A 6-element
    lattice is accepted and read with a zero phason matrix.'''
    p = np.zeros(TRIPFIT_NPARAMS, dtype=float)
    v = np.asarray(params, dtype=float).ravel()[:TRIPFIT_NPARAMS]
    p[:v.size] = v
    return p


def tripfit_expand(system, params):
    '''(lattice, phason) for a tripfit parameter vector: the symmetry-constrained
    [a,b,c,alpha,beta,gamma] and the 9-element phason matrix, row-major
    a11..a33.  The phason is zero for a conventional system and for
    cubic_no_strain, as in dmsfit_ico_hkl.imcalc.'''
    p = tripfit_params(params)
    if system in QUASI_SYSTEMS:
        a = float(p[0])
        phason = ([0.0] * 9 if system == 'cubic_no_strain'
                  else [float(m) for m in p[6:15]])
        return [a, a, a, 90.0, 90.0, 90.0], phason
    return expand_lattice(system, p[:6]), [0.0] * 9


def tripfit_reflections(reflist, system, tau=TAU_APPROX):
    '''(parallel, perpendicular) components, each 3x3, of one triple's
    reflections.  A quasicrystal mode takes 6D indices (3x6) and projects them
    exactly as the slider does (``Projection6dArrayApproximant`` with ``tau``);
    a conventional system takes Miller indices (3x3), whose perpendicular part
    is zero.  Raises ValueError on the wrong index count, which would otherwise
    only surface as the 500 failure penalty.'''
    ref = np.asarray(reflist, dtype=float)
    ncol = 6 if system in QUASI_SYSTEMS else 3
    if ref.shape != (3, ncol):
        raise ValueError('a %s triple needs a 3x%d reflection list (%s), got '
                         'shape %s' % (system, ncol,
                                       '6D indices' if ncol == 6 else 'h k l',
                                       ref.shape))
    if ncol == 3:
        return ref, np.zeros_like(ref)
    par, perp = Projection6dArrayApproximant(ref, tau).reflection_6d()
    return np.asarray(par, dtype=float), np.asarray(perp, dtype=float)


# Names of the tripfit parameter slots, as computation.locked spells them.
TRIPFIT_PARAM_NAMES = ('a', 'b', 'c', 'alpha', 'beta', 'gamma',
                       'a11', 'a12', 'a13', 'a21', 'a22', 'a23',
                       'a31', 'a32', 'a33')


def tripfit_locked_slots(names):
    '''Parameter slots named in ``names`` (computation.locked, the GUI's
    unticked slider boxes): parameters a fit holds at their starting value.
    Raises ValueError on an unknown name.'''
    if isinstance(names, str):
        names = [names]
    slots = set()
    for n in names:
        if n not in TRIPFIT_PARAM_NAMES:
            raise ValueError('computation.locked: unknown parameter %r (expected '
                             'any of %s)' % (n, ', '.join(TRIPFIT_PARAM_NAMES)))
        slots.add(TRIPFIT_PARAM_NAMES.index(n))
    return sorted(slots)


def tripfit_locked_names(slots):
    '''computation.locked names for a collection of parameter slots.'''
    return [TRIPFIT_PARAM_NAMES[s] for s in sorted(set(slots))]


def tripfit_fit_positions(system, locked=()):
    '''Positions in the reduced vector (``tripfit_free_slots`` order) that the
    optimiser moves: the free slots that are not locked.'''
    locked = set(locked)
    return [i for i, s in enumerate(tripfit_free_slots(system)) if s not in locked]


class tripfit(object):
    '''Fit a lattice by driving three Kossel lines of a secondary-reflection
    triple to a common (triple-intersection) point on the stereographic
    projection.  This is the multiple-diffraction analogue of the image fit: no
    detector image is needed, only the geometry.

    Constructor:
        tripfit(hkl, reflist, azir, resolution, bravais, energy, target,
                params=None, tau=TAU_APPROX)

    ``bravais`` is one of ``TRIPFIT_SYSTEMS``.  For a conventional system
    ``reflist`` is a 3 x 3 matrix of Miller indices and the free parameters are
    the symmetry-allowed lattice slots (``lattice_free_slots`` /
    ``expand_lattice``, shared with the image fit).  For a quasicrystal mode
    (``QUASI_SYSTEMS``) ``reflist`` holds three 6D indices (3 x 6), projected
    with ``tau``, and the free parameters are those of the image fit's mode:
    ``a`` and the phason matrix, the phason alone, or ``a`` alone
    (``tripfit_free_slots``).  ``params`` is the full 15-element parameter
    vector the reduced one is scattered into; it supplies what a mode holds
    fixed (the ``a`` of icosahedral_fixed_a) — ``set_params`` updates it.

    Each line pair may cross at more than one point; the tightest
    (mutually-closest) triple is scored, so no per-pair intercept index is
    needed.  The score is that triple's summed squared pairwise distance
    (``triple_spread``).  ``target`` is the desired residual (0 for a perfect
    triple intersection).

    The crossings are solved on the sphere, as the intersections of the cones
    the Kossel lines exactly are, so the residual does not depend on
    ``resolution`` — that only sets how finely the lines are sampled for
    plotting, and for the plane fit each cone is recovered from.  See *Kossel
    lines as the circles they analytically are* above.

    ``fit(reduced)`` returns the scalar residual for a reduced free-parameter
    vector; ``full(reduced)`` returns (intercepts, st0, st1, st2, vr0, vr1, vr2)
    for plotting.'''
    def __init__(self, hkl, reflist, azir, resolution, bravais, energy, target,
                 params=None, tau=TAU_APPROX):
        self.hkl = hkl
        self.azir = azir
        self.resolution = resolution
        self.bravais = bravais
        self.energy = energy
        self.target = target
        self.tau = tau
        # Worst deviation of any sampled Kossel point from the plane of the
        # circle it lies on, from the last _intercepts; None when the analytic
        # solve was not used and the sampled polylines were crossed instead.
        self.circle_residual = None
        self.set_params(np.zeros(TRIPFIT_NPARAMS) if params is None else params)
        if bravais in QUASI_SYSTEMS:
            self.ref_6d = np.matrix(np.asarray(reflist, dtype=float))
            self.ref_par, self.ref_perp = tripfit_reflections(reflist, bravais, tau)
            self.reflist = np.matrix(self.ref_par)
        else:
            self.reflist = np.matrix(reflist)

    def set_params(self, params):
        '''Set the full parameter vector (6-element lattice or 15-element
        lattice + phason) that the reduced optimiser vector is scattered into.'''
        self.params = tripfit_params(params)

    def reflections(self, phason):
        '''The three secondary reflections as 3D vectors (r.l.u.) under the
        phason matrix ``phason``: par + M·perp for a quasicrystal — the slider's
        PhasonDistoArray step — and the Miller indices otherwise.'''
        if self.bravais in QUASI_SYSTEMS:
            return PhasonDistoArray(self.ref_par, self.ref_perp, phason).qe1()
        return self.reflist

    def _circle_intercepts(self):
        '''Every pair crossing solved on the sphere: each locus's cone plane
        (`sphere_circle_plane`), the two cones of a pair intersected in closed
        form (`cone_pair_directions`), and the directions projected.  Returns a
        list of three (k, 2) arrays in the order (0,1), (0,2), (1,2), or None
        when a locus is not usable as a circle, which asks the caller to cross
        the sampled polylines instead.  Leaves the worst plane deviation in
        `circle_residual` — the evidence that the loci are the circles this
        assumes (None when the analytic path was not used).

        A pair that is well posed but simply does *not* meet still raises, as
        the polyline path does: that is a real answer about the geometry, not a
        reason to fall back to a cruder method.'''
        planes, worst = [], 0.0
        for vr in (self.vr0, self.vr1, self.vr2):
            try:
                n, c, resid = sphere_circle_plane(np.asarray(vr))
            except ValueError:
                self.circle_residual = None
                return None
            # Written as `not <=` so a NaN deviation falls back too, rather
            # than passing a `>` test it cannot fail.
            if not resid <= KOSSEL_CIRCLE_TOL:
                self.circle_residual = None
                return None
            worst = max(worst, resid)
            planes.append((n, c))
        P = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            try:
                v = cone_pair_directions(planes[i][0], planes[i][1],
                                         planes[j][0], planes[j][1])
            except ValueError:            # coaxial: no isolated crossing
                self.circle_residual = None
                return None
            if not len(v):
                raise ValueError('a Kossel-line pair has no intersection')
            # Projected through stereoproj itself, so the analytic crossings and
            # the plotted lines cannot end up on different conventions.
            P.append(stereoproj(v).T)
        self.circle_residual = worst
        return P

    def _polyline_intercepts(self):
        '''Every pair crossing taken from the sampled stereographic polylines —
        the fallback for a locus the circle solve will not take.  Accurate only
        to the chord sagitta, so the result depends on `resolution`.'''
        P = []
        for A, B in ((self.st0, self.st1), (self.st0, self.st2),
                     (self.st1, self.st2)):
            x, y, _ea, _eb = intersections(A, B)
            if not len(x):
                raise ValueError('a Kossel-line pair has no intersection')
            P.append(np.column_stack((x, y)))
        return P

    def _intercepts(self):
        '''The three pairwise Kossel-line intersection points (3x2, rows [x, y])
        that lie closest together — the tightest triple.  Each pair crosses at
        one or more points; picking the mutually-closest one per pair follows the
        physical triple intersection directly and continuously, with no
        dependence on the point ordering, so the selection cannot jump as the
        lattice varies.'''
        P = self._circle_intercepts()
        if P is None:
            P = self._polyline_intercepts()
        best, best_cost = None, np.inf
        for a in P[0]:
            for b in P[1]:
                for c in P[2]:
                    cost = triple_spread((a, b, c))
                    # A crossing at the projection pole projects to infinity;
                    # it is a real direction but not a usable 2D point, so it
                    # loses to any finite alternative rather than poisoning the
                    # comparison.
                    if np.isfinite(cost) and cost < best_cost:
                        best_cost, best = cost, (a, b, c)
        if best is None:
            raise ValueError('no finite triple intersection')
        return np.array(best, dtype=float)

    def _params_from_reduced(self, reduced):
        '''(lattice, phason) for a reduced free-parameter vector: scattered into
        the full parameter vector and expanded with the mode's constraints.'''
        full = self.params.copy()
        full[tripfit_free_slots(self.bravais)] = np.asarray(reduced, dtype=float).ravel()
        return tripfit_expand(self.bravais, full)

    def kosselcalc(self, inputs):
        _lattice, _phason = self._params_from_reduced(inputs)
        # One vectorised call for all three Kossel lines; row blocks of length
        # ``resolution`` come back grouped by reflection.
        r = self.resolution
        vr = kosscalc(_lattice, self.energy, self.hkl, self.reflections(_phason),
                      self.azir, 0, 360, r)[:, :3]
        vr0, vr1, vr2 = vr[:r], vr[r:2 * r], vr[2 * r:]
        self.vr0 = np.matrix(vr0 / np.array([np.apply_along_axis(np.linalg.norm, 1, vr0)]).T)
        self.vr1 = np.matrix(vr1 / np.array([np.apply_along_axis(np.linalg.norm, 1, vr1)]).T)
        self.vr2 = np.matrix(vr2 / np.array([np.apply_along_axis(np.linalg.norm, 1, vr2)]).T)
        self.st0 = stereoproj(self.vr0).T
        self.st1 = stereoproj(self.vr1).T
        self.st2 = stereoproj(self.vr2).T

    def fit(self, inputs):
        try:
            self.kosselcalc(inputs)
            pts = self._intercepts()              # tightest (closest) triple
            opt = abs(triple_spread(pts) - self.target)
        except Exception:
            opt = 500
        return opt

    def full(self, inputs):
        self.kosselcalc(inputs)
        try:
            intercepts = self._intercepts()             # (3, 2), rows [x, y]
        except Exception:
            intercepts = np.zeros((3, 2))
        return intercepts, self.st0, self.st1, self.st2, self.vr0, self.vr1, self.vr2


# ── tripfit optimiser dispatch (shared by tripfit.py and tripslider.py) ─────────
# Gradient-based methods differentiate the objective by finite differences.  The
# Kossel lines are sampled polylines, but their vertices move smoothly with the
# lattice, so the objective is smooth down to ~machine precision and SciPy's
# default step (~1.5e-8) is the best choice: measured against both example
# configs, enlarging it to 1e-5 costs ~5 orders of magnitude of final residual.
# TRIPFIT_FD_STEP = None therefore means "leave SciPy's default alone";
# computation.fd_step can still override it.
GRADIENT_METHODS = ('L-BFGS-B', 'SLSQP', 'TNC', 'BFGS', 'CG')
BOUNDED_METHODS = ('L-BFGS-B', 'SLSQP', 'TNC')
DIRECT_METHODS = ('Powell', 'Nelder-Mead', 'COBYLA')
LOCAL_METHODS = DIRECT_METHODS + GRADIENT_METHODS
TRIPFIT_METHODS = (LOCAL_METHODS + ('GA',)
                   + tuple('BH' + m for m in LOCAL_METHODS))
TRIPFIT_FD_STEP = None          # None -> use SciPy's own finite-difference step
# The GUI once offered 'BHNelderMead', which fed SciPy the invalid inner method
# 'NelderMead'; accept it so an old saved config still loads.
TRIPFIT_METHOD_ALIASES = {'BHNelderMead': 'BHNelder-Mead'}


def tripfit_method(name):
    '''Canonical tripfit method name, resolving legacy aliases.'''
    return TRIPFIT_METHOD_ALIASES.get(name, name)


def tripfit_minimizer_options(method, tol, fd_step=TRIPFIT_FD_STEP):
    '''SciPy ``options`` dict for a tripfit local method.  Only passes keys the
    method actually accepts (SciPy warns on unknown options and ignores them):
    the tolerance pair each direct-search method understands, and the optional
    finite-difference step ``eps`` for the gradient methods.'''
    if method in GRADIENT_METHODS:
        opts = {} if fd_step is None else {'eps': fd_step}
        if method in ('L-BFGS-B', 'TNC'):
            opts['ftol'] = tol
        return opts
    if method == 'Powell':
        return {'xtol': tol, 'ftol': tol}
    if method == 'Nelder-Mead':
        return {'xatol': tol, 'fatol': tol}     # NOT xtol/ftol
    return {}                                   # COBYLA takes neither


def run_tripfit_optimiser(objective, x0, method, bounds, tol,
                          niter=100, strat='best1bin', fd_step=TRIPFIT_FD_STEP):
    '''Run one tripfit optimisation and return the SciPy result.

    ``method`` is any name in ``TRIPFIT_METHODS``: a local method (direct-search
    or gradient-based), ``GA`` (differential evolution), or ``BH<local>`` (basin
    hopping around that local method).  Bounds are passed only to the methods
    that support them, and each method receives only the SciPy options it
    understands.'''
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    method = tripfit_method(method)
    if method == 'GA':
        return differential_evolution(objective, bounds, strategy=strat,
                                      polish=True)
    if method.startswith('BH'):
        inner = method[2:] or 'Powell'
        kwargs = {'method': inner,
                  'options': tripfit_minimizer_options(inner, tol, fd_step)}
        if inner in BOUNDED_METHODS:
            kwargs['bounds'] = bounds
        return basinhopping(objective, x0, minimizer_kwargs=kwargs, niter=niter)
    kwargs = {'method': method, 'tol': tol,
              'options': tripfit_minimizer_options(method, tol, fd_step)}
    if method in BOUNDED_METHODS:
        kwargs['bounds'] = bounds
    return minimize(objective, x0, **kwargs)
