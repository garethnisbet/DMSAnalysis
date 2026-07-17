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
#import iotbx.cif
#from cctbx import sgtbx
#from cctbx.sgtbx import space_group, space_group_symbols
from scipy import ndimage
from scipy import interpolate
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


def loadcif(ciffile,energy):
    wavelength=12.3984187/energy
    d_min_val = (wavelength/(2*np.sin(np.radians(180)/2)))
    cf = iotbx.cif.reader(file_path=ciffile).model()
    cif_block = cf.values()[0]
    c = iotbx.cif.reader(file_path = ciffile).build_crystal_structures().values()[0]
    s = c.scatterers()
    for _s in s:
        _s.scattering_type = _s.scattering_type.replace('0+', '')
    sg=c.crystal_symmetry().space_group()
    c=c.expand_to_p1(append_number_to_labels=False, sites_mod_positive=True)
    lattice=list(c.crystal_symmetry().unit_cell().parameters())
    sf = c.structure_factors(True,algorithm='direct',d_min=wavelength/2).f_calc()
    sfc = list(sf.data())
    reflist = list(sf.indices())
    SF=abs(np.array(sfc))**2
    return SF, np.array(reflist), lattice, c, np.array(sfc)

def dms2px(detv1,detv2,o,v):
    ''' usage dms2px(detector vector 1,detector vector2, sample origin as vector, vectors which will be scaled to intersect detector'''
    v=np.array(v)
    n=np.cross(detv1,detv2)
    D=-(n[0]*detv1[0]+n[1]*detv1[1]+n[2]*detv1[2])# scalar equation for plane
    t=-(n[0]*o[0]+n[1]*o[1]+n[2]*o[2]+D)/(n[0]*v[:,0]+n[1]*v[:,1]+n[2]*v[:,2])# scalar from parametric representation of vectors 
    return (t*v.T).T+o # returns intersection coordinates

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
    kernelstack=np.zeros((imshape[0],imshape[1],numrefs*2))
    keep=np.array([[]]*1).T
    for i1 in range(0,numrefs,1):
        ref=reflist[i1,:]
        emptyim=np.zeros(imshape)
        dmsroi = dmscalc_ico_hkl([ref],hkllist,hklint,1,psirange,100,hkl,detvects,emptyim,simsigma,azir,psi,px,py,scatv,detdistancepx,rotx,roty,rotz,energy,reflist2,mtrx2)
        dmsroi.crystal_system = crystal_system

        roiindex=dmsroi.roiindex(ig)

        roi1=np.zeros(imshape)
        roi2=np.zeros(imshape)
        if roiindex.size>8:
#         roiindex=np.array([np.where(roi>0)[0],np.where(roi>0)[1]]).T
            # Choose sorting direction
            if abs(roiindex[:,0].min()-roiindex[:,0].max())>abs(roiindex[:,1].min()-roiindex[:,1].max()):
                roiindex=roiindex[np.argsort(roiindex[:,0],0),:]
            else:
                roiindex=roiindex[np.argsort(roiindex[:,1],0),:]
            x1=roiindex[:int(len(roiindex)/2),0]
            y1=roiindex[:int(len(roiindex)/2),1]
            x2=roiindex[int(len(roiindex)/2):,0]
            y2=roiindex[int(len(roiindex)/2):,1]
    #         if np.mean(np.gradient(roiindex[0]))>np.mean(np.gradient(roiindex[1])):
            if abs(x1.min()-x1.max())>abs(y1.min()-y1.max()):
                f = interpolate.interp1d(x1, y1)
                x1interp=range(x1.min()+1,x1.max()-1)
                y1interp=f(x1interp).astype(int)
                f2 = interpolate.interp1d(x2, y2)
                x2interp=range(x2.min()+1,x2.max()-1)
                y2interp=f2(x2interp).astype(int)
                roi1[x1interp,y1interp]=1
                roi2[x2interp,y2interp]=1
                kernelstack[:,:,(i1*2)]=roi1
                kernelstack[:,:,(i1*2)+1]=roi2
            else:
                f = interpolate.interp1d(y1, x1)
                y1interp=range(y1.min()+1,y1.max()-1)
                x1interp=f(y1interp).astype(int)
                f2 = interpolate.interp1d(y2, x2)
                y2interp=range(y2.min()+1,y2.max()-1)
                x2interp=f2(y2interp).astype(int)
                roi1[x1interp,y1interp]=1
                roi2[x2interp,y2interp]=1
                kernelstack[:,:,(i1*2)]=roi1
                kernelstack[:,:,(i1*2)+1]=roi2
            keep=np.vstack([keep,(i1*2)])
            keep=np.vstack([keep,(i1*2)+1])
        else:
            print('ROI '+str(i1)+' removed because lines miss the detector.')
#     keep=uniquearray(keep) # clean duplicates
    if keep.shape[0] >0:
        keep=tuple(map(tuple,keep.T.astype(int)))[0]
        return kernelstack[:,:,keep]
    else:
        print('No ROIS used!')
        return kernelstack+1
def msroi(img, kernel, width):
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

def multiroifit(img,kernel,width,percentileval,method='gauss'):
    v1=np.array([[]]*4).T
    vx=[]
    vy=[]
    v3=[]
    pcovlist=[]
    v4=np.zeros((img.shape[0],img.shape[1],kernel.shape[2]))
    for i1 in range(kernel.shape[2]):
        sumvals,roi = msroi(img,kernel[:,:,i1],width)
        xdata=np.arange(len(sumvals))
        ydata=sumvals[:,0]
        # xdata=xdata[ydata>np.percentile(ydata, percentileval)]
        # ydata=ydata[ydata>np.percentile(ydata, percentileval)]
#         ydata[ydata<np.percentile(ydata, 10)]=np.percentile(ydata, 10)
        try:
            coef, pcov,fitpoints = peakfit(xdata,ydata,method)
        except:
            print('Fit not possible for _'+str(i1))
            coef = np.array([0,0,500,0])
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
    except:
        print('Fit not possible for _' + str(idx))
        coef = np.array([0, 0, 500, 0])
        pcov = np.zeros((4, 4))
        fitpoints = ydata
    return coef, xdata, ydata, fitpoints, roi, pcov

def multiroifit2(img,kernel,width,percentileval,sig,method='gauss'):
    n = kernel.shape[2]
    results = Parallel(n_jobs=-1)(
        delayed(_multiroifit2_one)(img, kernel[:, :, i1], width, sig, i1, method)
        for i1 in range(n)
    )
    v4 = np.zeros((img.shape[0], img.shape[1], n))
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
        if self.scatv ==1:
            pxv2d[:,0]=self.px+pxv2d[:,0]
            pxv2d[:,1]=self.py+pxv2d[:,1]
        else:
            pxv2d[:,0]=self.px+pxv2d[:,0]
            pxv2d[:,1]=self.py-pxv2d[:,1]
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
        sin_arg = np.clip(
            (ko*np.sin(-b1*np.pi/180) + realvecthkl[:,:,2]) / ko, -1.0, 1.0
        )
        rewl     = ko * np.cos(np.arcsin(sin_arg))
        rhk      = np.sqrt(realvecthkl[:,:,0]**2 + realvecthkl[:,:,1]**2)
        rhkangle = np.arctan2(realvecthkl[:,:,0], realvecthkl[:,:,1]) * 180/np.pi

        numer  = orighk**2 - rhk**2 + rewl**2
        half_n = numer / (2*orighk)
        xint   = np.sqrt(np.maximum(rewl**2 - half_n**2, 0))

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
        vecs1=psith2v(self.psi-mslist[:,3]-psicorrection,mslist[:,5]+thetacorrection)
        vecs2=psith2v(self.psi-mslist[:,4]-psicorrection,mslist[:,5]+thetacorrection)
        vecs=np.concatenate((vecs1,vecs2),0)
        centralv=-psith2v(0,thb)*detdistancepx
        prepxvec=dms2px(detvs[0,:],detvs[1,:],centralv,vecs)
        pxvec=np.array(np.round(prepxvec*irmat).astype(int)) #build reverse matrix for detector
        imsim=np.zeros(np.shape(self.imdata))
        self.vecs = vecs
        #########  Shift vectors to non negative coordinates   ######################
        pxv2d=np.array(pxvec[:,[0,2]])
        if self.scatv ==1:
            pxv2d[:,0]=self.px+pxv2d[:,0]
            pxv2d[:,1]=self.py+pxv2d[:,1]
        else:
            pxv2d[:,0]=self.px+pxv2d[:,0]
            pxv2d[:,1]=self.py-pxv2d[:,1]
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
def _fit_roi_gauss(imsim, kernel_slice, width):
    sumvals, roi = msroi(imsim, kernel_slice, width)
    xdata = np.arange(len(sumvals))
    ydata = sumvals[:, 0]
    try:
        coef, pcov, fitpoints = fitgauss(xdata, ydata)
        return coef
    except:
        return np.array([100, 100, 100, 100])

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
        # or 'centroid' (centre of mass).  Set via setPeakMethod.
        self.peakmethod = 'gauss'
        # Full 24-element guess vector — used by the conventional-crystal branch
        # of imcalc to fill the non-refined parameters around the reduced
        # optimiser vector.  Defaults to args[26] (the lattice 'a') in slot 0.
        self.ig_full = None
    def setCalLattice(self, cal_lattice):
        self.calibration_lattice = cal_lattice
    def setIGFull(self, ig24):
        '''Store the full 24-element guess vector (conventional crystals only).
        The optimiser passes a reduced subset to imcalc; the remaining slots
        (constrained lattice params, detector/energy when not refined, and the
        always-zero phason block) are read back from this template.'''
        self.ig_full = np.asarray(ig24, dtype=float).copy()
    def setLattice(self, lattice):
        self.lattice = lattice
    def setPeakMethod(self, method):
        self.peakmethod = method
    def _simcoeffs(self):
        '''Per-ROI peak coefficients of the current simulated image, using the
        selected peak-position method.  v1[:,2] is the centre per ROI.'''
        v1=np.array([[]]*4).T
        for i1 in range(self.kernel.shape[2]):
            sumvals,roi = msroi(self.imsim,self.kernel[:,:,i1],self.width)
            xdata=np.arange(len(sumvals))
            ydata=sumvals[:,0]
            try:
                coef, pcov,fitpoints = peakfit(xdata,ydata,self.peakmethod)
                v1=np.vstack([v1,coef])
            except:
                v1=np.vstack([v1,[100,100,100,100]])
        return v1
        
###########################
    def imcalc(self,*inputs):
        inputs=inputs[0]
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
        prepxvec=dms2px(detvs[0,:],detvs[1,:],centralv,vecs)
        valid_px = ~np.isnan(prepxvec).any(axis=1)
        pxvec=np.array(np.round(prepxvec[valid_px]*irmat).astype(int))
        imsim=np.zeros(np.shape(self.imdata))
        pxv2d=np.array(pxvec[:,[0,2]])
        self.bragg = thb
        if self.scatv ==1:
            pxv2d[:,0]=self.px+pxv2d[:,0]
            pxv2d[:,1]=self.py+pxv2d[:,1]
        else:
            pxv2d[:,0]=self.px+pxv2d[:,0]
            pxv2d[:,1]=self.py-pxv2d[:,1]
        pxv2d=pxv2d[np.where(pxv2d[:,0]>-1)]
        pxv2d=pxv2d[np.where(pxv2d[:,0]< imsim.shape[0])]
        pxv2d=pxv2d[np.where(pxv2d[:,1]>-1)]
        pxv2d=pxv2d[np.where(pxv2d[:,1]< imsim.shape[1])]
        self.dmsindex=tuple([pxv2d[:,0],pxv2d[:,1]])
        self.inputarray = np.array([a,b,c,alpha,beta,gamma,psicorrection,h_correction,k_correction,l_correction,detdistancepx,detxrot,detyrot,detzrot,energy,self.a11,self.a12,self.a13,self.a21,self.a22,self.a23,self.a31,self.a32,self.a33])
        imsim[self.dmsindex]=1
        if self.simsigma != 0:
            self.imsim=ndimage.convolve(imsim,makekernel('gauss',15,self.simsigma))
        else:
            self.imsim=imsim

        # ── Per-reflection line data for visualisation ────────────────────────
        if N_steps > 0 and N_refs > 0:
            n_half = N_steps * N_refs + 1
            def _prepx_lines(pp_half):
                px = np.asarray(np.round(pp_half[1:] * irmat), dtype=float)  # (N_steps*N_refs, 3)
                px = px.reshape(N_steps, N_refs, 3)
                r = float(self.px) + px[:,:,0]
                c = float(self.py) + (px[:,:,2] if self.scatv == 1 else -px[:,:,2])
                h_im, w_im = imsim.shape
                oob = (r < 0) | (r >= h_im) | (c < 0) | (c >= w_im) | np.isnan(r) | np.isnan(c)
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

    def fit(self,inputs):
        try:
            self.imcalc(inputs) # adding attribute
            v1=self._simcoeffs()
            result = np.sum((v1[:,2]-self.centres[:,0])**2)
            return result
        except:
            return 500

    def residuals(self,inputs):
        """Per-ROI centre residuals (fitted Gaussian centre - target centre).
        This is the vector form of `fit` (which returns np.sum(residuals**2)),
        for use with scipy.optimize.least_squares.  A robust loss (soft_l1 /
        huber) then downweights the [100,...] fallback rows produced when a
        per-ROI Gaussian fit fails."""
        try:
            self.imcalc(inputs) # adding attribute
            v1=self._simcoeffs()
            return v1[:,2]-self.centres[:,0]
        except:
            return np.full(self.centres.shape[0], 100.0)

    def stats(self,inputs):
        self.imcalc(inputs) # adding attribute
        v1=self._simcoeffs()
        pcovarray=np.array(self.pcov)
        return np.abs((v1[:,2]-self.coefs[:,2])),pcovarray[:,2,2]

    def full(self,inputs):
        try:
            self.imcalc(inputs) # adding attribute
            v1=self._simcoeffs()
            result = np.sum((v1[:,2]-self.centres[:,0])**2)
            return result,self.imsim, self.dmsindex, self.imdata, self.inputarray
        except:
            return 500,np.zeros(self.imdata.shape), np.array([[],[]]), self.imdata, self.inputarray


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
    returned as a 2 x N array of [x, y].'''
    return np.concatenate((vin[:, 0] / (1. - vin[:, 2]),
                           vin[:, 1] / (1. - vin[:, 2])), 1).T


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


class tripfit(object):
    '''Fit a conventional lattice by driving three Kossel lines of a secondary-
    reflection triple to a common (triple-intersection) point on the stereographic
    projection.  This is the multiple-diffraction analogue of the image fit: no
    detector image is needed, only the geometry.

    Constructor:
        tripfit(hkl, reflist, azir, resolution, bravais, energy, target)

    ``reflist`` is a 3 x 3 matrix of the three secondary reflections; ``bravais``
    is one of ``CONVENTIONAL_SYSTEMS`` and selects which lattice parameters are
    free (via ``lattice_free_slots`` / ``expand_lattice``, shared with the image
    fit so the parameter packing cannot drift).  Each line pair may cross at more
    than one point; the tightest (mutually-closest) triple is scored, so no
    per-pair intercept index is needed.  The score is that triple's summed
    squared pairwise distance (``triple_spread``).  ``target`` is the desired
    residual (0 for a perfect triple intersection).

    ``fit(reduced)`` returns the scalar residual for a reduced free-parameter
    vector; ``full(reduced)`` returns (intercepts, st0, st1, st2, vr0, vr1, vr2)
    for plotting.'''
    def __init__(self, hkl, reflist, azir, resolution, bravais, energy, target):
        self.hkl = hkl
        self.reflist = np.matrix(reflist)
        self.azir = azir
        self.resolution = resolution
        self.bravais = bravais
        self.energy = energy
        self.target = target

    def _intercepts(self):
        '''The three pairwise Kossel-line intersection points (3x2, rows [x, y])
        that lie closest together — the tightest triple.  Each pair crosses at
        one or more points; picking the mutually-closest one per pair follows the
        physical triple intersection directly and continuously, with no
        dependence on shapely's (geometry-dependent) point ordering, so the
        selection cannot jump as the lattice varies.'''
        P = []
        for A, B in ((self.st0, self.st1), (self.st0, self.st2),
                     (self.st1, self.st2)):
            x, y, _ea, _eb = intersections(A, B)
            if not len(x):
                raise ValueError('a Kossel-line pair has no intersection')
            P.append(np.column_stack((x, y)))
        best, best_cost = None, np.inf
        for a in P[0]:
            for b in P[1]:
                for c in P[2]:
                    cost = triple_spread((a, b, c))
                    if cost < best_cost:
                        best_cost, best = cost, (a, b, c)
        return np.array(best, dtype=float)

    def _lattice_from_reduced(self, reduced):
        '''Expand a reduced free-parameter vector into the full constrained
        lattice [a,b,c,alpha,beta,gamma] for this crystal system.'''
        reduced = np.asarray(reduced, dtype=float).ravel()
        six = np.zeros(6, dtype=float)
        for slot, val in zip(lattice_free_slots(self.bravais), reduced):
            six[slot] = val
        return expand_lattice(self.bravais, six)

    def kosselcalc(self, inputs):
        _lattice = self._lattice_from_reduced(inputs)
        # One vectorised call for all three Kossel lines; row blocks of length
        # ``resolution`` come back grouped by reflection.
        r = self.resolution
        vr = kosscalc(_lattice, self.energy, self.hkl, self.reflist,
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
