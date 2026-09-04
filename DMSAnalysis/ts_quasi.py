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
    kernelstack=np.zeros((imshape[0],imshape[1],numrefs*2))
    keep=np.array([[]]*1).T
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
                kernelstack[:,:,(i1*2)]=roi1
                kernelstack[:,:,(i1*2)+1]=roi2
                keep=np.vstack([keep,(i1*2)])
                keep=np.vstack([keep,(i1*2)+1])
            else:
                print('ROI '+str(i1)+' removed because a half is off the detector.')
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


def roi_outline(kernel, width):
    '''Closed (rows, cols) outline of the strip `msroi` integrates for one ROI.

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


def multiroifit(img,kernel,width,percentileval,method='gauss',sig=None):
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
    vecs = np.asarray(vecs, dtype=float)
    half = n_steps * n_refs + 1
    empty = (np.empty((0, 3)), np.empty(0, int), np.empty(0, int))
    if n_steps < 1 or n_refs < 1 or vecs.shape[0] < 2 * half:
        return empty + (0.0, [])
    seg = None if segment is None else np.asarray(segment).ravel()

    owner, pts_all, fallback = [], [], []
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
        return empty + (0.0, fallback)

    P     = np.concatenate(pts_all)
    lens  = np.array([len(p) for p in pts_all])
    offs  = np.concatenate([[0], np.cumsum(lens)[:-1]])
    runs  = [np.arange(o, o + l) for o, l in zip(offs, lens)]
    owner = np.asarray(owner, dtype=int)

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
        for i1 in range(self.kernel.shape[2]):
            sumvals,roi = msroi(self.imsim,self.kernel[:,:,i1],self.width)
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

        def _to_pixels(v):
            '''Exit-direction vectors -> float (row, col) detector pixels.  The
            one map both curve methods project through, so an analytic curve and
            the sampled one it was fitted to cannot land a pixel apart.'''
            pre = dms2px(detvs[0,:], detvs[1,:], centralv, np.asarray(v, dtype=float))
            pv  = np.asarray(np.round(pre * irmat), dtype=float)
            return (float(self.px) + pv[:,0],
                    float(self.py) + (pv[:,2] if self.scatv == 1 else -pv[:,2]))

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
        imsim[self.dmsindex]=1
        if self.simsigma != 0:
            self.imsim=ndimage.convolve(imsim,makekernel('gauss',15,self.simsigma))
        else:
            self.imsim=imsim

        # ── Per-reflection line data for visualisation ────────────────────────
        if arcs is not None:
            self.dmslines = self._arc_dmslines(arcs, N_refs, _to_pixels,
                                               imsim.shape)
        elif N_steps > 0 and N_refs > 0:
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
        oob = ((rows < 0) | (rows >= h_im) | (cols < 0) | (cols >= w_im)
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
        try:
            resid, _, _ = self._centre_residuals(inputs)
            return float(np.sum(resid**2))
        except Exception:
            return self._total_failure_score()

    def residuals(self,inputs):
        """Per-ROI centre residual vector, for scipy.optimize.least_squares.
        `fit` is exactly np.sum(residuals**2).  A robust loss (soft_l1 / huber)
        then downweights the penalty rows produced by failed ROIs."""
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
