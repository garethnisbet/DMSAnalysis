"""Ridge + cone fit: fit the DMS geometry to where the lines actually are.

The ROI fit (`fit.py`, the slider's **Fit**) integrates a strip laid along the
*predicted* line, collapses each half-line to one profile and scores one centre
per half against the same strip drawn through a simulated image.  That keeps
two numbers per reflection — an offset and, from the two halves, a tilt — and
throws the rest of the line's shape away.

This prototype measures the line instead, at many points along it, and fits the
geometry to those points directly:

1. **Ridge extraction** (once per pass, from the experimental image).  Walk
   every predicted arc at a fixed pixel spacing; at each stop sample a short
   profile along the arc's *local* normal (bilinear interpolation, averaged over
   a short segment along the line), fit a Gaussian on a linear background and
   keep the peak position and its standard error.  Stops near another
   reflection's line (a crossing), off the plate, or whose peak is weak, too
   wide or outside the window are dropped.
2. **Cone fit** (no image rendering).  Every DMS line is exactly a cone of exit
   directions, a circle on the unit sphere `{v : v.n = d}`.  Each evaluation
   asks the engine for its exit directions (`dmsfit_ico_hkl._geometry`), fits
   each run's circle (`dms_fit_arcs`, exact to ~1e-15), back-projects every
   ridge point through the *current* detector parameters to a direction, takes
   the nearest point on its own reflection's cone and projects that back to the
   plate.  The residual is the signed pixel distance between the two, divided
   by the point's standard error — a chi^2 over every measured point.

It is smooth in the parameters (no rounding, no tessellation, no peak fit in the
loop), so it runs under `scipy.optimize.least_squares`, and its Jacobian gives
parameter uncertainties.

    python -m DMSAnalysis.ridgefit config.json [--passes 2] [--compare] [--save]

`config.json` is the batch-fit schema `fit.py` reads (what the slider's
**Export workflow JSON** writes).
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from . import ts_quasi as ts
from . import dat2config

PKGDIR = os.path.abspath(os.path.dirname(__file__))

IG_SLOT_NAMES = [
    'a', 'b', 'c', 'alpha', 'beta', 'gamma',
    'psicor', 'chicor', 'thcor', 'lcor',
    'detdist', 'dxrot', 'dyrot', 'dzrot', 'energy',
    'phason a11', 'phason a12', 'phason a13',
    'phason a21', 'phason a22', 'phason a23',
    'phason a31', 'phason a32', 'phason a33',
]


# ── The cone model ─────────────────────────────────────────────────────────────

class ConeModel(object):
    """The engine's DMS cones and its detector map, for one parameter vector.

    Wraps a `dmsfit_ico_hkl` (any mode); only its geometry is used, never its
    simulated image or ROI kernels."""

    def __init__(self, engine, tol=ts.DMS_CIRCLE_TOL):
        self.engine = engine
        self.tol = tol

    def circles(self, x):
        """Geometry and circles at `x`.  Returns (g, C) where C holds, per
        accepted run: owner (reflection index), centre c, radius r, normal n
        (as fitted — the frame the arc angles a0/a1 are in), the normal
        oriented so that duplicates of one cone agree (`nhat`, which fixes the
        residual's sign), and the cone's half-angle cosine `d = c.nhat`."""
        g = self.engine._geometry(np.asarray(x, dtype=float))
        seg = (np.asarray(g.hkllist, dtype=float)
               @ np.asarray(g.hkl, dtype=float)) >= 0
        P, runs, owner, _ = ts.dms_circle_runs(g.vecs, g.N_steps, g.N_refs, seg)
        if not runs:
            return g, None
        c, r, n, a0, a1, resid = ts.dms_fit_arcs(P, runs)
        good = (np.isfinite(r) & (r > 0) & np.isfinite(resid)
                & (resid <= self.tol))
        # Orient each normal so it is a property of the geometric circle, not of
        # the run it was fitted from: the two psi branches traverse the same
        # cone in opposite directions, and a ridge point that ties between them
        # must get the same sign from either.  c.n >= 0 does that (and gives an
        # antipodal circle the antipodal normal).  A near-great circle has no
        # usable c.n, so it is oriented by its largest normal component instead.
        cn = np.einsum('ij,ij->i', c, n)
        big = n[np.arange(len(n)), np.argmax(np.abs(n), axis=1)]
        sgn = np.where(np.abs(cn) > 1e-6, np.sign(cn), np.sign(big))[:, None]
        nhat = n * sgn
        C = dict(owner=owner[good], c=c[good], r=r[good], n=n[good],
                 nhat=nhat[good], d=np.einsum('ij,ij->i', c, nhat)[good],
                 a0=a0[good], a1=a1[good], resid=resid[good])
        return g, C

    def project(self, g, V):
        """Exit directions (N,3) -> float detector (rows, cols), unrounded.
        The same map as `imcalc`'s `_to_pixels`."""
        e = self.engine
        pre = ts.dms2px(g.detvs[0, :], g.detvs[1, :], g.centralv,
                        np.asarray(V, dtype=float))
        pv = np.asarray(pre @ np.asarray(g.irmat), dtype=float)
        return ts.det_to_pixels(pv[:, 0], pv[:, 2], float(e.px), float(e.py),
                                e.scatv)

    def backproject(self, g, rows, cols):
        """Detector (rows, cols) -> unit exit directions; inverse of `project`.
        The plate is the plane through the origin spanned by the detector
        vectors, which the rotation `irmat` takes to pv[1] = 0."""
        e = self.engine
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)
        pv = np.zeros((len(rows), 3))
        pv[:, 0], pv[:, 2] = ts.pixels_to_det(rows, cols, float(e.px),
                                              float(e.py), e.scatv)
        pre = pv @ np.linalg.inv(np.asarray(g.irmat))
        u = pre - np.asarray(g.centralv, dtype=float).ravel()
        return u / np.linalg.norm(u, axis=1, keepdims=True)

    def predicted_lines(self, x, shape, spacing=0.5):
        """Every accepted arc at `x` as an unrounded pixel polyline.
        Returns a list of (owner, rows, cols), NaN where off the plate."""
        g, C = self.circles(x)
        if C is None:
            return []
        out = []
        for k in range(len(C['owner'])):
            sl = lambda a: a[k:k + 1]
            # Size the tessellation from a coarse pass over the same arc.
            pts, _ = ts.dms_arc_points(sl(C['c']), sl(C['r']), sl(C['n']),
                                       sl(C['a0']), sl(C['a1']), np.array([128]))
            rr, cc = self.project(g, pts)
            length = np.nansum(np.hypot(np.diff(rr), np.diff(cc)))
            m = int(np.clip(length / spacing, 64, 20000)) if np.isfinite(length) else 64
            pts, _ = ts.dms_arc_points(sl(C['c']), sl(C['r']), sl(C['n']),
                                       sl(C['a0']), sl(C['a1']), np.array([m]))
            rr, cc = self.project(g, pts)
            off = ((rr < 0) | (rr > shape[0] - 1) | (cc < 0) | (cc > shape[1] - 1)
                   | ~np.isfinite(rr) | ~np.isfinite(cc))
            rr[off] = np.nan
            cc[off] = np.nan
            out.append((int(C['owner'][k]), rr, cc))
        return out


# ── Ridge extraction ───────────────────────────────────────────────────────────

def _gauss_lin(d, A, mu, s, b0, b1):
    return A * np.exp(-0.5 * ((d - mu) / s) ** 2) + b0 + b1 * d


def _lm_profiles(d, Yz, W, P, lo, hi, n_iter):
    """Batched Levenberg-Marquardt for `fit_profiles`, from starts `P` (N,5).
    Returns (P, cost, J) at the end."""
    N = len(P)

    def model_jac(P):
        A, mu, sg = P[:, 0:1], P[:, 1:2], P[:, 2:3]
        u = (d[None, :] - mu) / sg
        g = np.exp(-0.5 * u * u)
        f = A * g + P[:, 3:4] + P[:, 4:5] * d[None, :]
        J = np.stack([g, A * g * u / sg, A * g * u * u / sg,
                      np.ones_like(g), np.broadcast_to(d, g.shape)], axis=2)
        return f, J

    P = P.copy()
    lam = np.full(N, 1e-2)
    f, J = model_jac(P)
    cost = np.sum(W * (Yz - f) ** 2, axis=1)
    for _ in range(n_iter):
        r = W * (Yz - f)
        Jw = J * W[:, :, None]
        H = np.einsum('ndi,ndj->nij', Jw, J)
        gvec = np.einsum('ndi,nd->ni', Jw, r)
        Hd = H + lam[:, None, None] * (np.eye(5)[None] * np.maximum(
            np.diagonal(H, axis1=1, axis2=2)[:, :, None], 1e-12))
        step = np.zeros_like(P)
        okd = np.abs(np.linalg.det(Hd)) > 0
        if okd.any():
            step[okd] = np.linalg.solve(Hd[okd], gvec[okd][:, :, None])[:, :, 0]
        Pn = np.clip(P + step, lo, hi)
        fn, Jn = model_jac(Pn)
        cn = np.sum(W * (Yz - fn) ** 2, axis=1)
        better = np.isfinite(cn) & (cn < cost)
        P[better], f[better], J[better], cost[better] = (
            Pn[better], fn[better], Jn[better], cn[better])
        lam = np.where(better, lam * 0.3, lam * 10.0)
        if np.all(lam > 1e8):
            break
    return P, cost, J


def fit_profiles(d, Y, max_sigma, n_iter=100):
    """Gaussian on a linear background, fitted to many profiles at once.

    `d` is the shared offset grid (D,), `Y` the profiles (N, D), with NaN for
    bins that were not sampled (a module gap) — those carry no weight.  One
    batched Levenberg-Marquardt over all N, with an analytic Jacobian, instead
    of N separate `curve_fit` calls (which cost ~5 ms each: a few seconds per
    build on a full detector).  Bounds as before: A >= 0, |mu| <= max|d|,
    0.3 <= s <= max_sigma, enforced by clipping after each step.

    Two starts per profile — the smoothed maximum with a width from its
    half-maximum, and the same centre with a wide width — keeping whichever
    ends lower: from a single narrow start ~2% of profiles settled on a noise
    spike beside the peak where curve_fit found the peak.

    Returns arrays (mu, mu_err, A, s, snr, ok).  mu_err is curve_fit's
    (absolute_sigma=False) standard error: the residual variance times the
    (J^T J)^-1 diagonal."""
    d = np.asarray(d, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N, D = Y.shape
    W = np.isfinite(Y).astype(float)
    Yz = np.where(W > 0, Y, 0.0)
    hw = float(np.abs(d).max())
    with np.errstate(invalid='ignore'):
        b0 = np.nanmedian(Y, axis=1)
    b0 = np.where(np.isfinite(b0), b0, 0.0)
    Ys = ndimage.uniform_filter1d(np.where(W > 0, Y, b0[:, None]), 3, axis=1)
    i = np.argmax(Ys, axis=1)
    A0 = Ys[np.arange(N), i] - b0
    # Width from the number of bins above half maximum (FWHM = 2.355 s).
    above = (Ys - b0[:, None]) > 0.5 * A0[:, None]
    s0 = np.clip(above.sum(axis=1) * (d[1] - d[0]) / 2.355, 0.5, max_sigma)
    lo = np.array([0.0, -hw, 0.3, -np.inf, -np.inf])
    hi = np.array([np.inf, hw, max_sigma, np.inf, np.inf])
    best = None
    for sg in (s0, np.full(N, min(max_sigma, 0.5 * max_sigma + 1.0))):
        P0 = np.stack([np.maximum(A0, 0), d[i], sg, b0, np.zeros(N)], axis=1)
        P, cost, J = _lm_profiles(d, Yz, W, P0, lo, hi, n_iter)
        if best is None:
            best = [P, cost, J]
        else:
            b = cost < best[1]
            best[0][b], best[1][b], best[2][b] = P[b], cost[b], J[b]
    P, cost, J = best

    n_eff = W.sum(axis=1)
    var = cost / np.maximum(n_eff - 5, 1)
    Jw = J * W[:, :, None]
    H = np.einsum('ndi,ndj->nij', Jw, J)
    mu_err = np.full(N, np.inf)
    good = np.isfinite(H).all(axis=(1, 2))
    if good.any():
        Hi = np.linalg.pinv(H[good])
        mu_err[good] = np.sqrt(np.clip(Hi[:, 1, 1] * var[good], 0, None))
    noise = np.sqrt(cost / np.maximum(n_eff, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(noise > 0, P[:, 0] / noise, np.inf)
    ok = np.isfinite(P).all(axis=1) & np.isfinite(mu_err) & (P[:, 0] > 0)
    return P[:, 1], mu_err, P[:, 0], P[:, 2], snr, ok


def fit_profile(d, y, max_sigma):
    """One profile through `fit_profiles`.  Returns (mu, mu_err, A, s, snr) or
    None."""
    mu, err, A, s, snr, ok = fit_profiles(d, np.asarray(y, float)[None, :],
                                          max_sigma)
    if not ok[0]:
        return None
    return float(mu[0]), float(err[0]), float(A[0]), float(s[0]), float(snr[0])


def extract_ridge(img, lines, width, step=8.0, seg_len=None, cross_px=None,
                  min_snr=4.0, max_sigma=None, edge=4.0, mask_below=0.0,
                  jump_px=3.0, jump_nsig=4.0, jump_half=5):
    """Measure the observed line position along each predicted arc.

    `lines` is `ConeModel.predicted_lines`.  `width` is the full profile width
    in pixels (the ROI width).  Returns a dict of arrays, one entry per kept
    point: owner, row, col (the observed peak), err (its standard error, px),
    offset (observed - predicted, along the normal), sigma, amp, snr, plus
    `rejected`, a count per reason.

    Pixels below `mask_below` (the detector's module gaps read -1) are masked,
    not sampled.  Along each arc piece a point is dropped if its offset jumps
    more than max(`jump_px`, `jump_nsig` x its error) from the running median
    of its `jump_half` neighbours either side: a real line moves smoothly, so
    an isolated jump is the profile window catching a neighbouring line.

    Batched: the stops are collected first, then every profile is sampled in
    one `map_coordinates` call and fitted in one `fit_profiles` call."""
    img = np.array(img, dtype=float)
    if mask_below is not None:
        img[img < mask_below] = np.nan
    hw = width / 2.0
    seg_len = step if seg_len is None else seg_len
    cross_px = hw if cross_px is None else cross_px
    max_sigma = hw / 2.0 if max_sigma is None else max_sigma
    dgrid = np.arange(-hw, hw + 1e-9, 1.0)
    agrid = np.arange(-seg_len / 2.0, seg_len / 2.0 + 1e-9, 1.0)
    rej = dict(crossing=0, offplate=0, nofit=0, weak=0, wide=0, edge=0, jump=0)
    keys = ('owner', 'row', 'col', 'err', 'offset', 'sigma', 'amp', 'snr',
            'prow', 'pcol', 'piece')

    # Every line's points, for spotting crossings with other reflections.
    allp, allo = [], []
    for o, rr, cc in lines:
        ok = np.isfinite(rr)
        allp.append(np.stack([rr[ok], cc[ok]], 1))
        allo.append(np.full(ok.sum(), o))
    tree = cKDTree(np.concatenate(allp)) if allp and sum(map(len, allp)) else None
    allo = np.concatenate(allo) if allo else np.empty(0, int)

    # 1. The stops: position, tangent and normal along every on-plate piece.
    P0, T, NR, OW, PC = [], [], [], [], []
    npiece = 0
    for o, rr, cc in lines:
        ok = np.isfinite(rr)
        if ok.sum() < 3:
            continue
        idx = np.flatnonzero(ok)
        for pc in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
            if len(pc) < 3:
                continue
            npiece += 1
            r, c = rr[pc], cc[pc]
            sarc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(r), np.diff(c)))])
            if sarc[-1] < step:
                continue
            sk = np.arange(step / 2.0, sarc[-1] - step / 2.0 + 1e-9, step)
            p0 = np.stack([np.interp(sk, sarc, r), np.interp(sk, sarc, c)], 1)
            t = (np.stack([np.interp(sk + 1, sarc, r), np.interp(sk + 1, sarc, c)], 1)
                 - np.stack([np.interp(sk - 1, sarc, r), np.interp(sk - 1, sarc, c)], 1))
            t /= np.linalg.norm(t, axis=1, keepdims=True)
            P0.append(p0); T.append(t)
            NR.append(np.stack([-t[:, 1], t[:, 0]], 1))
            OW.append(np.full(len(sk), o)); PC.append(np.full(len(sk), npiece))
    if not P0:
        out = {k: np.empty(0, dtype=int if k in ('owner', 'piece') else float)
               for k in keys}
        out['rejected'] = rej
        return out
    P0, T, NR = np.concatenate(P0), np.concatenate(T), np.concatenate(NR)
    OW, PC = np.concatenate(OW), np.concatenate(PC)

    # 2. Crossings: another reflection's line inside the window.
    keep = np.ones(len(P0), bool)
    if tree is not None:
        for q, near in enumerate(tree.query_ball_point(P0, cross_px)):
            if np.any(allo[near] != OW[q]):
                keep[q] = False
    rej['crossing'] = int((~keep).sum())
    P0, T, NR, OW, PC = P0[keep], T[keep], NR[keep], OW[keep], PC[keep]

    # 3. Sample every profile at once: (stops, along, across).
    coords = (P0[:, None, None, :] + agrid[None, :, None, None] * T[:, None, None, :]
              + dgrid[None, None, :, None] * NR[:, None, None, :])
    vals = ndimage.map_coordinates(
        img, [coords[..., 0].ravel(), coords[..., 1].ravel()],
        order=1, mode='constant', cval=np.nan).reshape(coords.shape[:3])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN columns
        prof = np.nanmean(vals, axis=1)
    offplate = (np.isnan(vals).mean(axis=(1, 2)) > 0.2) | \
               (np.isfinite(prof).mean(axis=1) < 0.8)
    rej['offplate'] = int(offplate.sum())
    sel = ~offplate
    P0, T, NR, OW, PC, prof = P0[sel], T[sel], NR[sel], OW[sel], PC[sel], prof[sel]

    # 4. Fit them all, then apply the acceptance tests.
    out = {k: np.empty(0, dtype=int if k in ('owner', 'piece') else float)
           for k in keys}
    if len(P0):
        mu, err, A, sg, snr, ok = fit_profiles(dgrid, prof, max_sigma)
        rej['nofit'] = int((~ok).sum())
        weak = ok & (snr < min_snr)
        wide = ok & ~weak & (sg >= 0.99 * max_sigma)
        edgy = ok & ~weak & ~wide & (np.abs(mu) > hw - edge)
        rej['weak'], rej['wide'], rej['edge'] = (int(weak.sum()), int(wide.sum()),
                                                 int(edgy.sum()))
        g = ok & ~weak & ~wide & ~edgy
        Q = P0[g] + mu[g, None] * NR[g]
        out = dict(owner=OW[g].astype(int), row=Q[:, 0], col=Q[:, 1], err=err[g],
                   offset=mu[g], sigma=sg[g], amp=A[g], snr=snr[g],
                   prow=P0[g, 0], pcol=P0[g, 1], piece=PC[g].astype(int))

    # 5. Continuity along each piece.
    good = np.ones(len(out['row']), bool)
    for pid in np.unique(out['piece']):
        ix = np.flatnonzero(out['piece'] == pid)
        off = out['offset'][ix]
        for q, i in enumerate(ix):
            nb = np.r_[off[max(q - jump_half, 0):q], off[q + 1:q + 1 + jump_half]]
            if nb.size >= 2 and abs(off[q] - np.median(nb)) > max(
                    jump_px, jump_nsig * out['err'][i]):
                good[i] = False
    rej['jump'] = int((~good).sum())
    out = {k: v[good] for k, v in out.items()}
    out['rejected'] = rej
    return out


# ── The fit ────────────────────────────────────────────────────────────────────

class RidgeConeFit(object):
    """chi^2 of observed ridge points against the engine's cones.

    With `beam_centre=True` the parameter vector carries two extra trailing
    entries, the beam centre (px, py) in pixels.  Nothing else in the engine
    reads them — only the detector map — so they are set on the engine for the
    evaluation and need no slot in the 24-element guess."""

    def __init__(self, model, ridge, err_floor=0.1, miss_px=50.0,
                 beam_centre=False):
        self.model = model
        self.ridge = ridge
        self.sig = np.hypot(ridge['err'], err_floor)
        self.miss_px = miss_px
        self.beam_centre = beam_centre

    def _split(self, x):
        x = np.asarray(x, dtype=float)
        if self.beam_centre:
            self.model.engine.px, self.model.engine.py = x[-2], x[-1]
            return x[:-2]
        return x

    def pixel_residuals(self, x):
        """Signed distance (px) of every ridge point from its reflection's
        cone, measured on the plate; + is outside the cone (`ConeModel.circles`
        fixes which side that is).  A point whose
        reflection has no cone at `x` scores `miss_px`."""
        g, C = self.model.circles(self._split(x))
        R = self.ridge
        n = len(R['row'])
        res = np.full(n, self.miss_px)
        if C is None or n == 0:
            return res
        U = self.model.backproject(g, R['row'], R['col'])
        # Each point is scored against the cone whose *drawn arc* covers it,
        # falling back to the nearest full cone only if no arc does.  Nearest
        # alone is not enough: a theta range spanning zero puts the two halves
        # of a sweep on two different, nearly coincident cones (~0.1 px apart),
        # and a point between them would tie and flip sign from one evaluation
        # to the next.
        best = np.full(n, np.inf)
        near = np.zeros((n, 3))
        sign = np.ones(n)
        cov = np.zeros(n, bool)
        ub = ts.dms_plane_basis(C['n'])
        vb = np.cross(C['n'], ub)
        for k in range(len(C['owner'])):
            m = np.flatnonzero(R['owner'] == C['owner'][k])
            if not m.size:
                continue
            nk, ck, rk = C['nhat'][k], C['c'][k], C['r'][k]
            # Only the forward ray: a ridge point back-projects to the
            # direction that reaches the plate.  (dms2px would also put a
            # direction's antipode on the same pixel, but the engine's sweep
            # never lands a backward ray on the plate, and scoring antipodes
            # let a cone tie with another from the far side.)
            Uk = U[m]
            ang = (np.arccos(np.clip(Uk @ nk, -1, 1))
                   - np.arccos(np.clip(C['d'][k], -1, 1)))
            w = Uk - ck
            w = w - np.outer(w @ nk, nk)
            phi = np.arctan2(w @ vb[k], w @ ub[k])
            lo, hi = sorted((C['a0'][k], C['a1'][k]))
            inarc = ((hi - lo) >= 2 * np.pi) | (lo + np.mod(phi - lo, 2 * np.pi) <= hi)
            w /= np.linalg.norm(w, axis=1, keepdims=True)
            pk = ck + rk * w
            better = (inarc & ~cov[m]) | ((inarc == cov[m]) & (np.abs(ang) < best[m]))
            sel = m[better]
            best[sel] = np.abs(ang[better])
            near[sel] = pk[better]
            sign[sel] = np.where(ang[better] >= 0, 1.0, -1.0)
            cov[sel] |= inarc[better]
        hit = np.isfinite(best)
        if hit.any():
            pr, pc = self.model.project(g, near[hit])
            res[hit] = sign[hit] * np.hypot(R['row'][hit] - pr, R['col'][hit] - pc)
        return res

    def residuals(self, x):
        try:
            return self.pixel_residuals(x) / self.sig
        except Exception:
            return np.full(len(self.sig), self.miss_px) / self.sig

    def chi2(self, x):
        return float(np.sum(self.residuals(x) ** 2))

    def run(self, x0, loss='soft_l1', f_scale=3.0, diff_step=1e-6, bounds=None,
            tol=1e-12):
        # diff_step: the residual is smooth down to relative steps of ~1e-8
        # (below that the circle fit's own rounding shows), so 1e-6 leaves two
        # decades of margin.  The tolerances are tight because the problem is
        # ill-conditioned (detector tilts trade against chi/psi corrections) and
        # SciPy's defaults stop partway along that valley.
        kw = dict(loss=loss, f_scale=f_scale, x_scale='jac', diff_step=diff_step,
                  ftol=tol, xtol=tol, gtol=tol, max_nfev=200 * (len(x0) + 1))
        if bounds is not None:
            kw['bounds'] = bounds
        return least_squares(self.residuals, np.asarray(x0, dtype=float), **kw)


def parameter_errors(res, n_points):
    """1-sigma parameter errors from the Jacobian at the solution, scaled by
    the reduced chi^2 (the error bars treat the robust loss as quadratic)."""
    J = res.jac
    dof = max(n_points - J.shape[1], 1)
    red = 2.0 * res.cost / dof
    try:
        cov = np.linalg.pinv(J.T @ J) * red
        return np.sqrt(np.clip(np.diag(cov), 0, None)), red
    except Exception:
        return np.full(J.shape[1], np.nan), red


# ── Building a case from a fit.py config ───────────────────────────────────────

def load_case(cfg_path):
    """Engine, image and starting vector from a batch-fit config, set up as
    `fit.py` sets them up.  Returns a dict."""
    from . import loader as do
    import imageio.v2 as imageio

    with open(cfg_path) as f:
        cfg = json.load(f)
    zoomval = cfg['display']['zoomval']
    width = cfg['roi']['width_per_zoom'] * zoomval
    sc = cfg['scan']
    scanpath, scannum = sc['scanpath'], sc['scannum']
    datapoint, datapoint0 = sc['datapoint'], sc['datapoint0']
    bravais = cfg['computation']['bravais']
    detopt = bool(cfg['flags']['detoptimize'])
    energyopt = bool(cfg['flags']['energyopt'])
    numsteps = cfg['computation']['numsteps']
    scatv = ts.scatv_code(cfg['geometry']['scatv'],
                          cfg['geometry'].get('det_orientation'))
    psi = cfg['geometry']['psi']

    d = do.load(scanpath + str(scannum) + '.dat')
    met = d.metadata
    lattice = [met.a, met.b, met.c, met.alpha1, met.alpha2, met.alpha3]
    # Energies as dat2config reads them: energy2 when the scan has it, else the
    # metadata Energy (psi scans, 2009 SRS files).
    _meta = dat2config.extract_metadata(scanpath + str(scannum) + '.dat',
                                        datapoint, datapoint0)
    energy = _meta['energy']
    hkl = np.array(cfg['geometry']['hkl'], dtype=float)
    hkl = hkl * energy / _meta['energy0']
    # The azimuthal reference the slider exports (experiment block) wins, as it
    # does when the slider opens this config; then a geometry override; then
    # the .dat, which is all fit.py reads.
    azir = [met['azih'], met['azik'], met['azil']]
    if 'azir' in cfg.get('experiment', {}):
        azir = list(cfg['experiment']['azir'])
    elif 'azir' in cfg['geometry']:
        azir = list(cfg['geometry']['azir'])

    CONVENTIONAL = bravais in ts.CONVENTIONAL_SYSTEMS
    pc = int(cfg['computation'].get('pseudocubic_transform', 1))
    if CONVENTIONAL:
        reflist = np.array(cfg['crystal']['reflist_hkl'], dtype=float)
        if pc != 1:
            M = ts.pseudocubic_matrix(pc)
            hkl = M @ hkl
            azir = list(M @ np.asarray(azir, dtype=float))
            reflist = reflist @ M.T
        reflist2 = np.zeros_like(reflist)
        labels = [str(list(map(int, r))) for r in reflist]
    else:
        ref6 = np.array(cfg['crystal']['ref_6d'])
        rl, rl2 = ts.Projection6d(ref6).reflection_6d()
        reflist, reflist2 = np.asarray(rl), np.asarray(rl2)
        labels = [str(list(map(int, r))) for r in ref6]
    hklint = np.round(hkl)

    exp = cfg.get('experiment', {})
    tmpl = exp.get('image_template', str(scannum) + '-pilatus2M-files/%05d.tif')
    im = imageio.imread(scanpath + tmpl % dat2config.image_number(exp, datapoint)
                        ).astype(float)
    imdata = ndimage.zoom(im, zoomval, order=3)
    px = cfg['geometry']['px_unscaled'] * zoomval
    py = cfg['geometry']['py_unscaled'] * zoomval

    thb = ts.bragg(lattice, hkl, energy).th()[0]
    td = cfg['computation']['thrange_delta']
    hkllistrange = [thb + td[0], thb + td[1], numsteps]
    psirange = [psi - 360, psi + 360]
    detvects = np.matrix([[1, 0, 0], [0, 0, 1]])

    ig = np.array(cfg['crystal']['initial_guess_base'], dtype=float)
    ig[10] = ig[10] / 2 * zoomval
    ig[14] = energy + ig[14]
    simsigma = cfg['computation']['simsigma_per_zoom'] * zoomval
    eng = ts.dmsfit_ico_hkl(
        np.matrix(reflist), hkllistrange, hklint, psirange, width,
        np.zeros((1, 1)), np.zeros((1, 1, 1)), hkl, detvects, imdata, simsigma,
        azir, psi, px, py, scatv, bravais, detopt, energyopt,
        ig[10], ig[11], ig[12], ig[13], ig[14], np.matrix(reflist2),
        list(ig[15:24]), ig[0])
    eng.setCalLattice(ig[:6])
    eng.setLattice(ig[:6])
    eng.setIGFull(ig)
    eng.setCurveMethod(cfg['computation'].get('curve_method', 'sweep'))
    eng.setPeakMethod(cfg['computation'].get('peak_method', 'gauss'),
                      ts.AUTO_DOUBLET_SIG)
    slots = list(ts.imcalc_param_indices(bravais, detopt, energyopt))
    builderargs = (reflist, ts.pilkhlrange(lattice, hkl, energy, hkllistrange[0],
                                           hkllistrange[1]).hklscan(numsteps),
                   hklint, 1, psirange, 0, hkl, detvects, imdata.shape, simsigma,
                   azir, psi, px, py, scatv, ig[10], ig[11], ig[12], ig[13],
                   ig[14], ig, reflist2, list(ig[15:24]),
                   bravais if CONVENTIONAL else None)
    return dict(cfg=cfg, engine=eng, img=imdata, ig=ig, slots=slots,
                x0=ig[slots], names=[IG_SLOT_NAMES[i] for i in slots],
                width=width, labels=labels, builderargs=builderargs,
                manual_centres=cfg.get('manual_centres', {}), zoomval=zoomval)


def roi_chi2(case, xs, apply=lambda x: x):
    """The existing ROI-centre objective at each vector in `xs`, with kernels
    and experimental centres built at the starting guess exactly as fit.py
    does — so the two methods can be compared on the same footing.  `apply`
    maps a vector to the engine's (setting any beam centre it carries)."""
    eng = case['engine']
    kernel = ts.roibuilder_ico_hkl(case['builderargs'])
    co = ts.multiroifit2(case['img'], kernel, case['width'], 0.02,
                         ts.AUTO_DOUBLET_SIG, eng.peakmethod)[0]
    centres = np.array([co[:, 2]]).T
    for k, v in case['manual_centres'].items():
        centres[int(k)] = v / 2 * case['zoomval']
    eng.kernel, eng.centres = kernel, centres
    return [eng.fit(apply(x)) for x in xs]


# ── Command line ───────────────────────────────────────────────────────────────

def _report(case, ridge, fitter, x0, res, errs, red, t):
    lines = []
    R = ridge
    lines.append('ridge points  %d   (rejected: %s)'
                 % (len(R['row']), ', '.join('%s %d' % kv for kv in R['rejected'].items())))
    lines.append('median point error  %.3f px    median peak sigma %.2f px'
                 % (np.median(R['err']), np.median(R['sigma'])))
    r0 = fitter.pixel_residuals(x0)
    r1 = fitter.pixel_residuals(res.x)
    lines.append('rms distance to cone   start %.3f px   ->   fit %.3f px'
                 % (np.sqrt(np.mean(r0 ** 2)), np.sqrt(np.mean(r1 ** 2))))
    lines.append('chi2  start %.6g  ->  fit %.6g   reduced %.4g   (%d evals, %.1f s, %s)'
                 % (fitter.chi2(x0), fitter.chi2(res.x), red, res.nfev, t, res.message))
    lines.append('')
    lines.append('%-12s %16s %16s %14s %10s' % ('param', 'start', 'fit', '1-sigma', 'shift/sig'))
    for nm, a, b, e in zip(case['names'], x0, res.x, errs):
        lines.append('%-12s %16.8g %16.8g %14.3g %10.2f'
                     % (nm, a, b, e, (b - a) / e if e > 0 else np.nan))
    lines.append('')
    lines.append('per reflection (points, rms px start -> fit):')
    for j in np.unique(R['owner']):
        m = R['owner'] == j
        lines.append('  %-3d %-22s %4d   %8.3f -> %8.3f'
                     % (j, case['labels'][j], m.sum(),
                        np.sqrt(np.mean(r0[m] ** 2)), np.sqrt(np.mean(r1[m] ** 2))))
    return lines


def _plot(path, case, lines_at, ridge, x0, x1, r1):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    img = case['img']
    fig = Figure(figsize=(10, 10))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    lo, hi = np.percentile(img, [1, 99.7])
    ax.imshow(img, cmap='gray', vmin=lo, vmax=hi)
    for x, col, lab in ((x0, 'tab:blue', 'start'), (x1, 'tab:red', 'fit')):
        for i, (_, rr, cc) in enumerate(lines_at(x, 2.0)):
            ax.plot(cc, rr, '-', c=col, lw=0.6, label=lab if i == 0 else None)
    sc = ax.scatter(ridge['col'], ridge['row'], c=r1, s=4, cmap='coolwarm',
                    vmin=-2, vmax=2)
    fig.colorbar(sc, ax=ax, shrink=0.6, label='distance to fitted cone (px)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    fig.savefig(path, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config', nargs='?', default=os.path.join(
        PKGDIR, 'configs', 'fit_conventional_tetragonal_PMN_PT_example.json'))
    ap.add_argument('--passes', type=int, default=2,
                    help='extract ridge, fit; repeat from the fitted geometry')
    ap.add_argument('--step', type=float, default=8.0, help='ridge spacing (px)')
    ap.add_argument('--seg', type=float, default=None,
                    help='segment averaged along the line per point (px; default = step)')
    ap.add_argument('--width', type=float, default=None,
                    help='profile width (px; default = ROI width)')
    ap.add_argument('--narrow', type=float, default=24.0,
                    help='profile width (px) on passes after the first, once the '
                         'lines are close; 0 keeps the full width')
    ap.add_argument('--min-snr', type=float, default=4.0)
    ap.add_argument('--beam-centre', action='store_true',
                    help='also refine the beam centre (px, py)')
    ap.add_argument('--loss', default='soft_l1')
    ap.add_argument('--compare', action='store_true',
                    help='also score start and result with the ROI-centre objective')
    ap.add_argument('--save', action='store_true',
                    help='write Result.txt and an overlay PNG under Processing/')
    a = ap.parse_args(argv)

    case = load_case(os.path.abspath(a.config))
    model = ConeModel(case['engine'])
    width = a.width or case['width']
    x0 = case['x0'].copy()
    eng = case['engine']
    if a.beam_centre:
        x0 = np.r_[x0, float(eng.px), float(eng.py)]
        case['names'] = case['names'] + ['px', 'py']

    def apply(xv):
        """Engine vector for `xv`, setting the beam centre if it carries one."""
        if a.beam_centre:
            eng.px, eng.py = xv[-2], xv[-1]
            return xv[:-2]
        return xv

    def lines_at(xv, spacing=0.5):
        return model.predicted_lines(apply(xv), case['img'].shape, spacing)
    x = x0.copy()
    t0 = time.time()
    for p in range(a.passes):
        w = width if (p == 0 or not a.narrow) else min(width, a.narrow)
        ridge = extract_ridge(case['img'], lines_at(x),
                              w, step=a.step, seg_len=a.seg, min_snr=a.min_snr)
        if not len(ridge['row']):
            sys.exit('no ridge points survived extraction')
        fitter = RidgeConeFit(model, ridge, beam_centre=a.beam_centre)
        res = fitter.run(x, loss=a.loss)
        print('pass %d (width %g px): %d points, chi2 %.6g -> %.6g'
              % (p + 1, w, len(ridge['row']), fitter.chi2(x), fitter.chi2(res.x)))
        x = res.x
    t = time.time() - t0
    errs, red = parameter_errors(res, len(ridge['row']))
    lines = _report(case, ridge, fitter, x0, res, errs, red, t)
    if a.compare:
        c0, c1 = roi_chi2(case, [x0, res.x], apply)
        lines.append('')
        lines.append('ROI-centre chi2 (fit.py objective)  start %.6g  ->  ridge fit %.6g'
                     % (c0, c1))
    print('\n'.join(lines))
    if a.save:
        sc = case['cfg']['scan']
        out = os.path.join(os.getcwd(), 'Processing', '%s_dp%s_%s_RidgeCone'
                           % (sc['scannum'], sc['datapoint'],
                              time.strftime('%Y%m%d-%H%M%S')))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'Result.txt'), 'w') as f:
            f.write('config  %s\n\n' % os.path.abspath(a.config))
            f.write('\n'.join(lines) + '\n\n')
            full = case['ig'].copy()
            full[case['slots']] = apply(res.x)
            f.write('initial_guess = np.array([%s])\n'
                    % ', '.join('%.10g' % v for v in full))
            if a.beam_centre:
                f.write('beam centre (zoomed px)  px %.4f  py %.4f\n'
                        % (res.x[-2], res.x[-1]))
        _plot(os.path.join(out, 'RIDGE.png'), case, lines_at, ridge, x0, res.x,
              fitter.pixel_residuals(res.x))
        print('-> ' + out)
    return res


if __name__ == '__main__':
    main()
