"""Batch multiple-intersection (Renninger triple-intersection) lattice fit.

Refine a conventional crystal lattice by driving the three Kossel lines of one
or more secondary-reflection triples to a common point on the stereographic
projection.  This is the image-free, multiple-diffraction analogue of
``fit.py``: instead of matching ROI peak positions in a detector image it
matches the geometry of simultaneous (multiple) diffraction, which is highly
sensitive to the small lattice distortions of pseudo-symmetric crystals.

Run as a module from the repository root:

    python -m DMSAnalysis.tripfit [config.json]

With no argument it falls back to the example config shipped in ``configs/``.
The engine (``ts_quasi.tripfit`` and helpers) is shared with the rest of the
package, and the crystal-system lattice constraints use the same table-driven
``lattice_free_slots`` / ``expand_lattice`` layer as the image fit, so the free
parameters cannot drift between the two workflows.
"""

import inspect, os, sys, subprocess, json
from time import strftime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import minimize, differential_evolution, basinhopping

from . import ts_quasi as ts

plt.ion()
rc('xtick', labelsize=6)
rc('ytick', labelsize=6)

PKGDIR  = os.path.abspath(os.path.dirname(__file__))
CONFIGS = os.path.join(PKGDIR, 'configs')
datestr = strftime('%Y%m%d%H%M')


# ── config (optional path arg; example config is the fallback) ──────────────────
if len(sys.argv) > 1:
    _cfg_path = os.path.abspath(sys.argv[1])
else:
    _cfg_path = os.path.join(CONFIGS, 'tripfit_rhombohedral_PMN_PT_example.json')
with open(_cfg_path) as _f:
    cfg = json.load(_f)

_flags = cfg.get('flags', {})
save   = int(_flags.get('save', 0))
fit    = int(_flags.get('fit', 0))

_geo = cfg['geometry']
hkl  = np.array([_geo['hkl']], dtype=float)
azir = list(_geo['azir'])

_comp      = cfg['computation']
bravais    = _comp['bravais']
resolution = int(_comp.get('resolution', 1000))
OptMethod  = _comp.get('opt_method', 'Powell')
tolerance  = float(_comp.get('tolerance', 1e-17))
boundrange = _comp.get('boundrange', [-0.012, 0.012])
rr         = float(_comp.get('rr', 0.0))          # azimuthal pre-rotation (deg)
niter      = int(_comp.get('bh_niter', 50))
strat      = ts.DE_Strategy.get(_comp.get('de_strategy', 'best1bin'), 'best1bin')

if bravais not in ts.CONVENTIONAL_SYSTEMS:
    raise SystemExit(
        "computation.bravais must be one of the conventional systems %s, got %r"
        % (list(ts.CONVENTIONAL_SYSTEMS), bravais))

initial_guess = np.array(cfg['crystal']['initial_guess'], dtype=float)
free_slots    = ts.lattice_free_slots(bravais)
ig            = initial_guess[free_slots]          # reduced free-parameter vector

_disp = cfg.get('display', {})
lim   = float(_disp.get('lim', 0.005))
dpi   = int(_disp.get('dpi', 130))

# ── intersection groups ─────────────────────────────────────────────────────────
# Each group is a secondary-reflection triple whose three Kossel lines should
# meet at one stereographic point.  An optional azimuthal pre-rotation ``rr``
# about the reference vector is applied to hkl and every reflection list (kept
# for parity with the original script; rr = 0 leaves the indices untouched).
_groups_cfg = cfg['intersections']
if not _groups_cfg:
    raise SystemExit('config "intersections" must list at least one triple')

if rr != 0.0:
    _R = ts.rotxyz(azir, rr).rmat()
    hkl = np.round((_R * np.matrix(hkl).T).T)

groups = []
for gi, gc in enumerate(_groups_cfg):
    reflist = np.matrix(np.array(gc['reflist'], dtype=float))
    if reflist.shape != (3, 3):
        raise SystemExit('intersection group %d: reflist must be 3x3' % gi)
    if rr != 0.0:
        reflist = np.round((_R * reflist.T).T)
    tf = ts.tripfit(hkl, reflist, azir, resolution, bravais,
                    float(gc['energy']), list(gc.get('intercepts', [0, 0, 0])),
                    float(gc.get('target', 0.0)))
    groups.append({'label': gc.get('label', 'T%d' % (gi + 1)),
                   'reflist': reflist, 'tf': tf})

n_groups = len(groups)


def combined(x):
    '''Summed residual across all intersection groups at reduced params ``x``.'''
    return float(np.sum([g['tf'].fit(x) for g in groups]))


objective = groups[0]['tf'].fit if n_groups == 1 else combined

# ── output directory (snapshot on save) ─────────────────────────────────────────
outpath = os.path.join(os.getcwd(), 'Processing', datestr + '_TripFit') + os.sep
if save:
    os.makedirs(outpath, exist_ok=True)
    subprocess.call('cp ' + inspect.getfile(inspect.currentframe()) + ' ' + outpath + '.',
                    shell=True)
    subprocess.call('cp ' + os.path.join(PKGDIR, 'ts_quasi.py') + ' ' + outpath + '.',
                    shell=True)
    subprocess.call('cp ' + _cfg_path + ' ' + outpath + '.', shell=True)

# ── fit ──────────────────────────────────────────────────────────────────────
lb = ig + boundrange[0]
ub = ig + boundrange[1]
bounds = list(zip(lb, ub))

if fit:
    if OptMethod == 'GA':
        print('Fitting with Differential Evolution (%s), %s constraints'
              % (strat, bravais))
        res = differential_evolution(objective, bounds, strategy=strat, polish=True)
    elif OptMethod.startswith('BH'):
        inner = OptMethod[2:] or 'Powell'
        print('Fitting with Basin Hopping + %s, %s constraints' % (inner, bravais))
        res = basinhopping(objective, ig, minimizer_kwargs={'method': inner},
                           niter=niter)
    elif OptMethod == 'TNC':
        print('Fitting with TNC, %s constraints' % bravais)
        res = minimize(objective, ig, method='TNC', bounds=bounds, tol=tolerance,
                       options={'xtol': tolerance, 'ftol': tolerance})
    else:
        print('Fitting with %s, %s constraints' % (OptMethod, bravais))
        res = minimize(objective, ig, method=OptMethod, tol=tolerance,
                       options={'xtol': tolerance, 'ftol': tolerance})
    xbest = np.atleast_1d(np.asarray(res.x, dtype=float))
    opt = objective(xbest)
else:
    xbest = ig
    opt = objective(ig)
    res = ts.res(ig)
    OptMethod = 'None'

def reduced_to_lattice(x):
    '''Full [a,b,c,alpha,beta,gamma] for a reduced free-parameter vector.'''
    six = np.zeros(6)
    for slot, val in zip(free_slots, np.atleast_1d(x)):
        six[slot] = val
    return ts.expand_lattice(bravais, six)


lattice_fit = reduced_to_lattice(xbest)

# ── per-group full solutions for plotting ───────────────────────────────────────
for g in groups:
    g['full'] = g['tf'].full(xbest)


# ── plotting ────────────────────────────────────────────────────────────────
def _ref_title(reflist):
    return ''.join(str(np.squeeze(np.array(reflist[i, :]))) for i in range(3))


def plotster(ax, interc, r1, r2, r3, alpha=1.0):
    ax.plot(interc[0, 0], interc[0, 1], 'or')
    ax.plot(interc[1, 0], interc[1, 1], 'og')
    ax.plot(interc[2, 0], interc[2, 1], 'ob')
    ax.plot(r1[:, 0], r1[:, 1], 'r', alpha=alpha)
    ax.plot(r2[:, 0], r2[:, 1], 'g', alpha=alpha)
    ax.plot(r3[:, 0], r3[:, 1], 'b', alpha=alpha)


fig = plt.figure(figsize=(max(4, 3 * n_groups), 4), facecolor='white', dpi=dpi)
for gi, g in enumerate(groups):
    interc, st0, st1, st2, vr0, vr1, vr2 = g['full']
    ax = fig.add_subplot(1, n_groups, gi + 1, adjustable='box', aspect=1)
    plotster(ax, interc, st0, st1, st2, 1)
    ax.set_title(_ref_title(g['reflist']), fontsize=6)
    ax.set_xlabel(g['label'])
plt.subplots_adjust(wspace=0.28)
fig.canvas.manager.set_window_title('TripFit — %s' % bravais)


# ── report / save ─────────────────────────────────────────────────────────────
def saveResult():
    with open(outpath + 'Result.txt', 'w') as f:
        f.write('bravais    = %s\n' % bravais)
        f.write('method     = %s\n' % OptMethod)
        f.write('resolution = %d\n' % resolution)
        f.write('reduced_x  = %s\n' % np.array2string(np.atleast_1d(xbest)))
        f.write('lattice    = %s\n' % np.array2string(np.array(lattice_fit)))
        f.write('opt        = %s\n' % opt)


if save:
    plt.savefig(outpath + 'PLOT.svg', format='svg')
    saveResult()
    print('Data written to ' + outpath)

print('bravais    :', bravais)
print('method     :', OptMethod)
print('resolution :', resolution)
print('reduced x  :', np.atleast_1d(xbest))
print('lattice    :', np.array(lattice_fit))
print('residual   :', opt)

if __name__ == '__main__':
    plt.show()
