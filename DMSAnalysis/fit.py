import inspect, os, sys, subprocess
from mpl_toolkits.mplot3d import Axes3D
PKGDIR  = os.path.abspath(os.path.dirname(__file__))
CONFIGS = os.path.join(PKGDIR, 'configs')

from numpy import linalg as LA
from . import ts_quasi as ts
import json
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from joblib import Parallel, delayed
from collections import OrderedDict
from . import loader as do
import imageio.v2 as imageio
import re
plt.ion()
import copy
#from pyswarm import pso
from time import strftime
from matplotlib import rc
rc('xtick', labelsize=6)
rc('ytick', labelsize=6)
datestr=strftime("%Y%m%d%H%M")

if len(sys.argv) > 1:
    _cfg_path = os.path.abspath(sys.argv[1])
else:
    _cfg_path = os.path.join(
        CONFIGS, 'fit_fivefold_axis_AlPdMn_Not_Annealed_2M_2ROIS_internal_hkl.json')
with open(_cfg_path) as _f:
    cfg = json.load(_f)

#------------------------------------------------------------------------------
# Reduce zoomval to 0.25 or 0.5 make simulated pattern clearer
# This also sets the resolution
#------------------------------------------------------------------------------
zoomval  = cfg["display"]["zoomval"]
width    = cfg["roi"]["width_per_zoom"] * zoomval
comwidth = cfg["roi"]["comwidth_per_zoom"] * zoomval

scan = scannum = cfg["scan"]["scannum"]
datapoint0    = cfg["scan"]["datapoint0"]
datapoint     = cfg["scan"]["datapoint"]
scanpath      = cfg["scan"]["scanpath"]

imnum=datapoint+1
tolerance    = cfg["computation"]["tolerance"]
scatv        = cfg["geometry"]["scatv"]
save         = cfg["flags"]["save"]
fit          = cfg["flags"]["fit"]
firstplot    = cfg["flags"]["firstplot"]
show_centres = cfg["flags"]["show_centres"]
show_numbers = cfg["flags"]["show_numbers"]
axis_off     = cfg["flags"]["axis_off"]
detoptimize  = cfg["flags"]["detoptimize"]
energyopt    = cfg["flags"]["energyopt"]

colourlim = cfg["display"]["colourlim"]
colmap    = cfg["display"]["colmap"]

#------------------------------------------------------------------------------
#                            set bravais constraints for fitting
#------------------------------------------------------------------------------
bravais    = cfg["computation"]["bravais"]
autoreflist= cfg["flags"]["autoreflist"]

#===============================================================================
#                               Minimize Method
#===============================================================================
OptMethod = cfg["computation"]["opt_method"]
# OptMethod = ts.minimizers['Nelder-Mead']
# OptMethod = ts.minimizers['Differential Evolution']
# OptMethod = ts.minimizers['CG']
# OptMethod = ts.minimizers['L_BFGS_B']
# OptMethod = ts.minimizers['Powell']
# OptMethod = ts.minimizers['SW']
# OptMethod = ts.minimizers['TNC'] # good for fast testing
# OptMethod = ts.minimizers['Swarm']
# OptMethod = ts.minimizers['dogleg']
# OptMethod = ts.minimizers['trust_ncg']
# OptMethod = 'COBYLA'
# OptMethod = 'BHCOBYLA'
# OptMethod = 'BHNelderMead'
# OptMethod = 'BHPowell'

strat=ts.DE_Strategy['best1exp']

intensity  = cfg["computation"]["intensity"]
threshold  = cfg["computation"]["threshold"]
numsteps   = cfg["computation"]["numsteps"]
simsigma   = cfg["computation"]["simsigma_per_zoom"] * zoomval

colourmap  = cfg["display"]["colourmap"]
cmap=ts.cmap()['hot']
lattice2   = cfg["crystal"]["lattice2"]

cif_file = cfg["paths"]["cif_file"]

if fit:
    fittype = OptMethod
else:
    fittype = 'NoFit'
outpath=os.path.join(os.getcwd(), 'Processing', datestr+'_'+str(imnum)+'_'+str(scan)+'_fivefold_2ROIS_AlPdMn_Not_Annealed_'+fittype)+'/'

if save:
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    cmdstr='cp '+inspect.getfile(inspect.currentframe()) + ' '+outpath + '.'
    subprocess.call(cmdstr, shell=True)
    cmdstr2='cp '+os.path.join(PKGDIR, 'ts_quasi.py')+ ' '+outpath + '.'
    subprocess.call(cmdstr2, shell=True)
    cmdstr3='cp '+_cfg_path+ ' '+outpath + '.'
    subprocess.call(cmdstr3, shell=True)

PV=[0,1]
d = do.load(scanpath + str(scannum) + '.dat')
met=d.metadata
lattice=[met.a,met.b,met.c,met.alpha1,met.alpha2,met.alpha3]

psi = cfg["geometry"]["psi"]

hkl = np.array(cfg["geometry"]["hkl"])
hkl = hkl*d.energy2[datapoint]/d.energy2[datapoint0]

hklint=np.round(hkl)
imtemplate=str(scannum)+'-pilatus2M-files/%05d.tif'

try:
    energy = d.energy2[datapoint]
except:
    try:
        energy = d.metadata.Energy
    except:
        try:
            energy = d.DCMenergy
        except:
            energy = d.metadata.en


azir=[d.metadata['azih'],d.metadata['azik'],d.metadata['azil']]

################################### Load and filter image ##################

mask = [494,1274,980,1470]
im = imageio.imread(str(scanpath + str(imtemplate % imnum)))

mask_array = np.meshgrid(np.r_[1269:1470], np.r_[494:980])
mask_array = np.meshgrid(np.r_[845:1470], np.r_[494:980])
# im[mask_array[0],mask_array[1]]=0
lowpass = ndimage.gaussian_filter(im,60,0)
# im=im+lowpass.min()
imdata = im-lowpass
imdata = im
im=ndimage.zoom(imdata, zoomval, order=3)
imdata=np.copy(im)

px = cfg["geometry"]["px_unscaled"] * zoomval
py = cfg["geometry"]["py_unscaled"] * zoomval

thb=ts.bragg(lattice,hkl,energy).th()[0]
_td = cfg["computation"]["thrange_delta"]
thrange=[thb+_td[0],thb+_td[1]]

psirange=[psi-360,psi+360]
detvects=np.matrix([[1,0,0],[0,0,1]])
hkllist=ts.pilkhlrange(lattice,hkl,energy,thrange[0],thrange[1]).hklscan(numsteps) # external
hkllistrange = [thrange[0],thrange[1], numsteps]
#===============================================================================
##################### Get reduced reflection list  #######################
if autoreflist:
    mslist=[[np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN,np.NAN]]
    hkllistcorse=ts.pilkhlrange(lattice,hkl,energy,thrange[0],thrange[1]).hklscan(30)
    SF, reflist, lattice2 , structure, sfc = ts.loadcif(cif_file,energy)
    for hklval in range(len(hkllistcorse[:,0])):
        ms=ts.calcms(lattice,hkllistcorse[hklval,:],hklint,reflist,energy,azir)
        mslist=np.concatenate((mslist,ms.full()),0)
    mslist=ts.reducebypsirange(mslist,psirange)
    reflist=np.matrix(ts.uniquearray(mslist[:,0:3]))
else:
    ref_6d = np.array(cfg["crystal"]["ref_6d"])

    p6d=ts.Projection6d(ref_6d)
    reflist0=p6d.reflection_6d()
    reflist=reflist0[0]   # parallel component   HKL = h+h'*tau, k+k'*tau, l+l'*tau,
    reflist2=reflist0[1]  # perpendicular component

#                            0_a        1_b    2_c    3_alpha     4_beta  5_gamma  6_psicor  7_hcor   8_kcor   9_lcor   10_detdist              11_dxrot   12_dyrot    13_dzrot    14_energy           15_pmatrix ->
ig_base = np.array(cfg["crystal"]["initial_guess_base"], dtype=float)
ig_base[10] = ig_base[10] / 2 * zoomval   # detdist scales with zoom
ig_base[14] = energy + ig_base[14]         # energy: stored offset added to loaded value
initial_guess = ig_base

ig = initial_guess

#detdistancepx,rotx,roty,rotz,energy= 2.90633089e+03, -1.43122115, -0.400263388, -36.1330508,7.30926274
detdistancepx,rotx,roty,rotz,energy=ig[10],ig[11],ig[12],ig[13],ig[14]

# imdata=ts.fft2_filter(imdata,190, 90,2,2)[0]
# lowpass = ndimage.gaussian_filter(imdata, 10,0)
# imdata=imdata-lowpass
# imdata=ndimage.convolve(imdata,ts.makekernel('gauss',15,2,0.5))
mtrx2 = [ig[15],ig[16],ig[17],ig[18],ig[19],ig[20],ig[21],ig[22],ig[23]]

intensity = 1
threshold = 0
builderargs=reflist,hkllist,hklint,intensity,psirange,threshold,hkl,detvects,imdata.shape,simsigma,azir,psi,px,py,scatv,detdistancepx,rotx,roty,rotz,energy,ig,reflist2,mtrx2


# builderargs=reflist,hkllist,hklint,psirange,width,[0,0,0,0],imdata,hkl,detvects,imdata,simsigma,azir,psi,px,py,scatv,detdistancepx,rotx,roty,rotz,energy,ig
kernel=ts.roibuilder_ico_hkl(builderargs)

# linedataxc,linedatayc,linedataxcs,linedataycs, centres = ts.multiroigconv(imdata,kernel,width,[0,5,20],3.0, 2.0)
imcoeffs,linedatax,linedatay, fitpoints, rois, pcov =ts.multiroifit2(imdata,kernel,width,0.02,10.0)
# imcoeffs,linedatax,linedatay, fitpoints, rois =ts.multiroimin(imdata,kernel,width,0,0.01)
centres=np.array([imcoeffs[:,2]]).T
# linedatax,linedatay, centres, rois = ts.multiroicom(imdata,kernel,width,comwidth)
fitpoints=linedatay
#------------------------------------------------------------------------------
#                manually set centres if auto fit isn't good
#------------------------------------------------------------------------------
for _idx, _val in cfg["manual_centres"].items():
    centres[int(_idx)] = _val / 2 * zoomval

#                            0_a        1_b    2_c    3_alpha     4_beta  5_gamma  6_psicor  7_hcor   8_kcor   9_lcor   10_detdist              11_dxrot   12_dyrot    13_dzrot    14_energy           15_pmatrix ->
if bravais == 'icosahedral':

    if detoptimize:
        if energyopt:
            ig = initial_guess[[0,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
        else:
            ig = initial_guess[[0,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23]]
    else:
        if energyopt:
            ig = initial_guess[[0,6,7,8,9,14,15,16,17,18,19,20,21,22,23]]
            print(ig)
        else:
            ig = initial_guess[[0,6,7,8,9,15,16,17,18,19,20,21,22,23]]

elif bravais == 'cubic_no_strain':

    if detoptimize:
        if energyopt:
            ig = initial_guess[[0,6,7,8,9,10,11,12,13]]
        else:
            ig = initial_guess[[0,6,7,8,9,10,11,12]]
    else:
        if energyopt:
            ig = initial_guess[[0,6,7,8,13]]
        else:
            ig = initial_guess[[0,6,7,8]]

elif bravais == 'icosahedral_fixed_a':

    if detoptimize:
        if energyopt:
            ig = initial_guess[[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
        else:
            ig = initial_guess[[6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23]]
    else:
        if energyopt:
            ig = initial_guess[[6,7,8,13,14,15,16,17,18,19,20,21,22,23]]
        else:
            ig = initial_guess[[6,7,8,14,15,16,17,18,19,20,21,22,23]]

elif bravais == 'calibrate':

    if detoptimize:
        if energyopt:
            ig = initial_guess[[6,7,8,9,10,11,12,13]]
        else:
            ig = initial_guess[[6,7,8,9,10,11,12]]
    else:
        if energyopt:
            ig = initial_guess[[6,7,8,13]]
        else:
            ig = initial_guess[[6,7,8]]

iglow = ig - 1.5
ighigh = ig + 1.5
bounds=zip(iglow,ighigh)
# linedataxc,linedatayc,linedataxcs,linedataycs, centres = ts.multiroispline(imdata,kernel,width,5,2,1)
# imcentres=imcoeffs[:,2]

#################################################################
starttime=time.time()

dms = ts.dmsfit_ico_hkl(reflist, hkllistrange, hklint, psirange, width, centres, kernel, hkl, detvects, imdata,
             simsigma, azir, psi, px, py, scatv, bravais, detoptimize, energyopt, detdistancepx, rotx, roty, rotz, energy, reflist2, mtrx2,ig[0])

dms.setCalLattice(initial_guess[:6])
dms.setLattice(initial_guess[:6])


if fit:
#     dms = ts.dmsfit2(detdistancepx,reflist,hkllist,hklint,intensity,psirange,threshold,hkl,detvects,imdata,azir,psi,px,py,scatv) #init arguments
#     minimizer_kwargs = {"method": "COBYLA"}
    if OptMethod == 'GA':
        print('Using Differential Evolution with strategy '+strat)
        res = differential_evolution(dms.fit,bounds,strategy=strat,polish=True,workers=-1)

    elif OptMethod == 'SW':
        print('Using Pyswarm')
        lb=list(np.array(bounds)[:,0])
        ub=list(np.array(bounds)[:,1])
        res1, fopt = pso(dms.fit,lb,ub)
        res=ts.res(res1)

    elif OptMethod == 'BHPowell':
        print('Using Basinhopping')
        minimizer_kwargs = {"method": "Powell"}
        res=basinhopping(dms.fit, ig, minimizer_kwargs=minimizer_kwargs, niter=150)
    elif OptMethod == 'BHCOBYLA':
        print('Using Basinhopping')
        minimizer_kwargs = {"method": "COBYLA"}
        res=basinhopping(dms.fit, ig, minimizer_kwargs=minimizer_kwargs, niter=400)
    elif OptMethod == 'BHNelderMead':
        print('Using Basinhopping')
        minimizer_kwargs = {"method": "Nelder-Mead"}
        res=basinhopping(dms.fit, ig, minimizer_kwargs=minimizer_kwargs, niter=400)
    else:
        _n_starts = cfg["computation"].get("n_parallel_starts", 4)
        print('Using ' + OptMethod + ' with ' + str(_n_starts) + '-start parallel search')
        _rng = np.random.default_rng(42)
        _starts = [ig] + [ig + _rng.uniform(-0.5, 0.5, ig.shape) for _ in range(_n_starts - 1)]
        def _run_one(_ig_start):
            _dms = copy.deepcopy(dms)
            return minimize(_dms.fit, _ig_start, method=OptMethod, tol=tolerance,
                            options={'xtol': tolerance, 'ftol': tolerance})
        _results = Parallel(n_jobs=_n_starts)( delayed(_run_one)(s) for s in _starts )
        res = min(_results, key=lambda r: r.fun)


#     opt,simim,dmsindex,dataim2,inputs=dms.full(res.x)
else:
#     inputvals = dms.full(ig)[-1]
    res=ts.res(ig)

opt,simim,dmsindex,dataim2,inputs=dms.full(res.x)


print('Fit took '+str(time.time()-starttime)+' s')

#===============================================================================


im2=np.copy(im)
# imoverlay=im2.T
imoverlay=im2
imoverlay[dmsindex]=colourmap[1]
#===============================================================================
#                         Plotting
#===============================================================================

# plt.figure()
# plt.imshow(simim, cmap=colmap,clim=(simim.min(), simim.max()))
# plt.title('simim')
#
plt.figure()
imroisr=(np.sum(rois,2)*colourlim[1]).astype(int)
plt.imshow(imroisr, cmap=colmap,clim=(colourlim[0], colourlim[1]))
plt.title('ROI')
#
# plt.figure()
# plt.imshow(im, cmap=colmap,clim=(colourlim[0], colourlim[1]))
# plt.title('Raw Image')
#
# plt.figure()
# plt.imshow(imdata, cmap=colmap,clim=(colourlim[0], colourlim[1]))
# plt.title('Filtered Image')
#
plt.figure()
plt.imshow(imoverlay, cmap=colmap,clim=(colourlim[0], colourlim[1]))
plt.title('Overlay')


imcoeffs,linedatasimx,linedatasimy, fitpointssim, rois2, covmat =ts.multiroifit(simim,kernel,width,10)
# imcentres=imcoeffs[:,2]

#
# subcells=np.ceil(np.sqrt(kernel.shape[2])).astype(int)
# subcellsx = subcells
# subcellsy = subcells
subcellsx = cfg["display"]["subcellsx"]
subcellsy = cfg["display"]["subcellsy"]

if firstplot:
    fig, axlist = plt.subplots(subcellsx,subcellsy,figsize=(6, 10))


ii = np.indices((subcellsx,subcellsy))
irow = ii[0].flatten()
icol = ii[1].flatten()

refnum = 0
roicount = 0
axislist=[]

def millerSring(input_string):
    output_string = re.sub('-(\d)', r'\\bar{\1}', input_string)
    output_string = output_string.replace('[','$(')
    output_string = output_string.replace(']',')$')
    return output_string


for i1 in range(kernel.shape[2]):
    plt.sca(axlist[irow[i1],icol[i1]])
    # plt.plot(linedatax[i1],linedatay[i1],c='black')
    # plt.plot(linedatax[i1],linedatay[i1],'-',c='red',linewidth=0.5)
    plt.plot(linedatax[i1],linedatay[i1],'-',c='green',linewidth=0.5)

    plt.plot(linedatax[i1],fitpoints[i1],'.',markersize=2,c='r')
    if show_centres:
        plt.plot([centres[i1],centres[i1]],[fitpoints[i1].min(),linedatay[i1].max()],'g',linewidth=0.5)

    # plt.title(str(i1)+' '+str(ref_6d[refnum,:]),fontsize=10)
    if show_numbers:
        title_string = str(i1)+' '+millerSring(str(ref_6d[refnum,:]))
    else:
        title_string = millerSring(str(ref_6d[refnum,:]))
    plt.title(title_string,fontsize=10)

    yscale = (linedatay[i1].max()-linedatay[i1].min())/(linedatasimy[i1].max()-linedatasimy[i1].min())
    yoffset = linedatay[i1].min()-(linedatasimy[i1]*yscale).min()
    # plt.plot(linedatasimx[i1],(linedatasimy[i1]*yscale)+yoffset,c='grey')
    plt.plot(linedatasimx[i1],(linedatasimy[i1]*yscale)+yoffset,'-.',c='blue',linewidth=0.5)
    # plt.plot(linedatasimx[i1],(linedatasimy[i1]*yscale)+yoffset,'--',c='m',linewidth=0.5)

    plt.plot(linedatasimx[i1],(fitpointssim[i1]*yscale)+yoffset,'.', markersize=2,c='g')
    # plt.ticklabel_format(axis = 'Y', style = 'plain')
    if axis_off:
        plt.axis('off')
    if roicount == 1:
        refnum += 1
        roicount = -1
    roicount +=1

plt.tight_layout()

fig = plt.gcf()
fig.canvas.setWindowTitle('iAlPdMn Not Annealed')



im3 = np.copy(im)
holder=np.zeros((imdata.shape[0],imdata.shape[1],3))
imr=np.zeros((imdata.shape[0],imdata.shape[1]))
img=np.zeros((imdata.shape[0],imdata.shape[1]))
imb=np.zeros((imdata.shape[0],imdata.shape[1]))
imr[dmsindex]=255
holder[:,:,0]=imr
holder[:,:,1] = imr
im3[im3>colourmap[1]]=colourmap[1]
holder[:,:,2] = (255./im3.max())*im3

with_roi = np.copy(holder)
with_roi[:,:,1]=imroisr

doubleim=np.zeros([im.shape[0],im.shape[1]*2])
doubleim[:,:im.shape[1]]=im
doubleim[:,im.shape[1]:]=np.sum(holder,2)


sumholder = np.sum(holder,2)


if save:
    imageio.imsave(str(outpath+'IM'+'_%05d.png' % scan),holder)
    imageio.imsave(str(outpath+'ROIS'+'%05d.png' % scan),with_roi.astype(np.uint8))
    plt.savefig(str(outpath+'_PLOT'+'_%05d.svg' % scan),format='svg')
    np.savetxt(outpath+'res.x.txt',res.x)
    plt.close(fig)
    print('Data written to '+str(outpath))

plt.figure()
plt.imshow(sumholder, cmap=colmap,clim=(colourlim[0], colourlim[1]))
plt.title('Overlay')

def saveResult():
    f = open(outpath+"Result.txt", "w")
    f.write('initial_guess = np.array([')
    [f.write("%f," % inputs[ii]) for ii in np.arange(len(inputs))]
    f.write('])\n')
    f.write('opt = '+str(opt))
    f.close()

if save:
    saveResult()

print(str(time.localtime()[3])+':'+ '%02d' %time.localtime()[4])


# plt.show()
