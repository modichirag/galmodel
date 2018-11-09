import numpy as np
import re, os, h5py, json
import features as ft
from functools import partial
from nbodykit.source.catalog import BigFileCatalog
from nbodykit.lab import cosmology, HaloCatalog, Zheng07Model
from pmesh.pm import ParticleMesh


pfile = "/project/projectdirs/astro250/chmodi/cosmo4d/ics_matterpow_0.dat"
#mf = MF.Mass_Func(pfile, 0.3175)
#proj = '/project/projectdirs/cosmosim/lbl/chmodi/cosmo4d/'
#proj = '/global/cscratch1/sd/chmodi/cosmo4d/'
proj = '/project/projectdirs/astro250/chmodi/cosmo4d/'

def relu(x):
    xx = x.copy()
    mask=x>0
    xx[~mask] = 0
    return xx


def elu(x, alpha=1):
    xx = x.copy()
    mask=x>0
    xx[~mask] = alpha*(np.exp(x[~mask]) - 1)
    return xx

def sigmoid(x, t=0, w=1):
    return 1/(1+np.exp(-w*(x-t)))

def linear(x):
    return x.copy()


def readfiles(pm, dpath, R1, R2, abund = True, quad=False, z=0, shear=False, doexp=False, mexp=None, cc=None, stellar=False, Rmore=None, verbose=True):
    '''Read halo and matter file to create features for net.
    The path should have folder dynamic/1 in it.
    If quad, return quadratic combinations of mesh
    If shear, estimate shear
    Rmore - list of more smoothing scales to be returned. 
    If abund, match to mass function - not correctly implemented anymore
    stellar, doexp, mexp, cc are not correctly implemented anymore
    '''
    #Matter
    bs, nc = pm.BoxSize[0], pm.Nmesh[0]
    from nbodykit.lab import CurrentMPIComm
    CurrentMPIComm.set(pm.comm) #WORKAROUND because BigFileCatalog ignores comm o/w. Should not need after nbodykit v0.3.4
    dyn = BigFileCatalog(dpath + 'dynamic/1/', header = 'Header', comm=pm.comm)
    ppos = dyn['Position'].compute()
    final = pm.paint(ppos)
    dmdict = smoothft(final, R1, R2, quad=quad, shear=shear, pm=pm, Rmore=Rmore)

    #Halos
    if verbose: print('Read Halo files')
    hdict, halos = gridhalos(pm, dpath=dpath, R1=R1, R2=R2, abund=abund, z=z, doexp=doexp, mexp=mexp, cc=cc, stellar=stellar, verbose=verbose)

    meshdict ={}
    meshdict.update(dmdict)
    meshdict.update(hdict)    
    return meshdict, halos



def smoothft(mesh, R1=None, R2=None, quad=False, shear=False, pm=None, Rmore=None):
    '''Currently does the following to the mesh-
    Deconvolve -> Subtract mean -> Smooth at given scales & GD
    '''
    pmdc = ft.decic(mesh)
    meshdict = {'final':mesh.value, 'decic':pmdc.value}
    mean = pmdc[...].mean()
    pmdc[...] -= mean
    pmdc[...] /= mean

    dk = pmdc.r2c()

    #Create features and parameters
    if R1 is not None:
        R1mesh = ft.smooth(dk, R1, 'fingauss')
        meshdict['R1'] = R1mesh.value
    if R2 is not None:
        R2mesh = ft.smooth(dk, R2, 'fingauss')
        meshdict['R2'] = R2mesh.value
    if R2 is not None and R2 is not None:
        R12mesh = R1mesh - R2mesh
        meshdict['R12'] = R12mesh.value

    if shear: 
        s2 = ft.shear(pm, mesh)
        meshdict['shear'] = s2.value
    if Rmore is not None:
        for i, R in enumerate(Rmore):
            Rmesh = ft.smooth(dk, R, 'fingauss')
            meshdict['Rmore%d'%i] = Rmesh.value
            
    if quad:
        meshdict['fR1'] = (R1mesh*mesh).value
        meshdict['fR2'] = (R2mesh*mesh).value
        meshdict['R1R2'] = (R1mesh*R2mesh).value

    return meshdict
    

def gridhalos(pm, dpath=None, pos=None, mass=None, R1=None, R2=None, rank=None, abund = True, sigma = None, seed = 123, pmesh = True, z = 0,  
              doexp=False, mexp=None, cc=None, stellar=False, verbose=True):
    '''Read halo file or pos/mas arrays and grid and smooth them after scattering or matching abundance or both
    The path should have folder FOF in it.
    if pmesh: return the position and mass on grid, with other smoothing scales
    rank: rank of the last halo in the catalog to be reuturned. If none, all halos used
    If abund, match to mass function - not correctly implemented anymore
    stellar, doexp, mexp, cc are not correctly implemented anymore
    '''
    #Matter
    bs, nc = pm.BoxSize[0], pm.Nmesh[0]

    #Halos
    if dpath is not None:
        from nbodykit.lab import CurrentMPIComm
        CurrentMPIComm.set(pm.comm) #WORKAROUND because BigFileCatalog ignores comm o/w. Should not need after nbodykit v0.3.4
        fofcat = BigFileCatalog(dpath + 'FOF', header='Header', comm=pm.comm)

        halopos = fofcat['PeakPosition'].compute()[1:]
        halomass = (fofcat['Mass'].compute()*10**10)[1:]
        try:
            hvel = fofcat['CMVelocity'].compute()[1:]
        except Exception as e: 
            print(e)
            print('Cannot read velocity')
            hvel = None

        if verbose: print('BigFileCatalog read')
    elif pos is not None:
        halopos = pos
        if mass is not None:
            halomass = mass
        else:
            print('No halo masses given, so mass=1. Scatter and abundance not valid')
            halomass = np.ones(halopos.shape[0])
            sigma = None
            abund = False
    else:
        print('Need either path of Bigfile, or catalogs')
        return None

    if abund:
        print('Not implemented')
        #halomass = mf.icdf_sampling(hmass = halomass, bs = bs, z = z)

    if stellar:
        print('Not implemented')
        #halomass = mf.stellar_mass(halomass)

    if sigma is not None:
        print('Not implemented')
        #print('Scatter catalog with sigma = %0.2f'%sigma)
        #halomass, halopos = dg.scatter_catalog(halomass, halopos, sigma, seed)

        
    halos = {'position':halopos, 'mass':halomass, 'velocity':hvel}
    if pmesh:
        halomesh = pm.paint(halopos, mass = halomass)
        hposmesh = pm.paint(halopos)
        meshdict = {'hposmesh':hposmesh.value, 'halomesh':halomesh.value}

        dkh = halomesh.r2c()        

        if R1 is not None:
            hmR1mesh = ft.smooth(dkh, R1, 'fingauss')
            meshdict['hmR1mesh'] = hmR1mesh.value
        if R2 is not None:
            hmR2mesh = ft.smooth(dkh, R2, 'fingauss')
            meshdict['hmR2mesh'] = hmR2mesh.value
    else:
        meshdict  = None
    return meshdict, halos





def testdata(pm, bs, nc, nsteps, seeds, R1, R2, ftname, local=False, rethalos = False, abund=True, quad=False, z=0, shear=False, 
             doexp=False, mexp=None, cc=None, stellar=False, Rmore=None):
    '''Call create data for different seeds and concatenate them
    '''
    ndim = len(ftname)
    if local:
        fac = 1
    else:
        fac = 27
    tt = np.zeros([1, fac*ndim])
    for seed in seeds:
        print('Read testdata for seed  = ', seed)
        dpath = proj + 'data/z%02d/L%04d_N%04d_S%04d_%02dstep/'%(z*10, bs, nc, seed, nsteps)

        meshdict, halos = readfiles(pm, dpath, R1, R2, abund=abund, quad=quad, z=z, shear=shear, doexp=doexp, mexp=mexp, cc=cc, stellar=stellar, Rmore=Rmore)
        print('Features are - ', ftname)
        ftt = createdata(pm, meshdict, ftname, local)
        tt = np.concatenate([tt, ftt])
    tt = tt[1:]

    if rethalos:
        return meshdict, halos, tt
    else:
        return meshdict, tt


def createdata(pm, meshdict, ftname, local=False):
    '''Create testing data, basically the last part of balanced27gridpt
    '''
    ndim = len(ftname)
    if local:
        fac = 1
    else:
        fac = 27
    ftlist = [meshdict[key] for key in ftname]

    ftt = np.zeros([pm.Nmesh[0]**3, fac*ndim])
    for boo in range(ndim):
        if fac == 27:
            ftt[:, 27*boo: 27*boo + 27] = datalib.read27(ftlist[boo], pm=pm)
        else:
            ftt[:, boo] = ftlist[boo].flatten()
    return ftt


