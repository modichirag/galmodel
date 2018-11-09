import numpy as np
from nettools import readfiles, gridhalos
from pmesh.pm import ParticleMesh
from nbodykit.source.catalog import BigFileCatalog
from nbodykit.lab import BigFileMesh

import sys, os

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from time import time
#from itertools import product as iprod

from ruamel.yaml import YAML
yaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
import ruamel
import warnings
warnings.simplefilter('ignore', ruamel.yaml.error.MantissaNoDotYAML1_1Warning)



######################################################################################################################

if __name__ == '__main__':
   
    cfname = sys.argv[1]

    with open(cfname, 'r') as ymlfile:
        ddict = yaml.load(ymlfile)

    #bs, nc, zz = ddict['bs'], ddict['nc'], ddict['zz']
    for i in ['bs', 'nc', 'zz', 'nsteps', 'nfsteps', 'numd', 'fine', 'R1', 'kmin', 'kmax']: 
        locals()[i] = ddict[i]
    R2 = R1 * ddict['sfac']

    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc], dtype = 'f8')

    for key in ddict: print(key, ' = ', ddict[key])

    ###########

    #for seed in ddict['seeds']:
    #    pass
    seed = 100
    proj = '/project/projectdirs/astro250/chmodi/cosmo4d/'
    dpath = proj + 'data/z%02d/L%04d_N%04d_S%04d_%02dstep/'%(zz*10, bs, nc, seed, 5)
    dpathf = proj + 'data/z%02d/L%04d_N%04d_S%04d_%02dstep/'%(zz*10, bs, fine*nc, seed, 40)
    meshdict, halos = readfiles(pm, dpath, R1, R2, abund=False, quad=False, z=zz, shear=False)
    halosf = gridhalos(pm, dpath=dpathf, rank = None, abund=False, sigma=False, seed=seed, pmesh = True, z=zz)

    print('Keys in meshdict ',meshdict.keys())
    print('Keys in halos ', halos.keys())
    print('Length of list halosf ', len(halosf))
    print('Keys in halosf[0] ',halosf[0].keys())
    print('Keys in halosf[1] ',halosf[1].keys())

