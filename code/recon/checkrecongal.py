import numpy as np

import sys, os
sys.path.append('../utils/')
import tools
import matplotlib.pyplot as plt


bs, nc = 400, 128
seed = 100
numd = 1e-3
ofolder = './saved/L%04d_N%04d_S%04d_n%02d/anneal4/nc0norm-truth/'%(bs, nc, seed, numd*1e4)
figfolder = ofolder + 'figs/'
try: os.makedirs(figfolder)
except: pass

truemesh = np.load(ofolder + 'truth.f4.npy')
k, pt = tools.power(1+truemesh, boxsize=bs)

fig, ax = plt.subplots(1, 2, figsize = (9, 4), sharex=True)

#iters = [0, 100, 200, 300, 500]
iters = [1000, 600, 700, 800, 900]
subf = 'R20'
reconfile = '/%s/iter1000.f4.npy'%subf
nit = 1000


for j, it in enumerate(iters):
    reconmesh = np.load(ofolder + '/%s/iter%d.f4.npy'%(subf, it))
    k, pr = tools.power(1+reconmesh, boxsize=bs)
    k, px = tools.power(1+truemesh, 1+reconmesh, boxsize=bs)
    ax[0].plot(k, px/(pr*pt)**.5, 'C%d'%j, label=it)
    ax[1].plot(k, pr/pt, 'C%d'%j)
reconmesh = np.load(ofolder + '/%s/recon2.f4.npy'%subf)
k, pr = tools.power(1+reconmesh, boxsize=bs)
k, px = tools.power(1+truemesh, 1+reconmesh, boxsize=bs)
ax[0].plot(k, px/(pr*pt)**.5, 'C%d--'%j, label='init')
ax[1].plot(k, pr/pt, 'C%d'%j)

ax[0].legend()
ax[0].set_xscale('log')
ax[0].set_title('Correlation')
ax[1].set_title('Transfer (power)')
for axis in ax:
    axis.grid(which='both')
    axis.set_ylim(0.0, 1.2)

fig.savefig(figfolder + '2ptrecon-%s.png'%subf)



#########


truemesh = np.load(ofolder + 'truth.f4.npy')
reconmesh = np.load(ofolder + reconfile)

fig, ax = plt.subplots(1, 3, figsize = (9, 4), sharex=True)

fig, ax = plt.subplots(1, 3, figsize = (13, 4))
im = ax[0].imshow(truemesh.sum(axis=0))
plt.colorbar(im, ax=ax[0])
ax[0].set_title('Truth')
im = ax[1].imshow(reconmesh.sum(axis=0))
plt.colorbar(im, ax=ax[1])
ax[1].set_title('Recon it-%d'%nit)
im = ax[2].imshow((truemesh-reconmesh).sum(axis=0))
plt.colorbar(im, ax=ax[2])
ax[2].set_title('Diff')

fig.savefig(figfolder + 'imrecon-%s.png'%subf)
