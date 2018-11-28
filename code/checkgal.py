import numpy as np
import matplotlib.pyplot as plt
#
import sys, os
sys.path.append('./utils/')
import tools
import datalib as dlib
import datatools as dtools
from time import time
#
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import add_arg_scope
from layers import wide_resnet
import tensorflow_probability
tfd = tensorflow_probability.distributions

#############################
seed_in = 5
from numpy.random import seed
seed(seed_in)
from tensorflow import set_random_seed
set_random_seed(seed_in)

bs = 400
nc, ncf = 128, 512
ncp = 128
shape = (ncp, ncp, ncp)
step, stepf = 5, 40

path = './../data/z00/'
ftype = 'L%04d_N%04d_S%04d_%02dstep/'
numd = 5e-4
num = int(numd*bs**3)
tseeds = [100, 200, 500, 700]
R1 = 3
R2 = 3*1.2
kny = np.pi*ncp/bs
kk = tools.fftk((ncp, ncp, ncp), bs)

#############################

tf.reset_default_graph()

suff = 'pad2d9wt0v2'
ftname = ['cic']
nchannels = len(ftname)

cube_size = 32
max_offset = ncp - cube_size
pad = 2
cube_sizeft = cube_size + 2*pad

#
niter = 9000
sess = tf.Session()
chkname = suff #+'_it%d'%niter

saver = tf.train.import_meta_graph('./../code/models/gal%02d/%s/%s.meta'%(numd*1e4, suff, chkname))
saver.restore(sess,'./../code/models/gal%02d/%s/%s'%(numd*1e4, suff, chkname))
g = sess.graph
prediction = g.get_tensor_by_name('prediction:0')
input = g.get_tensor_by_name('input:0')
keepprob = g.get_tensor_by_name('keepprob:0')
rate = g.get_tensor_by_name('rate:0')
pdf = tfd.Poisson(rate=rate)
samplesat = pdf.sample()

#############################
meshes = {}
cube_features, cube_target = [], []
for seed in tseeds:
    mesh = {}
    partp = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'dynamic/1/Position/')
    mesh['cic'] = tools.paintcic(partp, bs, ncp)
    #mesh['decic'] = tools.decic(mesh['cic'], kk, kny)
    mesh['R1'] = tools.fingauss(mesh['cic'], kk, R1, kny)
    #mesh['R2'] = tools.fingauss(mesh['cic'], kk, R2, kny)
    #mesh['GD'] = mesh['R1'] - mesh['R2']

    hmesh = {}
    hpath = path + ftype%(bs, ncf, seed, stepf) + 'galaxies_n05/galcat/'
    hposd = tools.readbigfile(hpath + 'Position/')
    massd = tools.readbigfile(hpath + 'Mass/').reshape(-1)*1e10
    galtype = tools.readbigfile(hpath + 'gal_type/').reshape(-1).astype(bool)
    #hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]    
    #hposd = hposall[:num].copy()
    #massd = massall[:num].copy()
    #hmesh['pcic'] = tools.paintcic(hposd, bs, nc)
    hmesh['pnn'] = tools.paintnn(hposd, bs, ncp)
    #hmesh['mnn'] = tools.paintnn(hposd, bs, ncp, massd)
    hmesh['pnnsat'] = tools.paintnn(hposd[galtype], bs, ncp)
    hmesh['pnncen'] = tools.paintnn(hposd[~galtype], bs, ncp)

    print('All the mesh have been generated for seed = %d'%seed)


  #Create training voxels
    ftlist = [mesh[i].copy() for i in ftname]
    ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
    targetmesh = [hmesh['pnncen'], hmesh['pnnsat']]
    ntarget = len(targetmesh)

    ncube = int(ncp/cube_size)
    inp = dtools.splitvoxels(ftlistpad, cube_size=cube_sizeft, shift=cube_size, ncube=ncube)
    satmesh,cenmesh, rates = sess.run([samplesat, prediction, rate], feed_dict={input:inp, keepprob:1})
    mesh['predictsat'] = dtools.uncubify(satmesh[:,:,:,:,0], [nc,nc,nc])
    mesh['predictcen'] = dtools.uncubify(cenmesh[:,:,:,:,0], [nc,nc,nc])
    mesh['predict'] = mesh['predictcen'] + mesh['predictsat']
    mesh['rates'] =  dtools.uncubify(rates[:,:,:,:,0], [nc,nc,nc])
    meshes[seed] = [mesh, hmesh]    
    print('Number of predicted & true satellites = ', satmesh.sum(), hmesh['pnnsat'].sum())
    
##############################
##Power spectrum

kk = tools.fftk(shape, bs)
kmesh = sum(i**2 for i in kk)**0.5


fig, ax = plt.subplots(2, 3, figsize = (12, 8))
for seed in tseeds:
    for i, key in enumerate(['', 'cen', 'sat']):
        predict, hpmeshd = meshes[seed][0]['predict%s'%key] , meshes[seed][1]['pnn%s'%key], 
        k, pkpred = tools.power(predict/predict.mean(), boxsize=bs, k=kmesh)
        k, pkhd = tools.power(hpmeshd/hpmeshd.mean(), boxsize=bs, k=kmesh)
        k, pkhx = tools.power(hpmeshd/hpmeshd.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)    
    ##
        ax[0, i].semilogx(k[1:], pkpred[1:]/pkhd[1:], label=seed)
        ax[1, i].semilogx(k[1:], pkhx[1:]/(pkpred[1:]*pkhd[1:])**0.5)
        ax[0, i].set_title(key, fontsize=12)
#    mask = meshes[seed][0]['predictcen']
#    predict *= mask
#    k, pkpred = tools.power(predict/predict.mean(), boxsize=bs, k=kmesh)
#    k, pkhx = tools.power(hpmeshd/hpmeshd.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)    
#    ##
#    ax[0, i].semilogx(k[1:], pkpred[1:]/pkhd[1:], label=seed, ls="--")
#    ax[1, i].semilogx(k[1:], pkhx[1:]/(pkpred[1:]*pkhd[1:])**0.5, ls="--")
#    
        
for axis in ax.flatten():
    axis.legend(fontsize=14)
    axis.set_yticks(np.arange(0, 1.1, 0.1))
    axis.grid(which='both')
    axis.set_ylim(0.,1.1)
ax[0, 0].set_title('All Gal', fontsize=15)
ax[0, 0].set_ylabel('Transfer function', fontsize=14)
ax[1, 0].set_ylabel('Cross correlation', fontsize=14)
plt.savefig('./figs/gal%02d/2ptpredict%s.png'%(numd*1e4, suff))

fig, ax = plt.subplots(2, 3, figsize=(12,8))

for i, key in enumerate(['', 'cen', 'sat']):
    predict, hpmeshd = meshes[seed][0]['predict%s'%key] , meshes[seed][1]['pnn%s'%key], 
    vmin, vmax = 0, (hpmeshd[:, :, :].sum(axis=0)).max()
    im = ax[0, i].imshow(predict[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
#     plt.colorbar(im, ax=ax[0, i])
    im = ax[1, i].imshow(hpmeshd[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
#     plt.colorbar(im, ax=ax[1, i])
    ax[0, i].set_title(key, fontsize=15)
ax[0, 0].set_title('All Gal', fontsize=15)
ax[0, 0].set_ylabel('Prediction', fontsize=15)
ax[1, 0].set_ylabel('Truth', fontsize=15)
#plt.show()
plt.savefig('./figs/gal%02d/impredict%s.png'%(numd*1e4, suff))

##
plt.figure()
for seed in tseeds:
    rates = meshes[seed][0]['rates']
    plt.hist(rates.flatten(), range=(1e-3, 10), histtype='step', label=seed, log=True)
plt.legend(fontsize=14)
plt.savefig('./figs/gal%02d/allrates%s.png'%(numd*1e4, suff))
#
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
vmin, vmax = 0, meshes[seed][1]['pnnsat'].sum(axis=0).max()
im = ax[0].imshow(meshes[seed][1]['pnnsat'].sum(axis=0), vmin=vmin, vmax=vmax)
ax[0].set_title('Data')
im = ax[1].imshow(meshes[seed][0]['predictsat'].sum(axis=0), vmin=vmin, vmax=vmax)
ax[1].set_title('Prediction')
im = ax[2].imshow(meshes[seed][0]['rates'].sum(axis=0), vmin=vmin, vmax=vmax)
ax[2].set_title('Rates')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('./figs/gal%02d/satpred%s.png'%(numd*1e4, suff))

##
fig, ax = plt.subplots(2, 3, figsize = (12, 8))
seed = 100
for n, seed in enumerate([100, 500]):
    for i in range(6):
        axis = ax.flatten()[i]
        mask = np.where(meshes[seed][1]['pnnsat'] == i)
        if i == 5: mask = np.where(meshes[seed][1]['pnnsat'] >= i)
        rates = meshes[seed][0]['rates'] 
        predict = meshes[seed][0]['predictsat'] 
        axis.hist(rates[mask].flatten(), range=(-1e-3, 10), histtype='step',
                  label=seed, log=True, color='C%d'%n, ls="--", lw=2, alpha=0.7)
        axis.hist(predict[mask].flatten(), range=(-1e-3, 10), histtype='step',
                  label='Count', log=True, color='C%d'%(n+4), ls="-", lw=2, alpha=0.8)
        axis.set_title('Nsat=%d, Tot=%d'%(i, mask[0].shape[0]))
plt.legend(fontsize=14)
plt.savefig('./figs/gal%02d/histsat%s.png'%(numd*1e4, suff))
