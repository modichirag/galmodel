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
import tensorflow_hub as hub
from tensorflow.contrib.slim import add_arg_scope
from layers import wide_resnet

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
numd = 1e-3
num = int(numd*bs**3)
seeds = [100, 200, 500, 700]
R1 = 3
R2 = 3*1.2
kny = np.pi*ncp/bs
kk = tools.fftk((ncp, ncp, ncp), bs)

#############################

tf.reset_default_graph()

suff = 'pad2d8regvtest'
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

module = hub.Module('./../code/models/n%02d/%s/%s.hub'%(numd*1e4, suff, chkname))
xx = tf.placeholder(tf.float32, shape=[None, cube_sizeft, cube_sizeft, cube_sizeft, nchannels], name='input')
yy = tf.placeholder(tf.float32, shape=[None, cube_size, cube_size, cube_size, 1], name='labels')
output = module(dict(input=xx, label=yy, keepprob=1), as_dict=True)['prediction']
sess = tf.Session()
sess.run(tf.initializers.global_variables())
#
#############################
meshes = {}
cube_features, cube_target = [], []
for seed in seeds:
    mesh = {}
    partp = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'dynamic/1/Position/')
    mesh['cic'] = tools.paintcic(partp, bs, ncp)
    mesh['decic'] = tools.decic(mesh['cic'], kk, kny)
    mesh['R1'] = tools.fingauss(mesh['cic'], kk, R1, kny)
    mesh['R2'] = tools.fingauss(mesh['cic'], kk, R2, kny)
    mesh['GD'] = mesh['R1'] - mesh['R2']
    mesh['s'] = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'mesh/s/')

    hmesh = {}
    hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]
    hposd = hposall[:num].copy()
    hmesh['pcic'] = tools.paintcic(hposd, bs, nc)
    hmesh['pnn'] = tools.paintnn(hposd, bs, ncp)
    hmesh['target'] = hmesh['pnn'].copy()
    
    print('All the mesh have been generated for seed = %d'%seed)

    #Create training voxels
    ftlist = [mesh[i].copy() for i in ftname]
    ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
    targetmesh = hmesh['target']
    targetmesh[targetmesh > 1] = 1
    
    ncube = int(ncp/cube_size)
    inp = dtools.splitvoxels(ftlistpad, cube_size=cube_sizeft, shift=cube_size, ncube=ncube)
    yinp = dtools.splitvoxels(targetmesh, cube_size=cube_size, shift=cube_size, ncube=ncube)
    recp = sess.run(output, feed_dict={xx:inp, yy:yinp})
    mesh['predict'] = dtools.uncubify(recp[:,:,:,:,0], shape)
    
    meshes[seed] = [mesh, hmesh]
    print('All the predictions have been generated for seed = %d'%seed)

##############################
##Power spectrum

kk = tools.fftk(shape, bs)
kmesh = sum(i**2 for i in kk)**0.5


fig, ax = plt.subplots(1, 2, figsize = (10, 4))
for seed in seeds:
    predict, hpmeshd = meshes[seed][0]['predict'], meshes[seed][1]['target'], 
    k, pkpred = tools.power(predict/predict.mean(), boxsize=bs, k=kmesh)
    k, pkhd = tools.power(hpmeshd/hpmeshd.mean(), boxsize=bs, k=kmesh)
    k, pkhx = tools.power(hpmeshd/hpmeshd.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)
    #k, pkpredall = tools.power(predictall/predictall.mean(), boxsize=bs, k=kmesh)
    #k, pkhallx = tools.power(hpmeshd/hpmeshd.mean(), predictall/predictall.mean(), boxsize=bs, k=kmesh)  
    ##
    ax[0].semilogx(k, pkpred/pkhd, label=seed)
    ax[1].semilogx(k, pkhx/(pkpred*pkhd)**0.5)
    # plt.plot(k, pkpredall/pkhd)
    
ax[0].legend(fontsize=14)
ax[0].set_title('Trasnfer function', fontsize=14)
ax[1].set_title('Cross correlation', fontsize=14)
for axis in ax: axis.set_ylim(0., 1.1)
for axis in ax: axis.set_yticks(np.arange(0, 1.1, 0.1))
for axis in ax: axis.grid(which='both')

plt.savefig('./figs/n%02d/2ptpredict%s.png'%(numd*1e4, suff))

fig, ax = plt.subplots(1, 3, figsize = (14, 4))
ax[0].imshow(meshes[seed][0]['cic'].sum(axis=0))
ax[0].set_title('Density', fontsize=14)

vmin, vmax = meshes[seed][1]['target'].sum(axis=0).min(), meshes[seed][1]['target'].sum(axis=0).max()
ax[1].imshow(meshes[seed][1]['target'].sum(axis=0), vmin=vmin, vmax=vmax)
ax[1].set_title('Halos', fontsize=14)
ax[2].imshow(meshes[seed][0]['predict'].sum(axis=0), vmin=vmin, vmax=vmax)
ax[2].set_title('Predict', fontsize=14)
plt.savefig('./figs/n%02d/impredict%s.png'%(numd*1e4, suff))


