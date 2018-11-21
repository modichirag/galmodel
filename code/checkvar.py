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
seeds = [200, 500]
R1 = 3
R2 = 3*1.2
kny = np.pi*ncp/bs
kk = tools.fftk((ncp, ncp, ncp), bs)

#############################

tf.reset_default_graph()

suff = 'pad2d8varsizev2'
ftname = ['cic']
nchannels = len(ftname)

test_sizes = [8, 16, 32, 64, 128]
pad = 2

#
niter = 9000
sess = tf.Session()
chkname = suff #+'_it%d'%niter

saver = tf.train.import_meta_graph('./../code/models/n%02d/%s/%s.meta'%(numd*1e4, suff, chkname))
saver.restore(sess,'./../code/models/n%02d/%s/%s'%(numd*1e4, suff, chkname))
g = sess.graph
prediction = g.get_tensor_by_name('prediction:0')
input = g.get_tensor_by_name('input:0')
keepprob = g.get_tensor_by_name('keepprob:0')

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
    
    for size in test_sizes:
        ncube = int(ncp/size)
        print(size, ncube)
        inp = dtools.splitvoxels(ftlistpad, cube_size=size+2*pad, shift=size, ncube=ncube)
        recp = sess.run(prediction, feed_dict={input:inp, keepprob:1})
        mesh['predict%03d'%size] = dtools.uncubify(recp[:,:,:,:,0], shape)
    
    meshes[seed] = [mesh, hmesh]
    print('Prediction done for seed = %d'%seed)

##############################
##Power spectrum

kk = tools.fftk(shape, bs)
kmesh = sum(i**2 for i in kk)**0.5


lss = ['-', '--', ':', '-.']
fig, ax = plt.subplots(1, 2, figsize = (10, 4))
for ss, seed in enumerate(seeds):
    hpmeshd = meshes[seed][1]['target'] 
    k, pkhd = tools.power(hpmeshd/hpmeshd.mean(), boxsize=bs, k=kmesh)
    for i, size in enumerate(test_sizes):
        predict  = meshes[seed][0]['predict%03d'%size]
        k, pkpred = tools.power(predict/predict.mean(), boxsize=bs, k=kmesh)
        k, pkhx = tools.power(hpmeshd/hpmeshd.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)
        ##
        if ss==0: ax[0].semilogx(k, pkpred/pkhd, label=size, color='C%d'%i, ls=lss[ss], alpha=1-0.1*i)
        else: ax[0].semilogx(k, pkpred/pkhd, color='C%d'%i, ls=lss[ss], alpha=1-0.1*i)

        if i==0: ax[1].semilogx(k, pkhx/(pkpred*pkhd)**0.5, label=seed, color='C%d'%i, ls=lss[ss], alpha=1-0.1*i)
        else: ax[1].semilogx(k, pkhx/(pkpred*pkhd)**0.5, color='C%d'%i, ls=lss[ss], alpha=1-0.1*i)
    # plt.plot(k, pkpredall/pkhd)
    
ax[0].legend(fontsize=14)
ax[1].legend(fontsize=14)
ax[0].set_title('Trasnfer function', fontsize=14)
ax[1].set_title('Cross correlation', fontsize=14)
for axis in ax: axis.set_ylim(0., 1.1)
for axis in ax: axis.set_yticks(np.arange(0, 1.1, 0.1))
for axis in ax: axis.grid(which='both')
plt.savefig('./figs/n%02d/2ptpredict%s.png'%(numd*1e4, suff))

##
fig, ax = plt.subplots(2, 3, figsize = (14, 8))
ax[0, 0].imshow(meshes[seed][1]['target'].sum(axis=0))
ax[0, 0].set_title('Halos', fontsize=14)
for i, size in enumerate(test_sizes):
    axis = ax.flatten()[i+1]
    axis.imshow(meshes[seed][0]['predict%03d'%size].sum(axis=0))
    axis.set_title('Predict Size = %d'%size, fontsize=14)
plt.savefig('./figs/n%02d/impredict%s.png'%(numd*1e4, suff))


