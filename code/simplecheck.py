import numpy as np
import matplotlib.pyplot as plt
#
import sys
sys.path.append('./utils/')
import tools
import datalib as dlib
import datatools as dtools
from time import time
#
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import add_arg_scope

#############################
seed_in = 5
from numpy.random import seed
seed(seed_in)
from tensorflow import set_random_seed
set_random_seed(seed_in)

bs = 400
nc, ncf = 128, 512
ncp = 128
step, stepf = 5, 40

path = './../data/z00/'
ftype = 'L%04d_N%04d_S%04d_%02dstep/'
numd = 5e-4
num = int(numd*bs**3)
seed= 100
R1 = 3
R2 = 3*1.2
kny = np.pi*ncp/bs
kk = tools.fftk((ncp, ncp, ncp), bs)


#############################
#mesh = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'mesh/d/')
partpos = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'dynamic/1/Position/')
mesh = tools.paintcic(partpos, bs, ncp)
meshdecic = tools.decic(mesh, kk, kny)
meshR1 = tools.fingauss(mesh, kk, R1, kny)
meshR2 = tools.fingauss(mesh, kk, R2, kny)
meshdg = meshR1 - meshR2


hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]
hposd = hposall[:num].copy()
hposall = hposall[:2*num]

# hpmeshall = tools.paintcic(hposall, bs, ncp)
hpmeshcic = tools.paintcic(hposd, bs, ncp)
# clsgrid = hpmeshd.copy()
hpmeshall = tools.paintnn(hposall, bs, ncp)
hpmeshd = tools.paintnn(hposd, bs, ncp)
clsgrid = hpmeshd.copy()


#############################

tf.reset_default_graph()

suff = '-pad2'
ftlist = [mesh.copy()]
cube_size = 32
max_offset = ncp - cube_size
pad = 2
cube_sizep = cube_size + 2*pad
ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]

sess = tf.Session()
modname = 'test%s'%suff
saver = tf.train.import_meta_graph('./../code/models/n%02d/%s.meta'%(numd*1e4, modname))
saver.restore(sess,'./../code/models/n%02d/%s'%(numd*1e4, modname))
g = sess.graph
#print(g.get_operations())
prediction = g.get_tensor_by_name('prediction:0')
input = g.get_tensor_by_name('input:0')


#############################
##Predictions

predictall = np.zeros_like(mesh)
ncube = int(ncp/cube_size)

for i in range(ncube):
    for j in range(ncube):
        for k in range(ncube):
            x1, y1, z1 = i*cube_size, j*cube_size, k*cube_size
            x2, y2, z2 = x1+cube_sizep, y1+cube_sizep, z1+cube_sizep

            fts = (ar[x1:x2, y1:y2, z1:z2].copy() for ar in ftlistpad)
            inp = np.stack(fts, axis=-1)

            recp = sess.run(prediction, feed_dict={input:inp.reshape(1, *inp.shape)})

            x2, y2, z2 = x1+cube_size, y1+cube_size, z1+cube_size
            predictall[x1:x2, y1:y2, z1:z2] = recp[0, :, :, :, 0]

####
if pad == 0:
    shift = int(cube_size/2)
    predict = np.zeros_like(mesh)
    nshift = int(ncp/shift)


    start = time()
    for i in range(nshift):
        for j in range(nshift):
            for k in range(nshift):
                x1, y1, z1 = i*shift-shift//2, j*shift-shift//2, k*shift-shift//2
                x2, y2, z2 = x1+cube_size, y1+cube_size, z1+cube_size

                fts = (tools.readperiodic(ar.copy(), [[x1, x2], [y1, y2], [z1, z2]]) for ar in ftlist)
                inp = np.stack(fts, axis=-1)
                #recp = sess.run(prediction, feed_dict={input:inp.reshape(1, *inp.shape)})

                x1, y1, z1 = i*shift, j*shift, k*shift
                x2, y2, z2 = x1+shift, y1+shift, z1+shift
                #predict[x1:x2, y1:y2, z1:z2] = recp[0, shift//2:-shift//2, shift//2:-shift//2, shift//2:-shift//2, 0]

    predictsave  = predict.copy()
    end = time()
    print('Time taken for prediction = ', end-start)


    start = time()
    for i in range(nshift):
        for j in range(nshift):
            for k in range(nshift):
                x1, y1, z1 = i*shift-shift//2, j*shift-shift//2, k*shift-shift//2
                x2, y2, z2 = x1+cube_size, y1+cube_size, z1+cube_size

                fts = (tools.readperiodic(ar.copy(), [[x1, x2], [y1, y2], [z1, z2]]) for ar in ftlist)
                inp = np.stack(fts, axis=-1)
                recp = sess.run(prediction, feed_dict={input:inp.reshape(1, *inp.shape)})

                x1, y1, z1 = i*shift, j*shift, k*shift
                x2, y2, z2 = x1+shift, y1+shift, z1+shift
                predict[x1:x2, y1:y2, z1:z2] = recp[0, shift//2:-shift//2, shift//2:-shift//2, shift//2:-shift//2, 0]

    predictsave  = predict.copy()
    end = time()
    print('Time taken for prediction = ', end-start)

else:predict = predictall.copy()
#############################
##Power spectrum

kk = tools.fftk(mesh.shape, bs)
kmesh = sum(i**2 for i in kk)**0.5

k, pkpred = tools.power(predict/predict.mean(), boxsize=bs, k=kmesh)
k, pkpredall = tools.power(predictall/predictall.mean(), boxsize=bs, k=kmesh)
k, pkhd = tools.power(hpmeshd/hpmeshd.mean(), boxsize=bs, k=kmesh)
k, pkhx = tools.power(hpmeshd/hpmeshd.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)
k, pkhallx = tools.power(hpmeshd/hpmeshd.mean(), predictall/predictall.mean(), boxsize=bs, k=kmesh)
# k, pkhd = tools.power(hpmeshall/hpmeshall.mean(), boxsize=bs, k=kmesh)
# k, pkhx = tools.power(hpmeshall/hpmeshall.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)


fig, ax = plt.subplots(1, 3, figsize = (14, 4))
ax[0].imshow(mesh.sum(axis=0))
ax[0].set_title('Density', fontsize=14)
ax[1].imshow(hpmeshd.sum(axis=0))
ax[1].set_title('Halos', fontsize=14)
ax[2].imshow(predict.sum(axis=0))
ax[2].set_title('Predict', fontsize=14)
plt.savefig('./figs/n%02d/impredict%s.png'%(numd*1e4, suff))


##

fig, ax = plt.subplots(1, 2, figsize = (10, 4))
ax[0].semilogx(k, pkpred/pkhd)
ax[1].semilogx(k, pkhx/(pkpred*pkhd)**0.5)
# plt.plot(k, pkpredall/pkhd)
ax[0].grid(which='both')
ax[1].grid(which='both')
ax[0].set_title('Trasnfer function', fontsize=14)
ax[1].set_title('Cross correlation', fontsize=14)
plt.savefig('./figs/n%02d/2ptpredict%s.png'%(numd*1e4, suff))
