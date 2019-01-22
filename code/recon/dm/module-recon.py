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
seeds = [100]
R1 = 3
R2 = 3*1.2
kny = np.pi*ncp/bs
kk = tools.fftk((ncp, ncp, ncp), bs)

#############################

tf.reset_default_graph()

suff = 'pad2d8regvtest'
ftname = ['cic']
nchannels = len(ftname)

num_cubes = 100
cube_size = 32
max_offset = ncp - cube_size
pad = 2
cube_sizeft = cube_size + 2*pad


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
    targetmesh = hmesh['pnn']
    #Round off things to 1 again
    targetmesh[targetmesh > 1] = 1
    
    features, target = dtools.randomvoxels(ftlistpad, targetmesh, num_cubes, max_offset,
                                           cube_size, cube_sizeft, seed=seed)
    cube_features = cube_features + features
    cube_target = cube_target + target

#
cube_target = np.stack(cube_target,axis=0).reshape((-1,cube_size,cube_size,cube_size,1))
print(cube_target.sum(), cube_target.size, targetmesh.sum())
print(len(cube_features))
print(cube_features[0].shape)
cube_features = np.stack(cube_features,axis=0).reshape((-1,cube_sizeft,cube_sizeft,cube_sizeft,nchannels))

ind = np.random.randint(cube_target.shape[0])
print('Index = %d'%ind)
recontruth = cube_features[ind:ind+1]
reconmapp = cube_target[ind:ind+1]

##############################

#
tf.reset_default_graph()
sess = tf.Session()
chkname = suff #+'_it%d'%niter


module = hub.Module('./../code/models/n%02d/%s/%s.hub'%(numd*1e4, suff, chkname), trainable=False)
#To predict
xx = tf.placeholder(tf.float32, shape=[None, cube_sizeft, cube_sizeft, cube_sizeft, nchannels], name='input')
yy = tf.placeholder(tf.float32, shape=[None, cube_size, cube_size, cube_size, 1], name='labels')
output = module(dict(input=xx, label=yy, keepprob=1), as_dict=True)['prediction']
#To optimize
xopt = tf.get_variable(name='xopt', shape=[1, cube_sizeft, cube_sizeft, cube_sizeft, nchannels]
                       , initializer=tf.initializers.ones, trainable=True)
outputopt = module(dict(input=xopt, label=yy, keepprob=1))
loss = tf.losses.sigmoid_cross_entropy(yy, outputopt)

lr = tf.placeholder(tf.float32, name='learningrate')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
opt_op = optimizer.minimize(loss, var_list=[xopt])
# opt_op = optimizer.minimize(tf.reduce_sum(output))

###

sess.run(tf.initializers.global_variables())
#
losses = []
niter, nprint = 50000, 1000
lr0, lrfac, nlr = 10, 10, int(10000)
lr0 *= lrfac

print('niter, lr0, lrfac, nlr : ', niter, lr0, lrfac, nlr)
#Save

val0 = sess.run(xopt)

start, curr = time(), time()
for it in range(niter+1):
    _, l = sess.run([opt_op, loss], feed_dict={lr:lr0, yy:reconmapp})
    if it % nlr == 0:
        lr0 /= lrfac
        print('reduce learning rate by factor of %0.2f. New learning rate = %0.2e'%(lrfac, lr0))

    if it % nprint == 0:
        end = time()
        print('Iter %d of %d : Loss= %0.4f\nTime taken for last batch = %0.3f, Total time elapsed = %0.3f'%(it, niter, l, end-curr, end - start))
        curr = end
        #loss
        plt.figure()
        plt.semilogy(losses)
        #plt.loglog()
        plt.savefig('./figs/n%02d/reconloss%s.png'%(numd*1e4, suff))
        plt.close()
    losses.append(l)


recon = sess.run(xopt)
outtruth = sess.run(output, feed_dict={xx:recontruth, yy:reconmapp})
outrecon = sess.run(output, feed_dict={xx:recon, yy:reconmapp})

def getim(ar, axis=0):
    return ar[0, :, :, :, 0].sum(axis=axis)


fig, axar = plt.subplots(3, 3, figsize=(18, 18))

fsize = 16
ax = axar[0]
vmin, vmax = getim(recontruth).min(), getim(recontruth).max()
ax[0].imshow(getim(recontruth), vmin=vmin, vmax=vmax)
ax[1].imshow(getim(recon), vmin=vmin, vmax=vmax)
ax[2].imshow(getim(recontruth-recon), vmin=vmin, vmax=vmax)
ax[0].set_ylabel('Input Density', fontsize=fsize)
ax[0].set_title('Truth', fontsize=fsize)
ax[1].set_title('Output/Recon', fontsize=fsize)
ax[2].set_title('Difference', fontsize=fsize)

ax = axar[1]
vmin, vmax = getim(reconmapp).min(), getim(reconmapp).max()
ax[0].imshow(getim(reconmapp), vmin=vmin, vmax=vmax)
ax[1].imshow(getim(outtruth), vmin=vmin, vmax=vmax)
ax[2].imshow(getim(reconmapp-outtruth), vmin=vmin, vmax=vmax)
ax[0].set_ylabel('Halo (True Point)', fontsize=fsize)

ax = axar[2]
vmin, vmax = getim(reconmapp).min(), getim(reconmapp).max()
ax[0].imshow(getim(reconmapp), vmin=vmin, vmax=vmax)
ax[1].imshow(getim(outrecon), vmin=vmin, vmax=vmax)
ax[2].imshow(getim(reconmapp-outrecon), vmin=vmin, vmax=vmax)
ax[0].set_ylabel('Halo (Recon Point)', fontsize=fsize)

plt.savefig('./figs/n%02d/recon%s.png'%(numd*1e4, suff))
