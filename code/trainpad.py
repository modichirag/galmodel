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
seed_in = 3
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
numd = 1e-3
num = int(numd*bs**3)
R1 = 3
R2 = 3*1.2
kny = np.pi*ncp/bs
kk = tools.fftk((ncp, ncp, ncp), bs)
seeds = [100, 200, 300, 400]
rprob = 0.

#############################

suff = 'pad4v2'
num_cubes= 1000
cube_size = 32
pad = 4
cube_sizeft = cube_size + 2*pad
max_offset = ncp - cube_size
ftname = ['cic']
nchannels = len(ftname)

#############################
##Read data and generate meshes
#mesh = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'mesh/d/')
meshes = {}
cube_features, cube_target = [], []
for seed in seeds:
    mesh = {}
    partp = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'dynamic/1/Position/')
    mesh['cic'] = tools.paintcic(partp, bs, ncp)
    #mesh['decic'] = tools.decic(mesh['cic'], kk, kny)
    mesh['R1'] = tools.fingauss(mesh['cic'], kk, R1, kny)
    mesh['R2'] = tools.fingauss(mesh['cic'], kk, R2, kny)
    mesh['GD'] = mesh['R1'] - mesh['R2']

    hmesh = {}
    hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]
    massall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/Mass/')[1:].reshape(-1)*1e10
    hposd = hposall[:num].copy()
    massd = massall[:num].copy()
    hmesh['pcic'] = tools.paintcic(hposd, bs, nc)
    hmesh['pnn'] = tools.paintnn(hposd, bs, ncp)
    hmesh['mcic'] = tools.paintcic(hposd, bs, nc, mass=massd)
    hmesh['mnn'] = tools.paintnn(hposd, bs, ncp, mass=massd)

    meshes[seed] = [mesh, hmesh]

    print('All the mesh have been generated for seed = %d'%seed)

    #Create training voxels
    ftlist = [mesh[i].copy() for i in ftname]
    ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
    targetmesh = hmesh['pnn']
    #Round off things to 1 again
    targetmesh[targetmesh > 1] = 1
    
    features, target = dtools.randomvoxels(ftlistpad, targetmesh, num_cubes, max_offset, cube_size, cube_sizeft, rprob=rprob)
    cube_features = cube_features + features
    cube_target = cube_target + target

#
cube_target = np.stack(cube_target,axis=0).reshape((-1,cube_size,cube_size,cube_size,1))
print(cube_target.sum(), cube_target.size, targetmesh.sum())
print(len(cube_features))
print(cube_features[0].shape)
cube_features = np.stack(cube_features,axis=0).reshape((-1,cube_sizeft,cube_sizeft,cube_sizeft,nchannels))
trainingsize = cube_features.shape[0]
print('Training size is = ', trainingsize)

#Save a snapshot of features
fig, ax = plt.subplots(1, nchannels+1, figsize = (nchannels*4+4, 5))
n = 10
for i in range(nchannels):
    ax[i].imshow(cube_features[n][:,:,:,i].sum(axis=0))
    ax[i].set_title(ftname[i])
ax[-1].imshow(cube_target[n][:,:,:,0].sum(axis=0))
ax[-1].set_title('Target')
plt.savefig('./figs/n%02d/features%s.png'%(numd*1e4, suff))

#############################
### Model

x = tf.placeholder(tf.float32, shape=[None, cube_sizeft, cube_sizeft, cube_sizeft, nchannels], name='input')
y = tf.placeholder(tf.float32, shape=[None, cube_size, cube_size, cube_size, 1], name='labels')
lr = tf.placeholder(tf.float32, name='learningrate')
keepprob = tf.placeholder(tf.float32, name='keepprob')
#
wregwt, bregwt = 0.001, 0.001
wreg = slim.regularizers.l2_regularizer(wregwt)
breg = slim.regularizers.l2_regularizer(bregwt)
#
net = slim.conv3d(x, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid', weights_regularizer=wreg, biases_regularizer=breg)
drop = tf.nn.dropout(net, keep_prob=keepprob)
net = slim.conv3d(drop, 64, 5, activation_fn=tf.nn.leaky_relu, padding='valid', weights_regularizer=wreg, biases_regularizer=breg)
drop = tf.nn.dropout(net, keep_prob=keepprob)
#net = slim.conv3d(drop, 128, 3, activation_fn=tf.nn.leaky_relu, weights_regularizer=wreg, biases_regularizer=breg)
#drop = tf.nn.dropout(net, keep_prob=keepprob)
#net = slim.conv3d(drop, 64, 3, activation_fn=tf.nn.leaky_relu, weights_regularizer=wreg, biases_regularizer=breg)
#drop = tf.nn.dropout(net, keep_prob=keepprob)
net = slim.conv3d(drop, 16, 3, activation_fn=tf.nn.leaky_relu, weights_regularizer=wreg, biases_regularizer=breg)
drop = tf.nn.dropout(net, keep_prob=keepprob)
net = slim.conv3d(drop, 1, 3, activation_fn=None, weights_regularizer=wreg, biases_regularizer=breg)
drop = tf.nn.dropout(net, keep_prob=keepprob)
pred = tf.nn.sigmoid(drop, name='prediction')

loss = tf.losses.sigmoid_cross_entropy(y, net)
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='optimizer')

opt_op = optimizer.minimize(loss, name='minimize')


#############################
###Train

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses = []

niter= 1000
nprint = 100
batch_size=32
#
start = time()
curr = time()

lr0, lrfac, nlr = 0.001, 10, int(500)
lr0 *= lrfac
kprob = 1.0

#Save
saver = tf.train.Saver()

for it in range(niter):
    inds = np.random.choice(int(trainingsize), batch_size, replace=False)
    _, l = sess.run([opt_op, loss], feed_dict={lr:lr0, keepprob:kprob, x:cube_features[inds], y:cube_target[inds]})
    if it % nlr == 0:
        lr0 /= lrfac
        print('reduce learning rate by factor of %0.2f. New learning rate = %0.2e'%(lrfac, lr0))
        #Save loss
        plt.figure()
        plt.semilogy(losses)
        #plt.loglog()
        plt.savefig('./figs/n%02d/loss%s.png'%(numd*1e4, suff))
        plt.close()
        saver.save(sess, './models/n%02d/%s/%s_it%d'%(numd*1e4, suff, suff, it))

    if it % nprint == 0:
        print('Iteration %d of %d'%(it, niter), '\nLoss = ', l)
        end = time()
        print('Time taken for last batch = %0.3f, Total time elapsed = %0.3f'%(end-curr, end - start))
        curr = end
    losses.append(l)

print('Time taken for training = ', end - start)
saver.save(sess, './models/n%02d/%s/%s'%(numd*1e4, suff, suff))




ind = 0
input = cube_features[ind]
recp = sess.run(pred, feed_dict={x:input.reshape(1, *input.shape), keepprob:1})

fig, ax = plt.subplots(1, 3, figsize=(15,7))
ax[0].imshow(recp[ind,0,:,:,0]);
ax[0].set_title('predict')
ax[1].imshow(cube_target[ind,0,:,:,0]);
ax[1].set_title('target')
ax[2].imshow(cube_features[ind,0,:,:,0]);
ax[2].set_title('features')
plt.savefig('./figs/n%02d/pred%s.png'%(numd*1e4, suff))

