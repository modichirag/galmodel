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
seed= 100
R1 = 3
R2 = 3*1.2
kny = np.pi*ncp/bs
kk = tools.fftk((ncp, ncp, ncp), bs)

#############################
##Read data and generate meshes
#mesh = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'mesh/d/')
partp = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'dynamic/1/Position/')
mesh = tools.paintcic(partp, bs, ncp)
meshdecic = tools.decic(mesh, kk, kny)
meshR1 = tools.fingauss(mesh, kk, R1, kny)
meshR2 = tools.fingauss(mesh, kk, R2, kny)
meshdg = meshR1 - meshR2
#ftlist = [meshdecic.copy(), meshR1.copy(), meshdg.copy()]


hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]
hposd = hposall[:num].copy()
#hposall = hposall[:2*num]

# hpmeshall = tools.paintcic(hposall, bs, nc)
# hpmeshd = tools.paintcic(hposd, bs, nc)

#hpmeshall = tools.paintnn(hposall, bs, ncp)
hpmeshd = tools.paintnn(hposd, bs, ncp)

print('All the mesh have been generated')
#############################

#Create training voxels
suff = '-pad2'
num_cubes=2000
cube_size = 32
pad = 2
cube_sizep = cube_size + 2*pad
max_offset = ncp - cube_size
ftlist = [mesh.copy()]
ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
ftname = ['density', 'GD']
nchannels = len(ftlist)
cube_features = []
cube_target = []

rand = np.random.rand
for it in range(num_cubes):
    # Extract random cubes from the sim
    offset_x = round(rand()*max_offset)
    offset_y = round(rand()*max_offset)
    offset_z = round(rand()*max_offset)
    x1, x2, x2p = offset_x, offset_x+cube_size, offset_x+cube_sizep
    y1, y2, y2p = offset_y, offset_y+cube_size, offset_y+cube_sizep
    z1, z2, z2p = offset_z, offset_z+cube_size, offset_z+cube_sizep
        
    features = []
    for i in range(nchannels): features.append(ftlistpad[i][x1:x2p, y1:y2p, z1:z2p])
    features = np.stack(features, axis=-1)
    cube_features.append(features)
    cube_target.append((hpmeshd[x1:x2, y1:y2, z1:z2]))
    

cube_target = np.stack(cube_target,axis=0).reshape((-1,cube_size,cube_size,cube_size,1))
print(cube_target.sum(), cube_target.size, hpmeshd.sum())
print(len(cube_features))
print(cube_features[0].shape)
cube_features = np.stack(cube_features,axis=0).reshape((-1,cube_sizep,cube_sizep,cube_sizep,nchannels))


#Save a snapshot of features
fig, ax = plt.subplots(1, nchannels+1, figsize = (nchannels*4+4, 5))
n = 10
for i in range(nchannels):
    ax[i].imshow(cube_features[n][:,:,:,i].sum(axis=0))
    ax[i].set_title(ftname[i])
ax[-1].imshow(cube_target[n][:,:,:,0].sum(axis=0))
ax[-1].set_title('Target')
plt.savefig('./figs/features%s.png'%suff)

#############################
### Model

# Implement a simple masked CNN layer
# Using the shifting idea of the pixelCNN++

#@add_arg_scope
#def down_right_shifted_conv3d(x, num_filters, filter_size=[2,2,2], stride=[1,1,1], **kwargs):
#    x = tf.pad(x, [[0,0], [filter_size[0]-1, 0], [filter_size[1]-1, 0], [filter_size[2]-1, 0], [0,0]])
#    return slim.conv3d(x, num_filters, filter_size, stride=stride, padding='valid', **kwargs)

# Apply a masked convolution at the input y to prevent self-connection
# net = tf.concat([slim.conv3d(x, 16, 3, activation_fn=tf.nn.leaky_relu),
#                  down_right_shifted_conv3d(y, 16, activation_fn=tf.nn.leaky_relu)],axis=-1)


# Input features, density field
x = tf.placeholder(tf.float32, shape=[None, cube_sizep, cube_sizep, cube_sizep, nchannels], name='input')
y = tf.placeholder(tf.float32, shape=[None, cube_size, cube_size, cube_size, 1], name='labels')
lr = tf.placeholder(tf.float32, name='learningrate')

net = slim.conv3d(x, 16, 3, activation_fn=tf.nn.leaky_relu, padding='valid')
net = slim.conv3d(net, 64, 3, activation_fn=tf.nn.leaky_relu, padding='valid') 
net = slim.conv3d(net, 16, 3, activation_fn=tf.nn.leaky_relu)
net = slim.conv3d(net, 1, 3, activation_fn=None)
pred = tf.nn.sigmoid(net, name='prediction')

loss = tf.losses.sigmoid_cross_entropy(y, net)
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='optimizer')

opt_op = optimizer.minimize(loss, name='minimize')


#############################
###Train

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses = []

niter= 5000
nprint = 100
batch_size=32
#
start = time()
curr = time()
for it in range(niter):
    inds = np.random.choice(int(num_cubes), batch_size, replace=False)
    _, l = sess.run([opt_op, loss], feed_dict={lr:0.0001, x:cube_features[inds], y:cube_target[inds]})
    #print(it, l)
    if it % nprint == 0:
        print('Iteration %d of %d'%(it, niter), '\nLoss = ', l)
        end = time()
        print('Time taken for last batch = %0.3f, Total time elapsed = %0.3f'%(end-curr, end - start))
        curr = end
    losses.append(l)

print('Time taken for training = ', end - start)

#Save loss
plt.figure()
plt.plot(losses)
plt.loglog()
plt.savefig('./figs/n%02d/loss%s.png'%(numd*1e4, suff))


#Save
saver = tf.train.Saver()
saver.save(sess, './models/n%02d/test%s'%(numd*1e4, suff))


ind = 0
input = cube_features[ind]
recp = sess.run(pred, feed_dict={x:input.reshape(1, *input.shape)})

fig, ax = plt.subplots(1, 3, figsize=(15,7))
ax[0].imshow(recp[0,0,:,:,0]);
ax[0].set_title('predict')
ax[1].imshow(cube_target[ind,0,:,:,0],vmax=1);
ax[1].set_title('target')
ax[2].imshow(cube_features[ind,0,:,:,0]);
ax[2].set_title('features')
plt.savefig('./figs/n%02d/pred%s.png'%(numd*1e4, suff))

