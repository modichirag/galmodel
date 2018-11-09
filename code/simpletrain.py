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
step, stepf = 5, 40
path = './../data/z00/'
ftype = 'L%04d_N%04d_S%04d_%02dstep/'
numd = 5e-4
num = int(numd*bs**3)
seed= 100
R1 = 3
R2 = 3*1.2
kny = np.pi*nc/bs
kk = tools.fftk((nc, nc, nc), bs)


#############################

mesh = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'mesh/d/')
meshdecic = tools.decic(mesh, kk, kny)
meshR1 = tools.fingauss(mesh, kk, R1, kny)
meshR2 = tools.fingauss(mesh, kk, R2, kny)
meshdg = meshR1 - meshR2
ftlist = [meshdecic.copy(), meshR1.copy(), meshdg.copy()]


hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]
hposd = hposall[:num].copy()
hposall = hposall[:2*num]

# hpmeshall = tools.paintcic(hposall, bs, nc)
# hpmeshd = tools.paintcic(hposd, bs, nc)
# clsgrid = hpmeshd.copy()
hpmeshall = tools.paintnn(hposall, bs, nc)
hpmeshd = tools.paintnn(hposd, bs, nc)
clsgrid = hpmeshd.copy()

print('All the mesh have been generated')

dind = dtools.balancepts(mesh, blim=1e-2, hfinegrid=hpmeshall, hlim=0.01)
print(dind.shape, dind.shape[0]/nc**3)

#############################

num_cubes=1000
cube_size = 32
max_offset = 128 - cube_size
cube_features = []
cube_target = []

rand = np.random.rand
for it in range(num_cubes):
    # Extract random cubes from the sim
    offset_x = round(rand()*max_offset)
    offset_y = round(rand()*max_offset)
    offset_z = round(rand()*max_offset)
    
#     cube_features.append( clip(log10(mesh[offset_x:offset_x+cube_size,
#                                           offset_y:offset_y+cube_size,
#                                           offset_z:offset_z+cube_size] + 0.1),0,None))
    
    cube_features.append( 1.0*(mesh[offset_x:offset_x+cube_size,
                                          offset_y:offset_y+cube_size,
                                          offset_z:offset_z+cube_size]))

    cube_target.append(1.0*(hpmeshd[offset_x:offset_x+cube_size,
                                          offset_y:offset_y+cube_size,
                                          offset_z:offset_z+cube_size] ))
    

cube_target = np.stack(cube_target,axis=0).reshape((-1,cube_size,cube_size,cube_size,1))
print(cube_target.sum(), cube_target.size, hpmeshd.sum())
cube_features = np.stack(cube_features,axis=0).reshape((-1,cube_size,cube_size,cube_size,1))

#############################

fig, ax = plt.subplots(1, 2, figsize = (12, 5))

i = 10
# ax[0].imshow(cube_features[i][:,:,:,0].sum(axis=0))
# ax[1].imshow(cube_target[i][:,:,:,0].sum(axis=0))
ax[0].imshow(cube_features[i][0,:,:,0])
ax[1].imshow(cube_target[i][0,:,:,0])
plt.savefig('./figs/tmp.png')




# Implement a simple masked CNN layer
# Using the shifting idea of the pixelCNN++

@add_arg_scope
def down_right_shifted_conv3d(x, num_filters, filter_size=[2,2,2], stride=[1,1,1], **kwargs):
    x = tf.pad(x, [[0,0], [filter_size[0]-1, 0], [filter_size[1]-1, 0], [filter_size[2]-1, 0], [0,0]])
    return slim.conv3d(x, num_filters, filter_size, stride=stride, padding='valid', **kwargs)

batch_size=64

# Input features, density field
x = tf.placeholder(tf.float32, shape=[batch_size, 32,32,32,1])

# Output features, halos
y = tf.placeholder(tf.float32, shape=[batch_size, 32,32,32,1])
lr = tf.placeholder(tf.float32)

# Apply a masked convolution at the input y to prevent self-connection
# net = tf.concat([slim.conv3d(x, 16, 3, activation_fn=tf.nn.leaky_relu),
#                  down_right_shifted_conv3d(y, 16, activation_fn=tf.nn.leaky_relu)],axis=-1)
net = slim.conv3d(x, 16, 3, activation_fn=tf.nn.leaky_relu)
net = slim.conv3d(net, 64, 3, activation_fn=tf.nn.leaky_relu) #down_right_shifted_conv3d(net, 32, activation_fn=tf.nn.leaky_relu)
net = slim.conv3d(net, 16, 3, activation_fn=tf.nn.leaky_relu)#down_right_shifted_conv3d(net, 1, activation_fn=None)
net = slim.conv3d(net, 1, 3, activation_fn=None)
pred = tf.nn.sigmoid(net)

loss = tf.losses.sigmoid_cross_entropy(y, net)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

opt_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses = []

niter= 100
nprint = 10
start = time()
curr = time()
for it in range(niter):
    inds = np.random.choice(int(num_cubes), batch_size, replace=False)
    _, l = sess.run([opt_op, loss], feed_dict={lr:0.0001, x:cube_features[inds], y:cube_target[inds]})
    #print(it, l)
    if it % nprint == 0:
        print(it//nprint, l)
        end = time()
        print('Time taken for %d iteration = '%it, end - start)
        print('Time taken for %d iteration = '%nprint, end - curr)
        curr = end
    losses.append(l)

print('Time taken for training = ', end - start)

plt.figure()
plt.plot(losses)
plt.savefig('./figs/tmp2.png')


rec,recp = sess.run([net, pred], feed_dict={x:cube_features[inds], y:cube_target[inds]})


fig, ax = plt.subplots(1, 3, figsize=(15,7))
ax[0].imshow(recp[3,0,:,:,0]);
ax[0].set_title('predict')
ax[1].imshow(cube_target[inds][3,0,:,:,0],vmax=1);
ax[1].set_title('target')
ax[2].imshow(cube_features[inds][3,0,:,:,0]);
ax[2].set_title('features')
plt.savefig('./figs/tmp3.png')


saver = tf.train.Saver()
saver.save(sess, './models/test0')
