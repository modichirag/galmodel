import numpy as np
import matplotlib.pyplot as plt
#
import sys, os
sys.path.append('./utils/')
import tools
import datalib as dlib
import datatools as dtools
from time import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 #
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import add_arg_scope
from layers import wide_resnet
import tensorflow_probability
tfd = tensorflow_probability.distributions
 
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
numd = 5e-4
num = int(numd*bs**3)
R1 = 3
R2 = 3*1.2
kny = np.pi*ncp/bs
kk = tools.fftk((ncp, ncp, ncp), bs)
seeds = [100, 200, 300, 400, 800, 900]
rprob = 0.6

#############################

suff = 'pad2d9wt0v2'
if not os.path.exists('models/gal%02d/%s'%(numd*1e4, suff)):
    os.makedirs('models/gal%02d/%s'%(numd*1e4, suff))
fname = open('models/gal%02d/%s/log'%(numd*1e4, suff), 'w+', 1)
#fname  = None
num_cubes= 500
cube_size = 32
pad = 2
cube_sizeft = cube_size + 2*pad
max_offset = ncp - cube_size
ftname = ['cic']
nchannels = len(ftname)
print('Features are ', ftname, file=fname)
print('Pad with ', pad, file=fname)

#############################
##Read data and generate meshes
#mesh = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'mesh/d/')
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

    meshes[seed] = [mesh, hmesh]

    print('All the mesh have been generated for seed = %d'%seed)

    #Create training voxels
    ftlist = [mesh[i].copy() for i in ftname]
    ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
    targetmesh = [hmesh['pnncen'], hmesh['pnnsat']]
    ntarget = len(targetmesh)
    
    features, target = dtools.randomvoxels(ftlistpad, targetmesh, num_cubes, max_offset, cube_size, cube_sizeft,
                                           seed=seed, rprob=rprob)
    cube_features = cube_features + features
    cube_target = cube_target + target

#
cube_target = np.stack(cube_target,axis=0).reshape((-1,cube_size,cube_size,cube_size, ntarget))
cube_features = np.stack(cube_features,axis=0).reshape((-1,cube_sizeft,cube_sizeft,cube_sizeft,nchannels))
print('Centrals = ', np.unique(cube_target[:, :, :, :, 0], return_counts=True))
print('Satellites = ', np.unique(cube_target[:, :, :, :, 1], return_counts=True))
print(cube_target[0].shape)
print(cube_features[0].shape)
trainingsize = cube_features.shape[0]
print('Training size is = ', trainingsize)

#
#Save a snapshot of features
fig, ax = plt.subplots(1, nchannels+1, figsize = (nchannels*4+4, 5))
n = 10
for i in range(nchannels):
    ax[i].imshow(cube_features[n][:,:,:,i].sum(axis=0))
    ax[i].set_title(ftname[i])
ax[-1].imshow(cube_target[n][:,:,:,0].sum(axis=0))
ax[-1].set_title('Target')
plt.savefig('./figs/gal%02d/features%s.png'%(numd*1e4, suff))

#############################
### Model

# Define the network
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, cube_sizeft, cube_sizeft, cube_sizeft, nchannels], name='input')
ycen = tf.placeholder(tf.float32, shape=[None, cube_size, cube_size, cube_size, 1], name='centrals')
ysat = tf.placeholder(tf.float32, shape=[None, cube_size, cube_size, cube_size, 1], name='satellites')
ywt = tf.placeholder(tf.float32, shape=[None, cube_size, cube_size, cube_size], name='weights')
##x = tf.placeholder(tf.float32, shape=[None, cube_sizeft, cube_sizeft, cube_sizeft, nchannels], name='input')
##y = tf.placeholder(tf.float32, shape=[None, cube_size, cube_size, cube_size, 1], name='labels')
##m = tf.placeholder(tf.float32, shape=[None, cube_size, cube_size, cube_size], name='mask')

keepprob = tf.placeholder(tf.float32, name='keepprob')
lr = tf.placeholder(tf.float32, name='learningrate')

net = slim.conv3d(x, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
# net = slim.conv3d(net, 32, 5, activation_fn=None, padding='valid')
net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=keepprob)
net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=keepprob)
net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=keepprob)
net = slim.conv3d(net, 64, 1, activation_fn=tf.nn.relu6)
net = tf.nn.dropout(net, keep_prob=keepprob)

# Create mixture components from network output
#out_rate = slim.conv3d(net, 1, 3, activation_fn=tf.nn.relu)
out_rate = slim.conv3d(net, 1, 1, activation_fn=tf.nn.relu, 
                       #weights_initializer=tf.initializers.random_normal(mean=1, stddev=0.25))
                       weights_initializer=tf.initializers.random_uniform(minval=0.01, maxval=1))
out_rate = tf.math.add(out_rate, 1e-8, name='rate')

# Predicted mask
out_mask = slim.conv3d(net, 1, 1, activation_fn=None)
pred_mask = tf.nn.sigmoid(out_mask, name='prediction')

pdf = tfd.Poisson(rate=out_rate)

losssat = - tf.reduce_mean(pdf.log_prob(ysat) * ywt, name='losssat') 
losscen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,labels=ycen), name='losscen') 
loss = tf.math.add(losssat, losscen)
#loss = - tf.reduce_mean(pdf.log_prob(y) * m) + \
#    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,labels=tf.expand_dims(m,-1)))
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,labels=tf.expand_dims(m,-1)))

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='optimizer')

opt_op = optimizer.minimize(loss, name='minimize')


#############################
##Setup

cube_centrals = cube_target[:, :, :, :, 0:1].copy()
cube_centrals[cube_centrals > 1] = 1

cube_satellites = cube_target[:, :, :, :, 1:].copy()

mmask = cube_target[:, :, :, :, 0].astype(bool)
nsat = np.unique(cube_target[mmask][:, 1], return_counts=True)

wts = cube_target[:, :, :, :, 1].copy()
for i, ns in enumerate(nsat[0]):
    #wts[wts == ns] = 1/nsat[1][i]**0.5
    wts[wts == ns] = 1

wts *= mmask
print('Weights = ', np.unique(wts, return_counts=True))
print('Weights = ', np.unique(wts, return_counts=True), file=fname)

#############################
###Train

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses = []

niter = 25000
nprint = 100
batch_size = 32
#
start = time()
curr = time()
lr0, lrfac, nlr = 0.001, 10, int(4000)
lr0 *= lrfac
kprob = 0.9
lacc = 0 

print('Batch size, dropout, niter : ', batch_size, kprob, niter, file=fname)
print('lr0, lrfac, nlr : ', lr0, lrfac, nlr, file=fname)
#Save
saver = tf.train.Saver()


for it in range(niter):
    inds = np.random.choice(int(trainingsize), batch_size, replace=False)
    #_, l = sess.run([opt_op, loss], feed_dict={lr:lr0, keepprob:kprob,
    #                                           x:cube_features[inds], y:cube_satellites[inds],
    #                                           m:1.*(np.squeeze(cube_centrals[inds])>0) })
    _, l = sess.run([opt_op, loss], feed_dict={lr:lr0, keepprob:kprob,
                                               x:cube_features[inds], ysat:cube_satellites[inds],
                                               ywt:wts[inds], ycen:cube_centrals[inds] })
    lacc += l
    if it % nlr == 0:
        lr0 /= 10
        print('reduce learning rate by factor of %0.2f. New learning rate = %0.2e'%(lrfac, lr0), file=fname)
        #Save network
        saver.save(sess, './models/gal%02d/%s/%s_it%d'%(numd*1e4, suff, suff, it))
        print('reduce learning rate by half. New learning rate = %0.2e'%lr0)        

    if it % nprint == 0:
        end = time()
        print('Iter %d of %d : Loss= %0.4f\nTime taken for last batch = %0.3f, Total time elapsed = %0.3f'%(it, niter, l, end-curr, end - start), file=fname)
        print('Iter %d of %d : Loss= %0.4f\nTime taken for last batch = %0.3f, Total time elapsed = %0.3f'%(it, niter, l, end-curr, end - start))
        curr = end
        #loss
        plt.figure()
        plt.semilogy(losses)
        #plt.loglog()
        plt.savefig('./figs/gal%02d/loss%s.png'%(numd*1e4, suff))
        plt.close()
        lacc=0

        #Diagnose
        input = cube_features[inds]
        recp, recm, rates = sess.run([pdf.sample(), pred_mask, out_rate], feed_dict={keepprob:1, x:input})

        ind = 0
        fig, axar = plt.subplots(2, 2, figsize=(10,10))
        ax = axar[0]
        ax[0].imshow(recm[ind, 0,:,:,0] );
        ax[0].set_title('predict mask')
        im = ax[1].imshow(recp[ind,0,:,:,0] );
        plt.colorbar(im, ax=ax[1])
        ax[1].set_title('predict sats')
        ax = axar[1]
        ax[0].imshow(cube_target[inds][ind,0,:,:,0]);
        ax[0].set_title('target cen')
        im = ax[1].imshow(cube_target[inds][ind,0,:,:,1]);
        plt.colorbar(im, ax=ax[1])
        ax[1].set_title('target sat')
        plt.savefig('./figs/gal%02d/pred%s.png'%(numd*1e4, suff))
        plt.close()

        plt.figure()
        plt.hist(rates.flatten(), log=True, histtype='step')
        plt.hist(cube_satellites[ind].flatten(), log=True)
        plt.savefig('./figs/gal%02d/rates%s.png'%(numd*1e4, suff))
        plt.close()
        
    losses.append(l)
    

#Save
print('Time taken for training = ', end - start, file=fname)
saver.save(sess, './models/gal%02d/%s/%s'%(numd*1e4, suff, suff))

#Save loss
plt.figure()
plt.semilogy(losses)
#plt.loglog()
plt.savefig('./figs/gal%02d/loss%s.png'%(numd*1e4, suff))
plt.close()


input = cube_features[inds]
#recp, recm, rates = sess.run([pdf.sample(), pred_mask, out_rate], feed_dict={keepprob:1, x:input.reshape(1, *input.shape)})
recp, recm, rates = sess.run([pdf.sample(), pred_mask, out_rate], feed_dict={keepprob:1, x:input})

ind = 0
fig, axar = plt.subplots(2, 2, figsize=(10,10))
ax = axar[0]
ax[0].imshow(recm[ind, 0,:,:,0] );
ax[0].set_title('predict mask')
im = ax[1].imshow(recp[ind,0,:,:,0] );
plt.colorbar(im, ax=ax[1])
ax[1].set_title('predict sats')
ax = axar[1]
ax[0].imshow(cube_target[inds][ind,0,:,:,0]);
ax[0].set_title('target cen')
im = ax[1].imshow(cube_target[inds][ind,0,:,:,1]);
plt.colorbar(im, ax=ax[1])
ax[1].set_title('target sat')
plt.savefig('./figs/gal%02d/pred%s.png'%(numd*1e4, suff))

plt.figure()
plt.hist(rates.flatten(), log=True, histtype='step')
plt.hist(cube_satellites[ind].flatten(), log=True)
plt.savefig('./figs/gal%02d/rates%s.png'%(numd*1e4, suff))

fname.close()
