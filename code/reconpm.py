import numpy as np
import numpy
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from time import time
import matplotlib.pyplot as plt

import tensorflow as tf

sys.path.append('./flowpm/')
from background import *
import tfpm 
import tfpmfuncs as tfpf
from tfpmconfig import Config


#Generate DATA

bs, nc = 100, 32
seed = 100
ofolder = './recon/L%04d_N%04d_S%04d/'%(bs, nc, seed)
try: os.makedirs(ofolder)
except: pass

pkfile = '../code/flowpm/Planck15_a1p00.txt'
config = Config(bs=bs, nc=nc, seed=seed, pkfile=pkfile)
#bs, nc = config['boxsize'], config['nc']
grid = bs/nc*np.indices((nc, nc, nc)).reshape(3, -1).T.astype(np.float32)

#Generate Data
tf.reset_default_graph()

linear = tfpm.linfield(config)
lineark = tfpf.r2c3d(linear, config)
icstate = tfpm.lptinit(lineark, grid, config)
fnstate = tfpm.nbody(icstate, config, verbose=True)
final = tf.zeros_like(linear)
final = tfpf.cic_paint(final, fnstate[0], boxsize=bs)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    truth, data = sess.run([linear, final])

np.save(ofolder+'final.f4', data)
np.save(ofolder+'linear.f4', truth)
###


tf.reset_default_graph()

kmesh = sum(kk**2 for kk in config['kvec'])**0.5
priorwt = config['ipklin'](kmesh)
#
linear = tf.get_variable('linmesh', shape=(nc, nc, nc),
                         initializer=tf.random_normal_initializer(), trainable=True)
lineark = tfpf.r2c3d(linear, config)
icstate = tfpm.lptinit(lineark, grid, config)
fnstate = tfpm.nbody(icstate, config, verbose=False)
final = tf.zeros_like(linear)
final = tfpf.cic_paint(final, fnstate[0], boxsize=bs)
#
priormesh = tf.multiply(lineark, priorwt)
prior = tf.norm(priormesh)
prior = tf.cast(prior, tf.float32)
prior = tf.multiply(prior, 1/nc**3)

chisq = tf.losses.mean_squared_error(final, data, weights=1/0.1)
loss = tf.add(chisq, prior)

#Optimize
lr = tf.placeholder(tf.float32, name='learningrate')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
opt_op = optimizer.minimize(loss, var_list=[linear])
# opt_op = optimizer.minimize(loss)

niter = 10000
lr0 = 50
nlr, lrfac = 2000, 2
nprint = 1000


with tf.Session() as sess:
    losses = []
    sess.run(tf.global_variables_initializer())
    l, init = sess.run([loss, linear])
    losses.append(l)
    #
    start, curr = time(), time()
    
    for it in range(niter+1):
#         print(it)
        _, l = sess.run([opt_op, loss], feed_dict={lr:lr0})
        if it % nlr == 0:
            lr0 /= lrfac
            print('reduce learning rate by factor of %0.2f. New learning rate = %0.2e'%(lrfac, lr0))
        if it % nprint == 0:
            end = time()
            print('Iter %d of %d : Loss= %0.4f\nTime taken for last %d iterations = %0.3f, \
            \nTotal time elapsed = %0.3f'%(it, niter, l, nprint, end-curr, end - start))
            curr = end
            recon = sess.run(linear)
            np.save(ofolder + 'recon_%04d.f4'%it, recon)

        losses.append(l)
        plt.figure()
        plt.plot(losses)
        plt.savefig('./figs/recon/losses.png')
        plt.close()
        
    recon = sess.run(linear)
    np.save(ofolder + 'recon_%04d.f4'%it, recon)
    
