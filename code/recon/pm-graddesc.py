import numpy as np
import numpy
import os, sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from time import time
import matplotlib.pyplot as plt

import tensorflow as tf

sys.path.append('./flowpm/')
from background import *
import tfpm 
import tfpmfuncs as tfpf
from tfpmconfig import Config

from gendata import gendata

#Generate DATA

bs, nc = 400, 128
seed = 100
ofolder = './recon/L%04d_N%04d_S%04d/'%(bs, nc, seed)
try: os.makedirs(ofolder)
except: pass
pkfile = '../code/flowpm/Planck15_a1p00.txt'
config = Config(bs=bs, nc=nc, seed=seed, pkfile=pkfile)

truth, data = gendata(config, ofolder)


#################################################################
#Do reconstruction here
print('\nDo reconstruction\n')
tf.reset_default_graph()

kmesh = sum(kk**2 for kk in config['kvec'])**0.5
priorwt = config['ipklin'](kmesh)
# priorwt

linear = tf.get_variable('linmesh', shape=(nc, nc, nc), 
                         initializer=tf.random_normal_initializer(), trainable=True)
icstate = tfpm.lptinit(linear, grid, config)
fnstate = tfpm.nbody(icstate, config, verbose=False)
final = tf.zeros_like(linear)
final = tfpf.cic_paint(final, fnstate[0], boxsize=bs)
#

lineark = tfpf.r2c3d(linear, norm=nc**3)
prior = tf.square(tf.cast(tf.abs(lineark), tf.float32))
prior = tf.reduce_sum(tf.multiply(prior, priorwt))
prior = tf.multiply(prior, 1/nc**3)

sigma = 0.01**0.5

residual = tf.subtract(final, data)
residual = tf.multiply(residual, 1/sigma)

Rsm = tf.placeholder(tf.float32, name='smoothing')

#Rsm = tf.multiply(Rsm, bs/nc)
#Rsmsq = tf.multiply(Rsm, Rsm)
#smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
#residualk = tfpf.r2c3d(residual, norm=nc**3)
#residualk = tf.multiply(residualk, tf.cast(smwts, tf.complex64))
#residual = tfpf.c2r3d(residualk, norm=nc**3)
chisq = tf.multiply(residual, residual)
chisq = tf.reduce_sum(chisq)
chisq = tf.multiply(chisq, 1/nc**3)

# chisq = tf.losses.mean_squared_error(final, data, weights=1/sigma)

loss = tf.add(chisq, prior)


lr = tf.placeholder(tf.float32, name='learningrate')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
opt_op = optimizer.minimize(loss, var_list=[linear])
# opt_op = optimizer.minimize(loss)


niter = 10000
lr0 = 50
nlr, lrfac = 2000, 2
nprint = 200
R0 = 4.

with tf.Session() as sess:
    losses = []
    sess.run(tf.global_variables_initializer())
    l, init = sess.run([loss, linear], {Rsm:0.})
    losses.append(l)
    print('Initial loss = ', l)
    #
    start, curr = time(), time()
    
    lp = l
    Rcounter = 0 
    for it in range(niter+1):
        _, l = sess.run([opt_op, loss], feed_dict={lr:lr0, Rsm:R0})

        if it % nlr == 0:
            lr0 /= lrfac
            print('reduce learning rate by factor of %0.2f. New learning rate = %0.2e'%(lrfac, lr0))

        #Anneal
        Rcounter +=1
        if (R0 > 0) and (abs(l-lp)/lp < 0.001):
            if Rcounter  > 10:
                R0 /= 2
                print('reduced R0 to : ', R0)
                print('at iteration : ', it)
                Rcounter = 0
            else: print('Tried too soon to reduce R0')
            if R0 < 0.5: R0 = 0 
        lp = l
        #
        if it % nprint == 0:
            end = time()
            print('Iter %d of %d : Loss= %0.4f\nTime taken for last batch = %0.3f, \
            \nTotal time elapsed = %0.3f'%(it, niter, l, end-curr, end - start))
            curr = end
            recon = sess.run(linear)
            np.save(ofolder + 'recon_%04d.f4'%it, recon)
            plt.figure()
            plt.plot(losses)
            plt.savefig('./figs/recon/losses.png')
            plt.close()

        losses.append(l)

    recon = sess.run(linear)
    np.save(ofolder + 'recon_%04d.f4'%it, recon)

