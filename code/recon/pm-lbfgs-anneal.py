import numpy as np
import numpy
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from time import time
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

sys.path.append('../flowpm/')
from background import *
import tfpm 
import tfpmfuncs as tfpf
from tfpmconfig import Config
from gendata import gendata



def reconmodel(config, data, sigma=0.01**0.5, maxiter=100):

    bs, nc = config['boxsize'], config['nc']
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    priorwt = config['ipklin'](kmesh) * bs ** -3 
    
    g = tf.Graph()

    with g.as_default():
        
        initlin = tf.placeholder(tf.float32, data.shape, name='initlin')
        linear = tf.get_variable('linmesh', shape=(nc, nc, nc), 
                             initializer=tf.random_normal_initializer(), trainable=True)
        initlin_op = linear.assign(initlin, name='initlin_op')
        #PM
        icstate = tfpm.lptinit(linear, config, name='icstate')
        fnstate = tfpm.nbody(icstate, config, verbose=False, name='fnstate')
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, fnstate[0], boxsize=bs, name='final')
        #
        #Prior
        lineark = tfpf.r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')

        data2d = data.sum(axis=0)
        final2d = tf.reduce_sum(final, axis=0)
        residual = tf.subtract(final2d, data2d)

        residual = tf.multiply(residual, 1/sigma)

        chisq = tf.multiply(residual, residual)
        chisq = tf.reduce_sum(chisq)
        chisq = tf.multiply(chisq, 1/nc**2, name='chisq')

        loss = tf.add(chisq, prior, name='loss')
        
        optimizer = ScipyOptimizerInterface(loss, var_list=[linear], method='L-BFGS-B', 
                                            options={'maxiter': maxiter})
        
        tf.add_to_collection('utils', [initlin_op, initlin])
        tf.add_to_collection('opt', optimizer)
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate, final2d])
        tf.add_to_collection('data', [data, data2d])
    return g
    




def loss_callback(var, literals, nprint=50, nsave=50, t0=time()):
    losses = literals['losses']
    loss = var[0]
    mesh = var[1]
    nit = len(losses)  
    losses.append(loss)
    
    if nit % nprint == 0:
        print('Time taken for iterations %d = '%nit, time() - t0)
        print(nit, loss)
    if nit % nsave == 0:
        np.save(ofolder + 'iter%d.f4'%nit, mesh)
        

########################

if __name__=="__main__":

    npseed = 999
    np.random.seed(npseed)
    tf.set_random_seed(npseed)
    #
    bs, nc = 100, 32
    seed = 100
    maxiter = 500
    sigma = 1**0.5
    #sigma = 0.01**0.5
    nprint, nsave = 20, 50
    anneal = False
    R0s = [4, 2, 1, 0]
    #output folder
    ofolder = './saved/L%04d_N%04d_S%04d/'%(bs, nc, seed)
    if anneal : ofolder += 'anneal%d/'%len(R0s)
    ofolder += '2d/sigma01/'
    try: os.makedirs(ofolder)
    except: pass
    
    pkfile = '../flowpm/Planck15_a1p00.txt'
    config = Config(bs=bs, nc=nc, seed=seed, pkfile=pkfile)


    #Generate Data
    truth, data = gendata(config, ofolder)
    noise = np.random.normal(loc=0, scale=sigma, size=nc**3).reshape(nc, nc, nc).astype(config['dtype'])
    datan = data + noise
    np.save(ofolder + 'datan.f4', datan)

    ###
    #Do reconstruction here
    print('\nDo reconstruction\n')

    if anneal: recong = reconmodelanneal(config, datan, sigma=sigma, maxiter=maxiter)
    else: recong = reconmodel(config, datan, sigma=sigma, maxiter=maxiter)
    
    initval = np.random.normal(1, 0.5, size=nc**3).reshape(nc, nc, nc).astype(config['dtype'])#truth
    losses = []
    literals = {'losses':losses}
    tstart = time()
    lcallback = lambda x: loss_callback(x, literals=literals, nprint=nprint, nsave=nsave, t0=tstart)
    
    with tf.Session(graph=recong) as session:
        g = session.graph
        session.run(tf.global_variables_initializer())
        linmesh = g.get_tensor_by_name("linmesh:0")
        optimizer = g.get_collection_ref('opt')[0]
        loss = g.get_tensor_by_name('loss:0')
        chisq = g.get_tensor_by_name('chisq:0')
        prior = g.get_tensor_by_name('prior:0')
        grad = tf.norm(tf.gradients(loss, linmesh))
        if anneal : Rsm = g.get_tensor_by_name('smoothing:0')
        
        if initval is not None:
            print('Do init')
            initlinop = g.get_operation_by_name('initlin_op')
            initlin = g.get_tensor_by_name('initlin:0')
            session.run(initlinop, {initlin:initval})

        init, recon = [], []
        if anneal:
            for R0 in R0s:
                init.append(session.run(linmesh))
                np.save(ofolder + 'init%d.f4'%R0, init[-1])
                optimizer.minimize(session, {Rsm:R0}, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad], linmesh]])
                recon.append(session.run(linmesh))
                np.save(ofolder + 'recon%d.f4'%R0, recon[-1])
            
        else:
            init = session.run(linmesh)
            np.save(ofolder + 'init.f4', init)            
            optimizer.minimize(session, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad], linmesh]])
            recon = session.run(linmesh)
            np.save(ofolder + 'recon.f4', recon)
        
