import numpy as np
import numpy
import os, sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

        residual = tf.subtract(final, data)
        residual = tf.multiply(residual, 1/sigma)

        chisq = tf.multiply(residual, residual)
        chisq = tf.reduce_sum(chisq)
        chisq = tf.multiply(chisq, 1/nc**3, name='chisq')

        loss = tf.add(chisq, prior, name='loss')
        
        optimizer = ScipyOptimizerInterface(loss, var_list=[linear], method='L-BFGS-B', 
                                            options={'maxiter': maxiter})
        
        tf.add_to_collection('inits', [initlin_op, initlin])
        tf.add_to_collection('opt', optimizer)
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate])
        tf.add_to_collection('data', data)
    return g
    


def reconmodelanneal(config, data, sigma=0.01**0.5, maxiter=100):

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

        residual = tf.subtract(final, data)
        residual = tf.multiply(residual, 1/sigma)

        chisq = tf.multiply(residual, residual)
        chisq = tf.reduce_sum(chisq)
        chisq = tf.multiply(chisq, 1/nc**3, name='chisq')

        loss = tf.add(chisq, prior, name='loss')
        
        optimizer = ScipyOptimizerInterface(loss, var_list=[linear], method='L-BFGS-B', 
                                            options={'maxiter': maxiter})
        
        tf.add_to_collection('inits', [initlin_op, initlin])
        tf.add_to_collection('opt', optimizer)
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate])
        tf.add_to_collection('data', data)
    return g
    




def loss_callback(var, literals):
    losses = literals['losses']
    loss = var[0]
    mesh = var[1]
    nit = len(losses)  
    losses.append(loss)

    if nit % 100 == 0: print(nit, loss)
    if nit % 100 == 0:
        np.save(ofolder + 'recon%d.f4'%nit, mesh)
        

#Generate DATA

if __name__=="__main__":

    bs, nc = 100, 32
    seed = 100
    ofolder = './saved/L%04d_N%04d_S%04d/'%(bs, nc, seed)
    try: os.makedirs(ofolder)
    except: pass

    pkfile = '../flowpm/Planck15_a1p00.txt'
    config = Config(bs=bs, nc=nc, seed=seed, pkfile=pkfile)


    #Generate Data
    truth, data = gendata(config, ofolder)
    sigma = 0.01**0.5
    np.random.seed(100)
    noise = np.random.normal(loc=0, scale=sigma, size=nc**3).reshape(nc, nc, nc).astype(config['dtype'])
    datan = data + noise

    ###
    #Do reconstruction here
    print('\nDo reconstruction\n')

    recong = reconmodel(config, datan, sigma=sigma, maxiter=1000)

    
    initval = None
    losses = []
    literals = {'losses':losses}
    
    with tf.Session(graph=recong) as session:
        g = session.graph
        session.run(tf.global_variables_initializer())
        linmesh = g.get_tensor_by_name("linmesh:0")
        optimizer = g.get_collection_ref('opt')[0]
        loss = g.get_tensor_by_name('loss:0')
        chisq = g.get_tensor_by_name('chisq:0')
        prior = g.get_tensor_by_name('prior:0')

        if initval is not None:
            print('Do init')
            initlinop = g.get_operation_by_name('initlin_op')
            initlin = g.get_tensor_by_name('initlin:0')
            session.run(initlinop, {initlin:initval})

        init = session.run(linmesh)
        optimizer.minimize(session, loss_callback=lambda x:loss_callback(x, literals),
                           fetches=[[[loss, chisq, prior], linmesh]])
        recon = session.run(linmesh)

        np.save(ofolder + 'init.f4', init)
        np.save(ofolder + 'recon.f4', recon)
        
