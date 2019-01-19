import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import tensorflow_hub as hub

sys.path.append('../flowpm/')
import tfpm 
import tfpmfuncs as tfpf



def graphhposft1pad2(config, modpath, data, pad, maxiter=100, gtol=1e-5, anneal=True, resnorm=3):

    bs, nc = config['boxsize'], config['nc']
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    priorwt = config['ipklin'](kmesh) * bs ** -3 
    
    g = tf.Graph()

    with g.as_default():
        
        module = hub.Module(modpath)
        initlin = tf.placeholder(tf.float32, (nc, nc, nc), name='initlin')
        linear = tf.get_variable('linmesh', shape=(nc, nc, nc), 
                             initializer=tf.random_normal_initializer(mean=1.0, stddev=0.5), trainable=True)
        initlin_op = linear.assign(initlin, name='initlin_op')
        #PM
        icstate = tfpm.lptinit(linear, config, name='icstate')
        fnstate = tfpm.nbody(icstate, config, verbose=False, name='fnstate')
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, fnstate[0], boxsize=bs, name='final')
        #
        if pad:
            xx = tf.concat((final[-pad:, :, :], final, final[:pad, :, :]), axis=0)
            xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
            xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
        xx = tf.expand_dims(tf.expand_dims(xx, 0), -1)
        #Halos
        yy = tf.expand_dims(tf.expand_dims(data, 0), -1)
        print('xx, yy shape :', xx.shape, yy.shape)
        likelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)['loglikelihood']
        samples = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)['sample']
        samples = tf.identity(samples, name='samples')
        print(likelihood.shape)
        
        ##Anneal
        Rsm = tf.placeholder(tf.float32, name='smoothing')
        if anneal :
            Rsm = tf.multiply(Rsm, bs/nc)
            Rsmsq = tf.multiply(Rsm, Rsm)
            smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
            likelihood = tf.squeeze(likelihood)
            print(likelihood.shape)
            likelihoodk = tfpf.r2c3d(likelihood, norm=nc**3)
            likelihoodk = tf.multiply(likelihoodk, tf.cast(smwts, tf.complex64))
            likelihood = tfpf.c2r3d(likelihoodk, norm=nc**3)

        residual = - tf.reduce_sum(likelihood)
        
        #Prior
        lineark = tfpf.r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')
        #
        chisq = tf.multiply(residual, 1/nc**resnorm, name='chisq')

        loss = tf.add(chisq, prior, name='loss')
       
        optimizer = ScipyOptimizerInterface(loss, var_list=[linear], method='L-BFGS-B', 
                                            options={'maxiter': maxiter, 'gtol':gtol})
        
        tf.add_to_collection('inits', [initlin_op, initlin])
        tf.add_to_collection('opt', optimizer)
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate, samples])
        tf.add_to_collection('data', data)
    return g
    



def graphhposftGDpad2(config, modpath, data, pad, maxiter=100, gtol=1e-5, anneal=True, resnorm=3, R1=3, R2=3*1.2):

    bs, nc = config['boxsize'], config['nc']
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    priorwt = config['ipklin'](kmesh) * bs ** -3 
    
    g = tf.Graph()

    with g.as_default():
        
        module = hub.Module(modpath)
        initlin = tf.placeholder(tf.float32, (nc, nc, nc), name='initlin')
        linear = tf.get_variable('linmesh', shape=(nc, nc, nc), 
                             initializer=tf.random_normal_initializer(mean=1.0, stddev=0.5), trainable=True)
        initlin_op = linear.assign(initlin, name='initlin_op')
        #PM
        icstate = tfpm.lptinit(linear, config, name='icstate')
        fnstate = tfpm.nbody(icstate, config, verbose=False, name='fnstate')
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, fnstate[0], boxsize=bs, name='final')
        finalc = tfpm.r2c3d(final, norm=nc**3)
        R1wt = tf.exp(-kmesh**2 * R1**2)
        finalcR1 = tf.multiply(finalc, R1wt)
        finalR1 = tfpm.c2r3d(finalcR1, norm=nc**3)
        R2wt = tf.exp(-kmesh**2 * R2**2)
        finalcR2 = tf.multiply(finalc, R2wt)
        finalR2 = tfpm.c2r3d(finalcR2, norm=nc**3)
        finalGD = tf.add(finalR1, -finalR2)
        
        #
        if pad:
            xx = tf.concat((final[-pad:, :, :], final, final[:pad, :, :]), axis=0)
            xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
            xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
            xx2 = tf.concat((final2[-pad:, :, :], final2, final2[:pad, :, :]), axis=0)
            xx2 = tf.concat((xx2[:, -pad:, :], xx2, xx2[:, :pad, :]), axis=1)
            xx2 = tf.concat((xx2[:, :, -pad:], xx2, xx2[:, :, :pad]), axis=2)
        xx =  tf.concat((xx, xx2), axis=-1)
        xx = tf.expand_dims(xx, 0)
        #Halos
        yy = tf.expand_dims(tf.expand_dims(data, 0), -1)
        print('xx, yy shape :', xx.shape, yy.shape)
        likelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)['loglikelihood']
        print(likelihood.shape)
        
        ##Anneal
        Rsm = tf.placeholder(tf.float32, name='smoothing')
        if anneal :
            Rsm = tf.multiply(Rsm, bs/nc)
            Rsmsq = tf.multiply(Rsm, Rsm)
            smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
            likelihood = tf.squeeze(likelihood)
            print(likelihood.shape)
            likelihoodk = tfpf.r2c3d(likelihood, norm=nc**3)
            likelihoodk = tf.multiply(likelihoodk, tf.cast(smwts, tf.complex64))
            likelihood = tfpf.c2r3d(likelihoodk, norm=nc**3)

        residual = - tf.reduce_sum(likelihood)
        
        #Prior
        lineark = tfpf.r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')
        #
        chisq = tf.multiply(residual, 1/nc**resnorm, name='chisq')

        loss = tf.add(chisq, prior, name='loss')
        
        optimizer = ScipyOptimizerInterface(loss, var_list=[linear], method='L-BFGS-B', 
                                            options={'maxiter': maxiter, 'gtol':gtol})
        
        tf.add_to_collection('inits', [initlin_op, initlin])
        tf.add_to_collection('opt', optimizer)
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate])
        tf.add_to_collection('data', data)
    return g
    

