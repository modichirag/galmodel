import os, sys
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import tensorflow_probability
tfd = tensorflow_probability.distributions
#tfd = tfp.distributions
import tensorflow_hub as hub

sys.path.append('../flowpm/')
import tfpm 
import tfpmfuncs as tfpf



def graphdm(config, data, sigma=0.01**0.5, maxiter=100, anneal=False, dataovd=False, gtol=1e-5):

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
        if dataovd:
            print('\Converting final density to overdensity because data is that\n')
            fmean = tf.reduce_mean(final)
            final = tf.multiply(final, 1/ fmean)
            final = final - 1
        #
        #Prior
        lineark = tfpf.r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')

        likelihood = tf.subtract(final, data)
        likelihood = tf.multiply(likelihood, 1/sigma)
        ##Anneal
        Rsm = tf.placeholder(tf.float32, name='smoothing')
        if anneal :
            print('\nAdding annealing part to graph\n')
            Rsm = tf.multiply(Rsm, bs/nc)
            Rsmsq = tf.multiply(Rsm, Rsm)
            smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
            likelihood = tf.squeeze(likelihood)
            likelihoodk = tfpf.r2c3d(likelihood, norm=nc**3)
            likelihoodk = tf.multiply(likelihoodk, tf.cast(smwts, tf.complex64))
            residual = tfpf.c2r3d(likelihoodk, norm=nc**3)
        else:
            residual = tf.identity(likelihood)
            
        chisq = tf.multiply(residual, residual)
        chisq = tf.reduce_sum(chisq)
        chisq = tf.multiply(chisq, 1/nc**3, name='chisq')

        loss = tf.add(chisq, prior, name='loss')
        
        optimizer = ScipyOptimizerInterface(loss, var_list=[linear], method='L-BFGS-B', 
                                            options={'maxiter': maxiter, 'gtol':gtol})
        
        tf.add_to_collection('inits', [initlin_op, initlin])
        tf.add_to_collection('opt', optimizer)
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate])
        tf.add_to_collection('data', data)
    return g



def graphinit(config, modpath, data, pad, maxiter=100, gtol=1e-5, anneal=True, resnorm=3, loss='loglikelihood', sample='sample', log=False):

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
        #
        if pad:
            xx = tf.concat((linear[-pad:, :, :], linear, linear[:pad, :, :]), axis=0)
            xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
            xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
            xx = tf.expand_dims(tf.expand_dims(xx, 0), -1)
        else:
            xx = tf.expand_dims(tf.expand_dims(linear, 0), -1)
        #Halos
        #if inference : xx = tf.log1p(xx)
        if log : xx = tf.log(tf.add(1.0, xx))
        yy = tf.expand_dims(tf.expand_dims(data, 0), -1)
        print('xx, yy shape :', xx.shape, yy.shape)
        likelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[loss]
        samples = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[sample]
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

        residual = tf.reduce_sum(likelihood)
        if loss == 'loglikelihood':
            residual = tf.multiply(residual, -1.)
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
        tf.add_to_collection('reconpm', [linear,  samples])
        tf.add_to_collection('data', data)
    return g




def graphhposft1(config, modpath, data, pad, maxiter=100, gtol=1e-5, anneal=True, resnorm=3,  loss='loglikelihood', sample='sample', log=False, inverse=False):

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
        else:
            xx = tf.expand_dims(tf.expand_dims(final, 0), -1)
        #Halos
        if log : xx = tf.log(tf.add(1.0, xx))
        yy = tf.expand_dims(tf.expand_dims(data, 0), -1)
        print('xx, yy shape :', xx.shape, yy.shape)
        if inverse:
            likelihood = module(dict(features=tf.cast(yy, tf.float32), labels=tf.cast(xx, tf.float32)), as_dict=True)[loss]
            samples = module(dict(features=tf.cast(yy, tf.float32), labels=tf.cast(xx, tf.float32)), as_dict=True)[sample] # 
        else:
            likelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[loss]
            samples = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[sample]
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

        residual = tf.reduce_sum(likelihood)
        if loss == 'loglikelihood':
            residual = tf.multiply(residual, -1.)
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
    






from tensorflow.python.ops import gen_nn_ops
def graphhposft1pool(config, modpath, data, pad, maxiter=100, gtol=1e-5, pool=1, resnorm=3, inference=False, loss='loglikelihood', sample='sample', log=False):

    bs, nc = config['boxsize'], config['nc']
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    priorwt = config['ipklin'](kmesh) * bs ** -3 
    dx = pool
    g = tf.Graph()
    
    with g.as_default():
        
        #dx = tf.placeholder(tf.int16, name='smoothing')
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
        else:
            xx = tf.expand_dims(tf.expand_dims(final, 0), -1)
        #Halos
        #if inference : xx = tf.log1p(xx)
        if log : xx = tf.log(tf.add(1.0, xx))
        yy = tf.expand_dims(tf.expand_dims(data, 0), -1)

        loc = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)['lambda']
        likelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[loss]
        samples = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[sample]
        samples = tf.identity(samples, name='samples')
        
        print(likelihood)
        ##Anneal
##        pooltensor = tf.placeholder(shape=(None, None, None), dtype=tf.float32, name='pool')
##        poolshape = tf.shape(pooltensor)
##        print(poolshape, type(poolshape))
##        pshape = poolshape[0]
##        conv_pooled = gen_nn_ops.avg_pool3d(tf.cast(xx, tf.float32), ksize=[1,1, pshape, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool")
##
        finalpool = tf.nn.avg_pool3d(tf.cast(xx, tf.float32), ksize=[1, dx, dx, dx, 1], strides=[1, dx, dx, dx, 1], padding='VALID')*tf.cast(dx**3, tf.float32)
        yypool = tf.nn.avg_pool3d(tf.cast(yy, tf.float32), ksize=[1, dx, dx, dx, 1], strides=[1, dx, dx,dx, 1], padding='VALID')*tf.cast(dx**3, tf.float32)
        locpool = tf.nn.avg_pool3d(loc, ksize=[1, dx, dx, dx, 1], strides=[1, dx, dx, dx, 1],
                                       padding='VALID')*tf.cast(dx**3, tf.float32)

        print(locpool)
        dist = tfd.Poisson(rate=locpool)
        print(dist)
        samplepool = dist.sample()
        loglikepool = dist.log_prob(yypool)
        print(loglikepool)
        
        residual = tf.reduce_sum(loglikepool)

        #residual = tf.reduce_sum(likelihood)

        #residual = tf.reduce_sum(likelihood)
        if loss == 'loglikelihood':
            residual = tf.multiply(residual, -1.)
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






def graphhposft1poolmulti(config, modpath, data, pad, maxiter=100, gtol=1e-5, pool=1, resnorm=3, inference=False, loss='loglikelihood', sample='sample', log=False):

    bs, nc = config['boxsize'], config['nc']
    nc1, nc2, nc4 = nc, nc/2, nc/4
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    priorwt = config['ipklin'](kmesh) * bs ** -3 
    dx = pool
    g = tf.Graph()
    
    with g.as_default():
        
        #dx = tf.placeholder(tf.int16, name='smoothing')
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
        else:
            xx = tf.expand_dims(tf.expand_dims(final, 0), -1)
        #Halos
        #if inference : xx = tf.log1p(xx)
        if log : xx = tf.log(tf.add(1.0, xx))
        yy = tf.expand_dims(tf.expand_dims(data, 0), -1)

        loc = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)['lambda']
        likelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[loss]
        samples = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[sample]
        samples = tf.identity(samples, name='samples')
        
        ##Anneal

        dx = 2
        yypool2 = tf.nn.avg_pool3d(tf.cast(yy, tf.float32), ksize=[1, dx, dx, dx, 1], strides=[1, dx, dx,dx, 1], padding='VALID')*tf.cast(dx**3, tf.float32)
        locpool2 = tf.nn.avg_pool3d(loc, ksize=[1, dx, dx, dx, 1], strides=[1, dx, dx, dx, 1], padding='VALID')*tf.cast(dx**3, tf.float32)

        dx = 4
        yypool4 = tf.nn.avg_pool3d(tf.cast(yy, tf.float32), ksize=[1, dx, dx, dx, 1], strides=[1, dx, dx,dx, 1], padding='VALID')*tf.cast(dx**3, tf.float32)
        locpool4 = tf.nn.avg_pool3d(loc, ksize=[1, dx, dx, dx, 1], strides=[1, dx, dx, dx, 1], padding='VALID')*tf.cast(dx**3, tf.float32)

        
        dist1 = tfd.Poisson(rate=loc)
        samplepool1 = dist1.sample()
        loglikepool1 = dist1.log_prob(tf.cast(yy, tf.float32))
        residual1 = tf.reduce_sum(loglikepool1)

        dist2 = tfd.Poisson(rate=locpool2)
        samplepool2 = dist2.sample()
        loglikepool2 = dist2.log_prob(yypool2)
        residual2 = tf.reduce_sum(loglikepool2)

        dist4 = tfd.Poisson(rate=locpool4)
        samplepool4 = dist4.sample()
        loglikepool4 = dist4.log_prob(yypool4)
        residual4 = tf.reduce_sum(loglikepool4)

        #residual = tf.reduce_sum(likelihood)

        if loss == 'loglikelihood':
            residual1 = tf.multiply(residual1, -1.)
            residual2 = tf.multiply(residual2, -1.)
            residual4 = tf.multiply(residual4, -1.)
        chisq1 = tf.multiply(residual1, 1/nc1**resnorm, name='chisq1')
        chisq2 = tf.multiply(residual2, 1/nc2**resnorm, name='chisq2')
        chisq4 = tf.multiply(residual4, 1/nc4**resnorm, name='chisq4')
        chisq = tf.identity(chisq1, name='chisq')

        #Prior
        lineark = tfpf.r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')
        #

        loss1 = tf.add(chisq1, prior, name='loss1')
        loss2 = tf.add(chisq2, prior, name='loss2')
        loss4 = tf.add(chisq4, prior, name='loss4')
        loss = tf.identity(loss1, name='loss')
        
        lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        grads_and_vars11 = optimizer.compute_gradients(chisq1, [linear])
        grads_and_vars21 = optimizer.compute_gradients(prior, [linear])
        grads_and_vars12 = optimizer.compute_gradients(chisq2, [linear])
        grads_and_vars22 = optimizer.compute_gradients(prior, [linear])
        grads_and_vars14 = optimizer.compute_gradients(chisq4, [linear])
        grads_and_vars24 = optimizer.compute_gradients(prior, [linear])
        #opt_op = optimizer.apply_gradients(grads_and_vars1, name='apply_grad')


        
        tf.add_to_collection('inits', [initlin_op, initlin])
        tf.add_to_collection('opt', optimizer)
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate, samples])
        tf.add_to_collection('data', data)
        tf.add_to_collection('grads', [grads_and_vars14, grads_and_vars24,
                                       grads_and_vars12, grads_and_vars22,
                                       grads_and_vars11, grads_and_vars21])
    return g





def graphhposft1smgrads(config, modpath, data, pad, maxiter=100, gtol=1e-5, anneal=True, resnorm=3, inference=False, loss='loglikelihood', sample='sample', log=False, inverse=False):

    bs, nc = config['boxsize'], config['nc']
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    priorwt = config['ipklin'](kmesh) * bs ** -3 
    
    g = tf.Graph()

    with g.as_default():
        
        module = hub.Module(modpath)
        initlin = tf.placeholder(tf.float32, (nc, nc, nc), name='initlin')
        linear = tf.get_variable('linmesh', shape=(nc, nc, nc), 
                             initializer=tf.random_normal_initializer(mean=1.0, stddev=0.5), trainable=True)
        #smooth out the small scale modes in init
        initlink = tfpf.r2c3d(initlin, norm=nc**3)
        smwts = tf.exp(tf.multiply(-0.5*kmesh**2, 10))       
        initlinsm = tfpf.c2r3d(tf.multiply(initlink, tf.cast(smwts, tf.complex64)), norm=nc**3)
        #initlin_op = linear.assign(initlinsm, name='initlin_op')
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
        else:
            xx = tf.expand_dims(tf.expand_dims(final, 0), -1)
        #Halos
        #if inference : xx = tf.log1p(xx)
        if log : xx = tf.log(tf.add(log, xx))
        if len(data.shape) == 3: yy = tf.expand_dims(tf.expand_dims(data, 0), -1)
        elif len(data.shape) == 4: yy = tf.expand_dims(data, 0)
        print('xx, yy shape :', xx.shape, yy.shape)

        if inverse:
            likelihood = module(dict(features=tf.cast(yy, tf.float32), labels=tf.cast(xx, tf.float32)), as_dict=True)[loss]
            samples = module(dict(features=tf.cast(yy, tf.float32), labels=tf.cast(xx, tf.float32)), as_dict=True)[sample] # 
        else:
            likelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[loss]
            samples = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[sample]
        samples = tf.identity(samples, name='samples')
        print(likelihood.shape)
        
        #Prior
        lineark = tfpf.r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        if resnorm >=0: prior = tf.multiply(prior, 1/nc**3)
        else: pass
        prior = tf.identity(prior, name='prior')
        #

        residual = tf.reduce_sum(likelihood)
        if loss == 'loglikelihood':
            residual = tf.multiply(residual, -1.)

        if resnorm >=0: chisq = tf.multiply(residual, 1/nc**resnorm, name='chisq')
        else: chisq = tf.multiply(residual, 1., name='chisq')

        loss = tf.add(chisq, prior, name='loss')

        Rsm = tf.placeholder(tf.float32, name='smoothing')
        lr = tf.placeholder(tf.float32, name='learning_rate')

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        ##opt_op = optimizer.minimize(loss)
        
        #Anneal
        #Rsm = tf.multiply(Rsm, bs/nc)
        #Rsmsq = tf.multiply(Rsm, Rsm)
        #smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))

        grads_and_vars1 = optimizer.compute_gradients(chisq, [linear])
        grads_and_vars2 = optimizer.compute_gradients(prior, [linear])
        #opt_op = optimizer.apply_gradients(grads_and_vars1, name='apply_grad')

        
        tf.add_to_collection('inits', [initlin_op, initlin])
        tf.add_to_collection('opt', [optimizer])
        tf.add_to_collection('grads', [grads_and_vars1, grads_and_vars2])
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate, samples])
        tf.add_to_collection('data', data)
    return g









def graphhposftpot(config, modpath, data, norms, pad=0, R1=10, maxiter=100, gtol=1e-5, anneal=True, resnorm=3, loss='loglikelihood', sample='sample', log=False):
    '''Use a second field smoothed with R1 for reconstruction
    '''
    bs, nc = config['boxsize'], config['nc']
    kny = np.pi * nc/bs
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    kmeshfing = sum(((2*kny/np.pi)*np.sin(ki*np.pi/(2*kny)))**2  for ki in config['kvec'])**0.5
    priorwt = config['ipklin'](kmesh) * bs ** -3 
    lap = tfpm.laplace(config)
    
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


        fmean = tf.reduce_mean(final)
        ovd = (final-fmean)/fmean
        ovdc = tfpm.r2c3d(ovd, norm=nc**3)
        potc = ovdc*tf.cast(lap, tf.complex64)
        pot =  tfpm.c2r3d(potc, norm=nc**3)
        final = final / norms[0]
        pot = pot / norms[1]
        #
        if pad:
            xx = tf.concat((final[-pad:, :, :], final, final[:pad, :, :]), axis=0)
            xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
            xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
            xx2 = tf.concat((pot[-pad:, :, :], pot, pot[:pad, :, :]), axis=0)
            xx2 = tf.concat((xx2[:, -pad:, :], xx2, xx2[:, :pad, :]), axis=1)
            xx2 = tf.concat((xx2[:, :, -pad:], xx2, xx2[:, :, :pad]), axis=2)
#        else:
#            xx = tf.expand_dims(final, 0)
#            xx2 = tf.expand_dims(final2, 0)
#            
        xx =  tf.stack((xx, xx2), axis=-1)
        xx = tf.expand_dims(xx, 0)
        #Halos
        
        if log : xx = tf.log(tf.add(1.0, xx))
        yy = tf.expand_dims(tf.expand_dims(data, 0), -1)
        print('xx, yy shape :', xx.shape, yy.shape)
        likelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[loss]
        samples = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[sample]
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

        residual = tf.reduce_sum(likelihood)
        if loss == 'loglikelihood':
            residual = tf.multiply(residual, -1.)
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







def graphhposftR(config, modpath, data, pad=0, R1=10, maxiter=100, gtol=1e-5, anneal=True, resnorm=3, inference=False):
    '''Use a second field smoothed with R1 for reconstruction
    '''
    bs, nc = config['boxsize'], config['nc']
    kny = np.pi * nc/bs
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    kmeshfing = sum(((2*kny/np.pi)*np.sin(ki*np.pi/(2*kny)))**2  for ki in config['kvec'])**0.5
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
        R1wt = tf.exp(-0.5*kmeshfing**2 * R1**2)
        finalcR1 = tf.multiply(finalc, tf.cast(R1wt, tf.complex64))
        final2 = tfpm.c2r3d(finalcR1, norm=nc**3)
        
        #
        if pad:
            xx = tf.concat((final[-pad:, :, :], final, final[:pad, :, :]), axis=0)
            xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
            xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
            xx2 = tf.concat((final2[-pad:, :, :], final2, final2[:pad, :, :]), axis=0)
            xx2 = tf.concat((xx2[:, -pad:, :], xx2, xx2[:, :pad, :]), axis=1)
            xx2 = tf.concat((xx2[:, :, -pad:], xx2, xx2[:, :, :pad]), axis=2)
        else:
            xx = tf.expand_dims(final, 0)
            xx2 = tf.expand_dims(final2, 0)
            
        xx =  tf.stack((xx, xx2), axis=-1)
        #xx = tf.expand_dims(xx, 0)
        #Halos
        if inference : xx = tf.log1p(xx)
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






def graphhposftlin(config, modpath, data, pad=0, R1=10, maxiter=100, gtol=1e-5, anneal=True, resnorm=3, inference=False):
    '''Use linear field for reconstruction as well
    '''
    bs, nc = config['boxsize'], config['nc']
    kny = np.pi * nc/bs
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    kmeshfing = sum(((2*kny/np.pi)*np.sin(ki*np.pi/(2*kny)))**2  for ki in config['kvec'])**0.5
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
            xx2 = tf.concat((linear[-pad:, :, :], linear, linear[:pad, :, :]), axis=0)
            xx2 = tf.concat((xx2[:, -pad:, :], xx2, xx2[:, :pad, :]), axis=1)
            xx2 = tf.concat((xx2[:, :, -pad:], xx2, xx2[:, :, :pad]), axis=2)
        else:
            xx = tf.expand_dims(final, 0)
            xx2 = tf.expand_dims(linear, 0)
            
        xx =  tf.stack((xx, xx2), axis=-1)
        xx = tf.expand_dims(xx, 0)
        #Halos
        if inference : xx = tf.log1p(xx)
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



def graphhposft1regx(config, modpath, data, pad, maxiter=100, gtol=1e-5, anneal=True, resnorm=3):
    '''Try to add the regularization likelihood to the  likelihood of the forward model
    '''
    bs, nc = config['boxsize'], config['nc']
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    priorwt = config['ipklin'](kmesh) * bs ** -3 
    
    g = tf.Graph()

    with g.as_default():
        
        module = hub.Module(modpath + '/likelihood/')
        module_inf = hub.Module(modpath+ '/inference/')
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
        else:
            xx = tf.expand_dims(tf.expand_dims(final, 0), -1)
        #Halos
        xxlog = tf.log1p(xx)
        yy = tf.expand_dims(tf.expand_dims(data, 0), -1)
        print('xx, yy shape :', xx.shape, yy.shape)
        fwdlikelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)['loglikelihood']
        samples = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)['sample']
        samples = tf.identity(samples, name='samples')
        reglikelihood = module_inf(dict(features=tf.cast(xxlog, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)['loglikelihood']
        fwdlikelihood = tf.identity(fwdlikelihood, name='loglik')
        reglikelihood = tf.identity(reglikelihood, name='regloglik')
        likelihood = tf.add(fwdlikelihood, reglikelihood)
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
    kny = np.pi * nc/bs
    kmeshfing = sum(((2*kny/np.pi)*np.sin(ki*np.pi/(2*kny)))**2  for ki in config['kvec'])**0.5
    
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
        #R1wt = tf.exp(-kmesh**2 * R1**2)
        R1wt = tf.exp(-0.5*kmeshfing**2 * R1**2)
        finalcR1 = tf.multiply(finalc, tf.cast(R1wt, tf.complex64))
        finalR1 = tfpm.c2r3d(finalcR1, norm=nc**3)
        #R2wt = tf.exp(-kmesh**2 * R2**2)
        R2wt = tf.exp(-0.5*kmeshfing**2 * R2**2)
        finalcR2 = tf.multiply(finalc, tf.cast(R2wt, tf.complex64))
        finalR2 = tfpm.c2r3d(finalcR2, norm=nc**3)
        final2 = tf.add(finalR1, -finalR2)
        
        #
        if pad:
            xx = tf.concat((final[-pad:, :, :], final, final[:pad, :, :]), axis=0)
            xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
            xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
            xx2 = tf.concat((final2[-pad:, :, :], final2, final2[:pad, :, :]), axis=0)
            xx2 = tf.concat((xx2[:, -pad:, :], xx2, xx2[:, :pad, :]), axis=1)
            xx2 = tf.concat((xx2[:, :, -pad:], xx2, xx2[:, :, :pad]), axis=2)
        xx =  tf.stack((xx, xx2), axis=-1)
        xx = tf.expand_dims(xx, 0)
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
        tf.add_to_collection('reconpm', [linear, final, fnstate])
        tf.add_to_collection('data', data)
    return g
    




def graphdm_likelihood(config, likelihood, maxiter=100, anneal=False, dataovd=False, gtol=1e-5):

    bs, nc = config['boxsize'], config['nc']
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
    priorwt = config['ipklin'](kmesh) * bs ** -3 
    mean, sigma, prob = likelihood
    #prob = tf.nn.softmax(logits)
    
    g = tf.Graph()

    with g.as_default():
        
        initlin = tf.placeholder(tf.float32, (nc, nc, nc), name='initlin')
        linear = tf.get_variable('linmesh', shape=(nc, nc, nc), 
                             initializer=tf.random_normal_initializer(), trainable=True)
        initlin_op = linear.assign(initlin, name='initlin_op')
        #PM
        icstate = tfpm.lptinit(linear, config, name='icstate')
        fnstate = tfpm.nbody(icstate, config, verbose=False, name='fnstate')
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, fnstate[0], boxsize=bs, name='final')
        if dataovd:
            print('\Converting final density to overdensity because data is that\n')
            fmean = tf.reduce_mean(final)
            final = tf.multiply(final, 1/ fmean)
            final = final - 1
        #
        final = tf.log(tf.add(1.0, final))
        final = tf.expand_dims(final, axis=-1)
        #Prior
        lineark = tfpf.r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')
        
        loglikelihood = tf.subtract(final, mean)
        loglikelihood = tf.multiply(loglikelihood, 1/sigma)
        
        ##Anneal
        Rsm = tf.placeholder(tf.float32, name='smoothing')
        if anneal :
            print('\nAdding annealing part to graph\n')
            Rsm = tf.multiply(Rsm, bs/nc)
            Rsmsq = tf.multiply(Rsm, Rsm)
            smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
            loglikelihood = tf.squeeze(loglikelihood)
            residual = []
            for i in range(sigma.shape[-1]):
                loglikelihoodk = tfpf.r2c3d(loglikelihood[..., i], norm=nc**3)
                loglikelihoodk = tf.multiply(loglikelihoodk, tf.cast(smwts, tf.complex64))
                residual.append(tfpf.c2r3d(loglikelihoodk, norm=nc**3))
            residual = tf.stack(residual, axis=-1)
        else:
            residual = tf.identity(loglikelihood)

        probability = tf.multiply(tf.multiply(residual, residual), -0.5)
        probability = tf.multiply(tf.exp(probability), 1/(2*np.pi*sigma**2)**0.5)
        probability = tf.reduce_sum(tf.multiply(probability, prob), axis=-1)
        logprob = -tf.log(probability)
        chisq = tf.reduce_sum(logprob)
        chisq = tf.multiply(chisq, 1/nc**3, name='chisq')

        loss = tf.add(chisq, prior, name='loss')
        
        optimizer = ScipyOptimizerInterface(loss, var_list=[linear], method='L-BFGS-B', 
                                            options={'maxiter': maxiter, 'gtol':gtol})
        
        tf.add_to_collection('inits', [initlin_op, initlin])
        tf.add_to_collection('opt', optimizer)
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate])
        #tf.add_to_collection('data', data)
    return g







def graphhposftfourier(config, modpath, data, pad, maxiter=100, gtol=1e-5, anneal=True, resnorm=3, inference=False, loss='loglikelihood', sample='sample', log=False):

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
        else:
            xx = tf.expand_dims(tf.expand_dims(final, 0), -1)
        #Halos
        #if inference : xx = tf.log1p(xx)
        if log : xx = tf.log(tf.add(1.0, xx))
        yy = tf.expand_dims(tf.expand_dims(data, 0), -1)
        print('xx, yy shape :', xx.shape, yy.shape)
        likelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[loss]
        samples = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[sample]
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

        residual = tf.reduce_sum(likelihood)
        if loss == 'loglikelihood':
            residual = tf.multiply(residual, -1.)
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



##def graphhposft1revnet(config, modpath, data, pad, loss, maxiter=100, gtol=1e-5, anneal=True, resnorm=3):
##
##    bs, nc = config['boxsize'], config['nc']
##    kmesh = sum(kk**2 for kk in config['kvec'])**0.5
##    priorwt = config['ipklin'](kmesh) * bs ** -3 
##    
##    g = tf.Graph()
##
##    with g.as_default():
##        
##        module = hub.Module(modpath)
##        initlin = tf.placeholder(tf.float32, (nc, nc, nc), name='initlin')
##        linear = tf.get_variable('linmesh', shape=(nc, nc, nc), 
##                             initializer=tf.random_normal_initializer(mean=1.0, stddev=0.5), trainable=True)
##        initlin_op = linear.assign(initlin, name='initlin_op')
##        #PM
##        icstate = tfpm.lptinit(linear, config, name='icstate')
##        fnstate = tfpm.nbody(icstate, config, verbose=False, name='fnstate')
##        final = tf.zeros_like(linear)
##        final = tfpf.cic_paint(final, fnstate[0], boxsize=bs, name='final')
##        #
##        if pad:
##            xx = tf.concat((final[-pad:, :, :], final, final[:pad, :, :]), axis=0)
##            xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
##            xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
##            xx = tf.expand_dims(tf.expand_dims(xx, 0), -1)
##        else:
##            xx = tf.expand_dims(tf.expand_dims(final, 0), -1)
##        #Halos
##        if inference : xx = tf.log1p(xx)
##        yy = tf.expand_dims(tf.expand_dims(data, 0), -1)
##        print('xx, yy shape :', xx.shape, yy.shape)
##        likelihood = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)[loss]
##        samples = module(dict(features=tf.cast(xx, tf.float32), labels=tf.cast(yy, tf.float32)), as_dict=True)['sample']
##        samples = tf.identity(samples, name='samples')
##        print(likelihood.shape)
##        
##        ##Anneal
##        Rsm = tf.placeholder(tf.float32, name='smoothing')
##        if anneal :
##            Rsm = tf.multiply(Rsm, bs/nc)
##            Rsmsq = tf.multiply(Rsm, Rsm)
##            smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
##            likelihood = tf.squeeze(likelihood)
##            print(likelihood.shape)
##            likelihoodk = tfpf.r2c3d(likelihood, norm=nc**3)
##            likelihoodk = tf.multiply(likelihoodk, tf.cast(smwts, tf.complex64))
##            likelihood = tfpf.c2r3d(likelihoodk, norm=nc**3)
##
##        residual = - tf.reduce_sum(likelihood)
##        
##        #Prior
##        lineark = tfpf.r2c3d(linear, norm=nc**3)
##        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
##        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
##        prior = tf.multiply(prior, 1/nc**3, name='prior')
##        #
##        chisq = tf.multiply(residual, 1/nc**resnorm, name='chisq')
##
##        loss = tf.add(chisq, prior, name='loss')
##       
##        optimizer = ScipyOptimizerInterface(loss, var_list=[linear], method='L-BFGS-B', 
##                                            options={'maxiter': maxiter, 'gtol':gtol})
##        
##        tf.add_to_collection('inits', [initlin_op, initlin])
##        tf.add_to_collection('opt', optimizer)
##        tf.add_to_collection('diagnostics', [prior, chisq, loss])
##        tf.add_to_collection('reconpm', [linear, final, fnstate, samples])
##        tf.add_to_collection('data', data)
##    return g
##
