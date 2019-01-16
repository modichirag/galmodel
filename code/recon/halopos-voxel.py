import numpy as np
import numpy
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from time import time
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

sys.path.append('../flowpm/')
sys.path.append('../utils/')
from background import *
import tfpm 
import tfpmfuncs as tfpf
from tfpmconfig import Config
import tensorflow_hub as hub
#from gendata import gendata
import tools
import datatools as dtools
from standardrecon import standardrecon

pad = 2
modpath = '/home/chmodi/Projects/galmodel/code/models/n10/pad2-logistic/module/1546529135/likelihood/'
#modpath = '/home/chmodi/Projects/galmodel/code/models/n10/poisson/module/1547165819/likelihood/'
dpath = './../../data/z00/'
ftype = 'L%04d_N%04d_S%04d_%02dstep/'




def reconmodel(config, data, sigma=0.01**0.5, maxiter=100, gtol=1e-5, anneal=True):

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
        print(final.shape)
        xx = tf.multiply(final, 1.)
        #xx = tf.concat((final[-pad:, :, :], final, final[:pad, :, :]), axis=0)
        #xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
        #xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
        xx = tf.expand_dims(tf.expand_dims(xx, 0), -1)
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
            #print(likelihood.shape)
            likelihoodk = tfpf.r2c3d(likelihood, norm=nc**3)
            likelihoodk = tf.multiply(likelihoodk, tf.cast(smwts, tf.complex64))
            likelihood = tfpf.c2r3d(likelihoodk, norm=nc**3)

        residual = - tf.reduce_sum(likelihood)
        
        #Prior
        lineark = tfpf.r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')

        chisq = tf.multiply(residual, 1/nc**0, name='chisq')

        loss = tf.add(chisq, prior, name='loss')
        
        optimizer = ScipyOptimizerInterface(loss, var_list=[linear], method='L-BFGS-B', 
                                            options={'maxiter': maxiter, 'gtol':gtol})
        
        tf.add_to_collection('inits', [initlin_op, initlin])
        tf.add_to_collection('opt', optimizer)
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate])
        tf.add_to_collection('data', data)
    return g
    






def savefig(truemesh, reconmesh, fname):
    fig, ax = plt.subplots(2, 3, figsize = (12, 8))
    k, pt = tools.power(1+truemesh, boxsize=bs)
    k, pr = tools.power(1+reconmesh, boxsize=bs)
    k, px = tools.power(1+truemesh, 1+reconmesh, boxsize=bs)
    ax[0, 0].semilogx(k, px/(pr*pt)**.5, 'C0')
    ax[1, 0].semilogx(k, pr/pt, 'C0')
    ax[0, 1].loglog(k, pt)
    ax[1, 1].loglog(k, pr)
    ax[0, 2].imshow(truemesh.sum(axis=0))
    ax[1, 2].imshow(reconmesh.sum(axis=0))
    for axis in ax.flatten(): axis.grid(which='both', lw=0.5, color='gray')
    fig.tight_layout()
    fig.savefig(fname)
    

def loss_callback(var, literals, nprint=50, nsave=50, maxiter=500, t0=time()):
    losses = literals['losses']
    loss = var[0]
    mesh = var[1]
    nit = len(losses) %maxiter
    losses.append(loss)
    
    if nit % nprint == 0:
        print('Time taken for iterations %d = '%nit, time() - t0)
        print(nit, " - Loss, chisq, prior, grad : ", loss)
    if nit % nsave == 0:
        np.save(optfolder + '/iter%d.f4'%nit, mesh)
        np.savetxt(optfolder + '/losses.txt', np.array(losses))

        truemesh = literals['truth']
        reconmesh = mesh
        fname = optfolder + '/%d.png'%nit
        savefig(truemesh, reconmesh, fname)




def standardinit(config, base, pos, final, R=8):

    ##
    print('Initial condition from standard reconstruction')
    bs, nc = config['boxsize'], config['nc']
    
    if abs(base.mean()) > 1e-6: 
        base = (base - base.mean())/base.mean()
    pfin = tools.power(final, boxsize=bs)[1]
    ph = tools.power(1+base, boxsize=bs)[1]
    bias = ((ph[1:5]/pfin[1:5])**0.5).mean()
    print('Bias = ', bias)

    g = standardrecon(config, base, pos, bias, R=R)

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        tfdisplaced = g.get_tensor_by_name('displaced:0')
        tfrandom = g.get_tensor_by_name('random:0')

        displaced, random = sess.run([tfdisplaced, tfrandom])

    displaced /= displaced.mean()
    displaced -= 1
    random /= random.mean()
    random -= 1
    recon = displaced - random
    return recon
        
########################

if __name__=="__main__":

    bs, nc = 400, 128
    seed = 100
    step = 5
    ncf, stepf = 512, 40
    numd = 1e-3
    num = int(numd*bs**3)
    #
    maxiter = 1002
    gtol = 1e-8
    sigma = 1**0.5
    nprint, nsave = 50, 250
    anneal = False
    voxels = True
    cube_size = int(32)
    R0s = [4, 2, 1, 0]

    #output folder
    suffix = 'nc0norm/'
    ofolder = './saved/L%04d_N%04d_S%04d_n%02d/'%(bs, nc, seed, numd*1e4)
    if anneal : ofolder += 'anneal%d/'%len(R0s)
    elif voxels: ofolder += 'voxel%d/'%int(cube_size)
    else: ofolder += '/vanilla/'
    ofolder = ofolder + suffix
    try: os.makedirs(ofolder)
    except: pass
    print('Output in ofolder = \n%s'%ofolder)
    pkfile = '../flowpm/Planck15_a1p00.txt'
    config = Config(bs=bs, nc=nc, seed=seed, pkfile=pkfile)


    #Generate Data
    truth = tools.readbigfile(dpath + ftype%(bs, nc, seed, step) + 'mesh/s/')
    print(truth.shape)
    final = tools.readbigfile(dpath + ftype%(bs, nc, seed, step) + 'mesh/d/')
    print(final.shape)
    hposall = tools.readbigfile(dpath + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]    
    hposd = hposall[:num].copy()
    data = tools.paintnn(hposd, bs, nc)

    np.save(ofolder + '/truth.f4', truth)
    np.save(ofolder + '/data.f4', data)

    ###
    #Do reconstruction here
    print('\nDo reconstruction\n')

    #
    initval = None
    initval = np.random.normal(1, 0.5, size=nc**3).reshape(nc, nc, nc).astype(config['dtype'])#truth
    #initval = standardinit(config, data, hposd, final, R=8)
    #initval = tools.readbigfile(dpath + ftype%(bs, nc, 900, step) + 'mesh/s/')
    #initval = np.ones((nc, nc, nc))
    #initval = truth.copy()


    #Split data
    ncube = int(nc/cube_size)
    cube_sizeft = int(cube_size + 2*pad)
    config2 = Config(bs=bs/ncube, nc=cube_sizeft, seed=seed, pkfile=pkfile)
    initvalpad = np.pad(initval, pad, 'wrap')
    truthpad = np.pad(truth, pad, 'wrap')
    initsplit = dtools.splitvoxels([initvalpad], cube_size=cube_sizeft, shift=cube_size, ncube=ncube).astype('float32')[:, :, :, :, 0]
    truthsplit = dtools.splitvoxels([truthpad], cube_size=cube_sizeft, shift=cube_size, ncube=ncube).astype('float32')[:, :, :, :, 0]
    datasplit = dtools.splitvoxels([data], cube_size=cube_size, shift=cube_size, ncube=ncube).astype('float32')[:, :, :, :, 0]
    reconsplit = np.zeros_like(datasplit)
    print(truthsplit.shape, datasplit.shape)
    
    for ii in range(ncube**3):

        print('\nFor voxel %d of %d\n'%(ii, ncube**3))
        recong = reconmodel(config2, datasplit[ii], sigma=sigma, maxiter=maxiter, gtol=gtol, anneal=anneal)    
        losses = []
        literals = {'losses':losses, 'truth':truthsplit[ii]}
        tstart = time()
        lcallback = lambda x: loss_callback(x, literals=literals, nprint=nprint, nsave=nsave, maxiter=maxiter, t0=tstart)
        #

        with tf.Session(graph=recong) as session:
            g = session.graph
            session.run(tf.global_variables_initializer())
            linmesh = g.get_tensor_by_name("linmesh:0")
            optimizer = g.get_collection_ref('opt')[0]
            loss = g.get_tensor_by_name('loss:0')
            chisq = g.get_tensor_by_name('chisq:0')
            grad = tf.norm(tf.gradients(loss, linmesh))
            prior = g.get_tensor_by_name('prior:0')

            if initval is not None:
                print('Do init')
                initlinop = g.get_operation_by_name('initlin_op')
                initlin = g.get_tensor_by_name('initlin:0')
                session.run(initlinop, {initlin:initsplit[ii]})

            init, recon = [], []
            if anneal:
                pass
            else:
                optfolder = ofolder + '/vox%02d'%ii
                try: os.makedirs(optfolder)
                except:pass
                print('Output in ofolder = \n%s'%optfolder)

                init = session.run(linmesh)
                np.save(optfolder + '/init.f4', init)            
                optimizer.minimize(session, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad], linmesh]])
                recon = session.run(linmesh)
                reconsplit[ii] = recon
                np.save(optfolder + '/recon.f4', recon)
                print(truthsplit[ii].shape, recon.shape)
                savefig(truthsplit[ii], recon, optfolder+ 'recon.png')


    initval = dtools.uncubify(np.stack(reconsplit, axis=0), [nc,nc,nc])
    recong = reconmodel(config, data, sigma=sigma, maxiter=maxiter, gtol=gtol, anneal=anneal)    
    losses = []
    literals = {'losses':losses, 'truth':truth[ii]}
    tstart = time()
    lcallback = lambda x: loss_callback(x, literals=literals, nprint=nprint, nsave=nsave, maxiter=maxiter, t0=tstart)
    #
    optfolder = ofolder
    savefig(truth, initval, optfolder+ 'recon.png')
                
