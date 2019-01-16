import numpy as np
import numpy
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import tools
from standardrecon import standardrecon

pad = 2
#modpath = '/home/chmodi/Projects/galmodel/code/models/n10/pad2-logistic/module/1546529135/likelihood/'
#modpath = '/home/chmodi/Projects/galmodel/code/models/n10/poisson/module/1547165819/likelihood/'
modpath = '/home/chmodi/Projects/galmodel/code/models/galmodel/pad2-logistic/module/1547112461/likelihood/'
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
        #xx = tf.reshape(final, shape=[-1, cube_sizeft, cube_sizeft, cube_sizeft, nchannels], name='input')
        xx = tf.concat((final[-pad:, :, :], final, final[:pad, :, :]), axis=0)
        xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
        xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
        xx = tf.expand_dims(tf.expand_dims(xx, 0), -1)
        #Halos
        #yy = tf.reshape(data, shape=[-1, cube_size, cube_size, cube_size, 1], name='labels')
        yy = tf.expand_dims(data, 0)
        
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

        truemesh = np.load(ofolder + '/truth.f4.npy')
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
    maxiter = 100
    gtol = 1e-8
    sigma = 1**0.5
    nprint, nsave = 25, 25
    anneal = False
    R0s = [4, 2, 1, 0]

    #output folder
    suffix = 'nc0norm/'
    ofolder = './saved/L%04d_N%04d_S%04d_galmodel/'%(bs, nc, seed)
    if anneal : ofolder += 'anneal%d/'%len(R0s)
    else: ofolder += '/noanneal/'
    ofolder = ofolder + suffix
    try: os.makedirs(ofolder)
    except: pass
    print('Output in ofolder = \n%s'%ofolder)
    pkfile = '../flowpm/Planck15_a1p00.txt'
    config = Config(bs=bs, nc=nc, seed=seed, pkfile=pkfile)


    #Generate Data
    truth = tools.readbigfile(dpath + ftype%(bs, nc, seed, step) + 'mesh/s/')
    final = tools.readbigfile(dpath + ftype%(bs, nc, seed, step) + 'mesh/d/')
    #
    hpath = dpath + ftype%(bs, ncf, seed, stepf) + 'galaxies_n05/galcat/'
    hposd = tools.readbigfile(hpath + 'Position/')
    massd = tools.readbigfile(hpath + 'Mass/').reshape(-1)*1e10
    galtype = tools.readbigfile(hpath + 'gal_type/').reshape(-1).astype(bool)
    allgal = tools.paintnn(hposd, bs, nc)
    satmesh = tools.paintnn(hposd[galtype], bs, nc)
    cenmesh = tools.paintnn(hposd[~galtype], bs, nc)
    data = np.stack((cenmesh, satmesh), axis=-1)

    np.save(ofolder + '/truth.f4', truth)
    np.save(ofolder + '/data.f4', data)

    ###
    #Do reconstruction here
    print('\nDo reconstruction\n')

    recong = reconmodel(config, data, sigma=sigma, maxiter=maxiter, gtol=gtol, anneal=anneal)    
    #
    
    initval = None
    initval = np.random.normal(1, 0.5, size=nc**3).reshape(nc, nc, nc).astype(config['dtype'])#truth
    #initval = standardinit(config, data, hposd, final, R=8)
    #initval = tools.readbigfile(dpath + ftype%(bs, nc, 900, step) + 'mesh/s/')
    #initval = np.ones((nc, nc, nc))
    #initval = truth.copy()


    losses = []
    literals = {'losses':losses}
    tstart = time()
    lcallback = lambda x: loss_callback(x, literals=literals, nprint=nprint, nsave=nsave, maxiter=maxiter, t0=tstart)
    
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
            session.run(initlinop, {initlin:initval})

        init, recon = [], []
        if anneal:
            Rsm = g.get_tensor_by_name('smoothing:0')
            for R0 in R0s:
                optfolder = ofolder + "/R%02d/"%(R0*10)
                try: os.makedirs(optfolder)
                except:pass
                print('\nAnneal for Rsm = %0.2f\n'%R0)
                print('Output in ofolder = \n%s'%optfolder)
                init.append(session.run(linmesh))
                np.save(optfolder + '/init%d.f4'%R0, init[-1])
                optimizer.minimize(session, {Rsm:R0}, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad], linmesh]])
                recon.append(session.run(linmesh))
                np.save(optfolder + '/recon%d.f4'%R0, recon[-1])
                savefig(truth, recon[-1], optfolder+ 'recon%d.png'%R0)

                
        else:
            optfolder = ofolder
            try: os.makedirs(optfolder)
            except:pass
            print('\nNo annealing\n')
            print('Output in ofolder = \n%s'%optfolder)
            
            init = session.run(linmesh)
            np.save(optfolder + '/init.f4', init)            
            optimizer.minimize(session, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad], linmesh]])
            recon = session.run(linmesh)
            np.save(optfolder + '/recon.f4', recon)
            savefig(truth, recon, optfolder+ 'recon.png')
