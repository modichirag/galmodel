import numpy as np
import numpy
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from time import time
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import tensorflow_hub as hub

sys.path.append('../flowpm/')
from background import *
import tfpm 
import tfpmfuncs as tfpf
from tfpmconfig import Config

sys.path.append('../utils/')
import tools
import datatools as dtools
from standardrecon import standardinit
import diagnostics as dg
import reconmodels as rmods


def loss_callback(var, literals, nprint=50, nsave=50, maxiter=500, t0=time()):
    losses = literals['losses']
    loss = var[0]
    reconmeshes = var[1]
    nit = len(losses) %(maxiter*2)
    losses.append(loss)


    if nit % nprint == 0:
        print('Time taken for iterations %d = '%nit, time() - t0)
        print(nit, " - Loss, chisq, prior, grad : ", loss)

        fname = optfolder + '/%d.png'%nit
        stime = time()
        #dg.savehalofig(literals['truemeshes'], reconmeshes[0], fname, literals['hgraph'], boxsize=bs, title='%s'%loss)
        dg.makefig(literals['truemeshes'], reconmeshes, fname, boxsize=bs, title='%s'%loss)    
        print('Time taken to make figure = ', time()-stime)
        
    if nit % nsave == 0:
        np.save(optfolder + '/iter%d.f4'%nit, reconmeshes)
        np.savetxt(optfolder + '/losses.txt', np.array(losses))

        
########################

anneal = False
pad = 0
#modpath = '/home/chmodi/Projects/galmodel/code/models/n10/pad2-logistic128/module/1547856591/likelihood/'
modpath = '/home/chmodi/Projects/galmodel/code/models/n10/pad0-vireg-reg0p1/module/1547930039/likelihood/'
#modpath = '/home/chmodi/Projects/galmodel/code/models/n10/pad0-vireg-reg0p5/module/1547949556/likelihood/'
modpath = '/home/chmodi/Projects/galmodel/code/models/n10/pad0-vireg-reg1p0/module/1547981391/likelihood/'
modpath = '/home/chmodi/Projects/galmodel/code/models/n10/rev_nn_l2/module/1548456835/likelihood/'
loss = 'l2'
sample = 'lambda'

resnorm = 0
#suffix = 'nc%dnorm-fpm/'%(resnorm)
suffix = 'nc%dnorm-rev_nn_l2/'%(resnorm)


########################

if __name__=="__main__":

    bs, nc = 400, 128
    seed = 100
    step = 5
    ncf, stepf = 512, 40
    numd = 1e-3
    num = int(numd*bs**3)
    dpath = './../../data/z00/'
    ftype = 'L%04d_N%04d_S%04d_%02dstep/'
    #
    maxiter = 102
    gtol = 1e-8
    sigma = 1**0.5
    nprint, nsave = 50, 100
    anneal = False
    voxels = True
    cube_size = int(64)
    R0s = [4, 2, 1, 0]

    #output folder
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

    truemeshes = [truth, final, data]
    np.save(ofolder + '/truth.f4', truth)
    np.save(ofolder + '/final.f4', final)
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
    #initvalpad = np.pad(initval, pad, 'wrap')
    #truthpad = np.pad(truth, pad, 'wrap')
    #truthpad = np.pad(truth, pad, 'wrap')
    #initsplit = dtools.splitvoxels([initvalpad], cube_size=cube_sizeft, shift=cube_size, ncube=ncube).astype('float32')[:, :, :, :, 0]
    #truthsplit = dtools.splitvoxels([truthpad], cube_size=cube_sizeft, shift=cube_size, ncube=ncube).astype('float32')[:, :, :, :, 0]
    initsplit = dtools.splitvoxels([initval], cube_size=cube_size, shift=cube_size, ncube=ncube).astype('float32')[:, :, :, :, 0]
    truthsplit = dtools.splitvoxels([truth], cube_size=cube_size, shift=cube_size, ncube=ncube).astype('float32')[:, :, :, :, 0]
    datasplit = dtools.splitvoxels([data], cube_size=cube_size, shift=cube_size, ncube=ncube).astype('float32')[:, :, :, :, 0]
    finalsplit = dtools.splitvoxels([final], cube_size=cube_size, shift=cube_size, ncube=ncube).astype('float32')[:, :, :, :, 0]
    reconsplit = np.zeros_like(datasplit)
    print(truthsplit.shape, datasplit.shape)
    
    for ii in range(ncube**3):

        truemeshii = [truthsplit[ii], finalsplit[ii], datasplit[ii]]
        print('\nFor voxel %d of %d\n'%(ii, ncube**3))
        recong = rmods.graphhposft1(config2, modpath, datasplit[ii], pad,  maxiter=maxiter, gtol=gtol, anneal=anneal, resnorm=resnorm,
                                    inference=False,loss=loss, sample=sample)    
        #recong = rmods.graphhposft1(config2, modpath, datasplit[ii], pad,  maxiter=maxiter, gtol=gtol, anneal=anneal, resnorm=resnorm)    
        losses = []
        literals = {'losses':losses, 'truemeshes':truemeshii, 'bs':bs, 'nc':nc}
        tstart = time()
        lcallback = lambda x: loss_callback(x, literals=literals, nprint=nprint, nsave=nsave, maxiter=maxiter, t0=tstart)
        #

        with tf.Session(graph=recong) as session:
            g = session.graph
            session.run(tf.global_variables_initializer())
            linmesh = g.get_tensor_by_name("linmesh:0")
            samples = tf.squeeze(g.get_tensor_by_name("samples:0"))
            final = g.get_tensor_by_name("final:0")
            optimizer = g.get_collection_ref('opt')[0]
            loss = g.get_tensor_by_name('loss:0')
            chisq = g.get_tensor_by_name('chisq:0')
            grad = tf.norm(tf.gradients(loss, linmesh))
            prior = g.get_tensor_by_name('prior:0')
            Rsm = g.get_tensor_by_name('smoothing:0')
        

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

                meshs, meshf, meshd = session.run([linmesh, final, samples])
                title = session.run([loss, chisq, prior, grad])
                np.save(optfolder + '/init.f4', meshs)            
                dg.makefig(literals['truemeshes'], [meshs, meshf, meshd], optfolder+'%s.png'%('init'), boxsize=bs, title='%s'%title)
                optimizer.minimize(session, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad],
                                                                               [linmesh, final, samples]]])

                meshs, meshf, meshd = session.run([linmesh, final, samples])
                title = session.run([loss, chisq, prior, grad])
                reconsplit[ii] = meshs
                np.save(optfolder + '/recon.f4', meshs)
                dg.makefig(literals['truemeshes'], [meshs, meshf, meshd], optfolder+'%s.png'%('recon'), boxsize=bs, title='%s'%title)


    initval = dtools.uncubify(np.stack(reconsplit, axis=0), [nc,nc,nc])
    recong = rmods.graphhposft1(config, modpath, data, pad,  maxiter=maxiter, gtol=gtol, anneal=anneal, resnorm=resnorm)    
    losses = []
    literals = {'losses':losses, 'truemeshes':truemeshes, 'bs':bs, 'nc':nc}
    tstart = time()
    lcallback = lambda x: loss_callback(x, literals=literals, nprint=nprint, nsave=nsave, maxiter=maxiter, t0=tstart)
    #

    with tf.Session(graph=recong) as session:
        g = session.graph
        session.run(tf.global_variables_initializer())
        linmesh = g.get_tensor_by_name("linmesh:0")
        samples = tf.squeeze(g.get_tensor_by_name("samples:0"))
        final = g.get_tensor_by_name("final:0")
        optimizer = g.get_collection_ref('opt')[0]
        loss = g.get_tensor_by_name('loss:0')
        chisq = g.get_tensor_by_name('chisq:0')
        grad = tf.norm(tf.gradients(loss, linmesh))
        prior = g.get_tensor_by_name('prior:0')
        Rsm = g.get_tensor_by_name('smoothing:0')


        initlinop = g.get_operation_by_name('initlin_op')
        initlin = g.get_tensor_by_name('initlin:0')
        session.run(initlinop, {initlin:initval})

        optfolder = ofolder + '/total/'
        try: os.makedirs(optfolder)
        except:pass
        print('Output in ofolder = \n%s'%optfolder)
        
        meshs, meshf, meshd = session.run([linmesh, final, samples])
        title = session.run([loss, chisq, prior, grad])
        np.save(optfolder + '/init.f4', meshs)            
        dg.makefig(literals['truemeshes'], [meshs, meshf, meshd], optfolder+'%s.png'%('init'), boxsize=bs, title='%s'%title)
        optimizer.minimize(session, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad],
                                                                       [linmesh, final, samples]]])

        meshs, meshf, meshd = session.run([linmesh, final, samples])
        title = session.run([loss, chisq, prior, grad])
        np.save(optfolder + '/recon.f4', meshs)
        dg.makefig(literals['truemeshes'], [meshs, meshf, meshd], optfolder+'%s.png'%('recon'), boxsize=bs, title='%s'%title)

