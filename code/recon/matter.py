###Test exercise to reconstruct from dark matter

import numpy as np
import numpy
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

anneal = True
suffix = 'matter-ovd/'




if __name__=="__main__":

    bs, nc = 400, 128
    seed = 100
    step = 5
    ncf, stepf = 512, 40
    numd = 1e-3
    num = int(numd*bs**3)
    dpath = './../../data/z00/'
    ftype = 'L%04d_N%04d_S%04d_%02dstep/'
    ftypefpm = 'L%04d_N%04d_S%04d_%02dstep_fpm/'
    #
    maxiter = 505
    gtol = 1e-8
    sigma = 0.01**0.5
    nprint, nsave = 10, 50
    R0s = [4, 2, 1, 0]
    
    #output folder
    ofolder = './saved/L%04d_N%04d_S%04d_n%02d/'%(bs, nc, seed, numd*1e4)
    if anneal : ofolder += 'anneal%d/'%len(R0s)
    else: ofolder += '/noanneal/'
    ofolder = ofolder + suffix
    try: os.makedirs(ofolder)
    except: pass
    print('Output in ofolder = \n%s'%ofolder)
    pkfile = '../flowpm/Planck15_a1p00.txt'
    config = Config(bs=bs, nc=nc, seed=seed, pkfile=pkfile)

    
    #Generate Data
    truth = tools.readbigfile(dpath + ftypefpm%(bs, nc, seed, step) + 'mesh/s/')
    print(truth.shape)
    final = tools.readbigfile(dpath + ftypefpm%(bs, nc, seed, step) + 'mesh/d/')
    print(final.shape)
    data = final/final.mean() - 1
    
    
    #truemeshes = [truth, final, data]
    truemeshes = [truth, final, data]

    np.save(ofolder + '/truth.f4', truth)
    np.save(ofolder + '/final.f4', final)
    np.save(ofolder + '/data.f4', data)

    ###
    #Do reconstruction here
    print('\nDo reconstruction\n')

    recong = rmods.graphdm(config, data, sigma=sigma, maxiter=maxiter, anneal=anneal, dataovd=True)    
    #
    
    initval = None
    #initval = np.random.normal(1, 0.5, size=nc**3).reshape(nc, nc, nc).astype(config['dtype'])#truth
    #initval = standardinit(config, data, hposd, final, R=8)
    #initval = tools.readbigfile(dpath + ftype%(bs, nc, 900, step) + 'mesh/s/')
    #initval = np.ones((nc, nc, nc))
    #initval = truth.copy()


    losses = []
    literals = {'losses':losses, 'truemeshes':truemeshes, 'bs':bs, 'nc':nc}
    tstart = time()
    lcallback = lambda x: loss_callback(x, literals=literals, nprint=nprint, nsave=nsave, maxiter=maxiter, t0=tstart)
    
    with tf.Session(graph=recong) as session:
        g = session.graph
        session.run(tf.global_variables_initializer())
        linmesh = g.get_tensor_by_name("linmesh:0")
        final = g.get_tensor_by_name("final:0")
        samples = final
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
            session.run(initlinop, {initlin:initval})


        def checkiter(mode, optfolder, R0=0):
            print('\nChecking mode = %s\n'%mode)
            meshs, meshf, meshd = session.run([linmesh, final, samples], {Rsm:R0})
            title = session.run([loss, chisq, prior, grad], {Rsm:R0})
            np.save(optfolder + '/%s%d.f4'%(mode, R0), meshs) 
            dg.makefig(literals['truemeshes'], [meshs, meshf, meshd], optfolder+'%s%d.png'%(mode, R0), boxsize=bs, title='%s'%title)

            
        if anneal:

            for R0 in R0s:
                optfolder = ofolder + "/R%02d/"%(R0*10)
                try: os.makedirs(optfolder)
                except:pass
                print('\nAnneal for Rsm = %0.2f\n'%R0)
                print('Output in ofolder = \n%s'%optfolder)

                checkiter('init', optfolder, R0=R0)
                #
                optimizer.minimize(session, {Rsm:R0}, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad], \
                                                                           [linmesh, final, samples]]])
                #
                checkiter('recon', optfolder, R0=R0)

                
        else:
            optfolder = ofolder
            try: os.makedirs(optfolder)
            except:pass
            print('\nNo annealing\n')
            print('Output in ofolder = \n%s'%optfolder)
            
            checkiter('init', optfolder, R0=0)
            ##
            optimizer.minimize(session, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad], \
                                                                           [linmesh, final, samples]]])
            checkiter('recon', optfolder, R0=0)
            #


