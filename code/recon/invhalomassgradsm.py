#For models trained with halo_inv i.e. inv_model_fn, we did not interchange
#order of input features and target as we did for pixelcnn where we did change it
#at input level and not at model level. I think this can be handled here by setting
#the inverse to be false. Not sure. Lets see.


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

from standardrecon import standardinit
import diagnostics as dg
import reconmodels as rmods


def loss_callback(var, literals, nprint=50, nsave=50, maxiter=500, t0=time()):
    print('Callback')
    #losses = literals['losses']
    
    loss = var[0]
    reconmeshes = var[1]
    #nit = len(losses) %(maxiter*2)
    #losses.append(loss)
    nit, losses = var[2]

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

    print('Called')



########################

stellar = True
anneal = True
pad = 0
stdinit = False
truthinit = False
datainit = True
loss = 'loglikelihood'
inverse = False
key = 'mcicnomean'

datacic = True
logit = False
usemass = True
ovd = True
#modpath = '/home/chmodi/Projects/galmodel/code/modelsv2/n10/pad4-specres0p7_noz_poisson/module/1558550045/likelihood/'
#modpath = '//home/chmodi/Projects/galmodel/code/modelsv2/n10/pad4-noz_poisson2/module/1558651873/likelihood/'
#modpath = '/home/chmodi/Projects/galmodel/notebooks/models/galmodel/pix3dcond/module/1561681515/likelihood/'
#modpath = '/home/chmodi/Projects/galmodel/code/modelsv2/n10/pad0-pixinvcic-log/module/1561743877/likelihood/'
#modpath = '//home/chmodi/Projects/galmodel/code/modelsv2/n10/pad0-pixcicinv/module/1562024537/likelihood/'
#modpath  = '/home/chmodi/Projects/galmodel/code/modelsv2/n10/pad0-pixcicinvfmap8/module/1562441352/likelihood/'
#modpath = '/home/chmodi/Projects/galmodel/code/modelsv2/n10/pad0-pixmcicd-invfmap8/module/1562597388/likelihood/'
#modpath = '/home/chmodi/Projects/galmodel/code/modelsv2/n10/pad0-mcicd-inv/module/1562799664/likelihood/'
modpath = '/home/chmodi/Projects/galmodel/code/modelsv3/n10/pad0-pixScicnomean-invfmap8-mix4/module//1567099984/likelihood/'

resnorm = -3
lr0 = 0.01

suffix = 'nc%dnorm-pixinvScicnomeanf8mix4_lr0p01/'%(resnorm)
#suffix = 'nc%dnorm-pix3dinvmcicdf8d_lr0p01/'%(resnorm)
if stdinit : suffix = suffix[:-1] + '-stdinit/'
if truthinit : suffix = suffix[:-1] + '-truth/'
if datainit : suffix = suffix[:-1] + '-datainit/'



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
    gtol = 1e-8
    sigma = 1**0.5
    nprint, nsave = 20, 500
    R0s = [4, 2, 1, 0]
    #maxiter = 505
    #R0s = [0]
    maxiter = 1005
    
    #output folder
    ofolder = './saved/L%04d_N%04d_S%04d_n%02d-v3/'%(bs, nc, seed, numd*1e4)
    ofolder += 'gsm%d/'%len(R0s)
    ofolder = ofolder + suffix
    try: os.makedirs(ofolder)
    except: pass
    print('Output in ofolder = \n%s'%ofolder)
    pkfile = '../flowpm/Planck15_a1p00.txt'
    config = Config(bs=bs, nc=nc, seed=seed, pkfile=pkfile)
    kk = config['kvec']
    kmesh = sum(kk**2 for kk in config['kvec'])**0.5

    #hgraph = dg.graphlintomod(config, modpath, pad=pad, ny=1)
    print('Diagnostic graph constructed')
    fname = open(ofolder+'/README', 'w', 1)
    fname.write('Using module from path - %s \n'%modpath)
    fname.close()
    print('\nUsing module from path - %s \n'%modpath)

    
    #Generate Data
    truth = tools.readbigfile(dpath + ftypefpm%(bs, nc, seed, step) + 'mesh/s/')
    print(truth.shape)
    final = tools.readbigfile(dpath + ftypefpm%(bs, nc, seed, step) + 'mesh/d/')
    print(final.shape)
    hposall = tools.readbigfile(dpath + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]    
    #massall = tools.readbigfile(dpath + ftype%(bs, ncf, seed, stepf) + 'FOF/Mass/')[1:].reshape(-1)*1e10
    if stellar : massall = np.load(dpath + ftype%(bs, ncf, seed, stepf) + 'stellarmass.npy')
    else: massall = tools.readbigfile(dpath + ftype%(bs, ncf, seed, stepf) + 'FOF/Mass/')[1:].reshape(-1)*1e10
    massd = massall[:num].copy()
    hposd = hposall[:num].copy()
    #
    hmesh = {}
    hmesh['pcic'] = tools.paintcic(hposd, bs, nc)
    hmesh['pnn'] = tools.paintnn(hposd, bs, nc)
    hmesh['mnn'] = tools.paintnn(hposd, bs, nc, massd)
    hmesh['mcic'] = tools.paintcic(hposd, bs, nc, massd)
    hmesh['mcicnomean'] =  (hmesh['mcic'] )/hmesh['mcic'].mean()
    hmesh['mcicovd'] =  (hmesh['mcic'] - hmesh['mcic'].mean())/hmesh['mcic'].mean()
    hmesh['pcicovd'] =  (hmesh['pcic'] - hmesh['pcic'].mean())/hmesh['pcic'].mean()
    #hmesh['pcicovdR3'] = tools.fingauss(hmesh['pcicovd'], kk, R1, kny)

    datap = tools.paintcic(hposd, bs, nc)

    data = hmesh[key]
##    if datacic:
##        datam = tools.paintcic(hposd, bs, nc, massd)
##        datap = tools.paintcic(hposd, bs, nc)
##    else:
##        datam = tools.paintnn(hposd, bs, nc, massd)
##        datap = tools.paintnn(hposd, bs, nc)
##    if mcicnomean: hmesh['mcicnomean'] =  (hmesh['mcic'] )/hmesh['mcic'].mean()
##
##    if usemass: data = datam
##    else: data = datap
##    if ovd: data = (data - data.mean())/data.mean()
##

    print(data.min(), data.max(), data.mean(), data.std())
    
    truemeshes = [truth, final, data]
    np.save(ofolder + '/truth.f4', truth)
    np.save(ofolder + '/final.f4', final)
    np.save(ofolder + '/data.f4', data)

    ###
    #Do reconstruction here
    print('\nDo reconstruction\n')

    recong = rmods.graphhposft1smgrads(config, modpath, data, pad,  maxiter=maxiter, gtol=gtol, anneal=anneal, resnorm=resnorm,
                                       loss=loss, log=logit, inverse=inverse)    
    #
    
    initval = None
    initval = np.random.normal(0, 1, size=nc**3).reshape(nc, nc, nc).astype(config['dtype'])#truth
    if stdinit : initval = standardinit(config, datap, hposd, final, R=8)
    #initval = tools.readbigfile(dpath + ftype%(bs, nc, 900, step) + 'mesh/s/')
    #initval = np.ones((nc, nc, nc))
    if truthinit: initval = truth.copy()


    losses = []
    literals = {'losses':losses, 'truemeshes':truemeshes, 'bs':bs, 'nc':nc}
    tstart = time()
    lcallback = lambda x: loss_callback(x, literals=literals, nprint=nprint, nsave=nsave, maxiter=maxiter, t0=tstart)
    
    with tf.Session(graph=recong) as session:
        g = session.graph
        session.run(tf.global_variables_initializer())
        linmesh = g.get_tensor_by_name("linmesh:0")
        final = g.get_tensor_by_name("final:0")
        samples = tf.squeeze(g.get_tensor_by_name("samples:0"))
        optimizer = g.get_collection_ref('opt')[0][0]
        #opt_op = g.get_collection_ref('opt')[0][1]
        loss = g.get_tensor_by_name('loss:0')
        chisq = g.get_tensor_by_name('chisq:0')
        grad = tf.norm(tf.gradients(loss, linmesh))
        prior = g.get_tensor_by_name('prior:0')
        Rsm = g.get_tensor_by_name('smoothing:0')
        lr = g.get_tensor_by_name('learning_rate:0')
        gradandvars_chisq = g.get_collection_ref('grads')[0][0]
        gradandvars_prior = g.get_collection_ref('grads')[0][1]

                                                                                      
##        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)                
##        gradandvars_chisq = optimizer.compute_gradients(chisq, linmesh)                 
##        gradandvars_prior = optimizer.compute_gradients(prior, linmesh)
##        opt_op = optimizer.apply_gradients(grads_and_vars1, name='apply_grad')         
##
        gradandvars_new = []

        for i in range(len(gradandvars_chisq)):
            g1, v = gradandvars_chisq[i]
            g2, _ = gradandvars_prior[i]

            if len(R0s) > 1:
                gk = tfpf.r2c3d(g1, norm=nc**3)
                smwts = tf.exp(tf.multiply(-0.5*kmesh**2, tf.multiply(Rsm*bs/nc, Rsm*bs/nc)))       
                gk = tf.multiply(gk, tf.cast(smwts, tf.complex64))
                g1 = tfpf.c2r3d(gk, norm=nc**3)
                
            gradandvars_new.append((g1+g2, v))
       
        applygrads = optimizer.apply_gradients(gradandvars_new)

        if datainit:
            initval = session.run(samples, {Rsm:0})
        
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
            print('Checked iter')
            
        for R0 in R0s:
            optfolder = ofolder + "/R%02d/"%(R0*10)
            try: os.makedirs(optfolder)
            except: pass
            print('\nAnneal for Rsm = %0.2f\n'%R0)
            print('Output in ofolder = \n%s'%optfolder)

            #checkiter('init', optfolder, R0=R0)
            #
            for nit in range(maxiter):
                #_ = session.run(applygrads, feed_dict={Rsm:R0, lr:0.1})
                
                l, _ = session.run([[loss, chisq, prior], applygrads], feed_dict={Rsm:R0, lr:lr0})
                losses.append(l)
                if nit % nprint == 0:
                    print(nit, l)
                    lss, csq, pr, meshs, meshf, meshd = session.run([loss, chisq, prior,  linmesh, final, samples],
                                                                    feed_dict={Rsm:R0, lr:0.1}) 
                    var = [[lss, csq, pr], [meshs, meshf, meshd], [nit, losses]]
                    lcallback(var)
#
            checkiter('recon', optfolder, R0=R0)


