import numpy as np
import matplotlib.pyplot as plt
#
import sys, os
sys.path.append('../utils/')
import tools
import datatools as dtools
from time import time
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
 #
import tensorflow as tf
import tensorflow_hub as hub



defdict = {}
defdict['bs'] = 400
defdict['nc'], defdict['ncf'] = 128, 512
defdict['step'], defdict['stepf'] = 5, 40
defdict['path'] = '../../data/z00/'
defdict['ftype'] = 'L%04d_N%04d_S%04d_%02dstep/'
defdict['ftypefpm'] = 'L%04d_N%04d_S%04d_%02dstep_fpm/'
defdict['numd'] = 1e-3
defdict['R1'] = 3
defdict['R2'] = 3*1.2
#
defdict['num'] = int(defdict['numd']*defdict['bs']**3)
defdict['kny'] = np.pi*defdict['nc']/defdict['bs']
defdict['kk'] = tools.fftk((defdict['nc'], defdict['nc'], defdict['nc']), defdict['bs'])
defdict['seeds'] = [100, 200, 300, 400]
defdict['vseeds'] = [100, 300, 800, 900]
defdict['rprob'] = 0.5



def get_meshes(seed, pdict=defdict):
    for i in pdict.keys(): locals()[i] = pdict[i]

    mesh = {}
    mesh['s'] = tools.readbigfile(path + ftypefpm%(bs, nc, seed, step) + 'mesh/s/')
    partp = tools.readbigfile(path + ftypefpm%(bs, nc, seed, step) + 'dynamic/1/Position/')
    mesh['cic'] = tools.paintcic(partp, bs, ncp)
    #mesh['decic'] = tools.decic(mesh['cic'], kk, kny)
    mesh['R1'] = tools.fingauss(mesh['cic'], kk, R1, kny)
    mesh['R2'] = tools.fingauss(mesh['cic'], kk, R2, kny)
    mesh['GD'] = mesh['R1'] - mesh['R2']

    hmesh = {}
    hpath = path + ftype%(bs, ncf, seed, stepf) + 'FOF/'
    hposd = tools.readbigfile(hpath + 'PeakPosition/')
    massd = tools.readbigfile(hpath + 'Mass/').reshape(-1)*1e10
    #galtype = tools.readbigfile(hpath + 'gal_type/').reshape(-1).astype(bool)
    hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]    
    massall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/Mass/')[1:].reshape(-1)*1e10
    hposd = hposall[:num].copy()
    massd = massall[:num].copy()
    #hmesh['pcic'] = tools.paintcic(hposd, bs, nc)
    hmesh['pnn'] = tools.paintnn(hposd, bs, ncp)
    hmesh['mnn'] = tools.paintnn(hposd, bs, ncp, massd)
    #hmesh['pnnsat'] = tools.paintnn(hposd[galtype], bs, ncp)
    #hmesh['pnncen'] = tools.paintnn(hposd[~galtype], bs, ncp)

    return mesh, hmesh


def generate_training_data(pdict):

    for i in pdict.keys():
        print(i)
        exec(i + "=pdict['" + i +"']")
        locals()[i] = pdict[i]
    locals().update(pdict)
    #print(locals())
    meshes = {}
    cube_features, cube_target = [[] for i in range(len(cube_sizes))], [[] for i in range(len(cube_sizes))]

    for seed in seeds:

        mesh, hmesh = get_meshes(seed)
        meshes[seed] = [mesh, hmesh]

        print('All the mesh have been generated for seed = %d'%seed)

        #Create training voxels
        ftlist = [mesh[i].copy() for i in ftname]
        ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
        targetmesh = [hmesh[i].copy() for i in tgname]

        for i, size in enumerate(cube_sizes):
            print('For size = ', size)
            if size==nc:
                features = [np.stack(ftlistpad, axis=-1)]
                target = [np.stack(targetmesh, axis=-1)]
            else:
                numcubes = int(num_cubes/size*4)
                features, target = dtools.randomvoxels(ftlistpad, targetmesh, numcubes, max_offset[i], 
                                                size, cube_sizesft[i], seed=seed, rprob=0)
            cube_features[i] = cube_features[i] + features
            cube_target[i] = cube_target[i] + target

    # #
    for i in range(cube_sizes.size):
        cube_target[i] = np.stack(cube_target[i],axis=0)
        cube_features[i] = np.stack(cube_features[i],axis=0)
        print(cube_features[i].shape, cube_target[i].shape)

    return meshes, cube_features, cube_target




def check_module(modpath):
    
    print('\nTest module\n')

    tf.reset_default_graph()
    module = hub.Module(modpath + '/likelihood/')
    xx = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
    yy = tf.placeholder(tf.float32, shape=[None, None, None, None, ntargets], name='labels')
    samples = module(dict(features=xx, labels=yy), as_dict=True)['sample']
    loglik = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']

    preds = {}
    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())

        for seed in vseeds:
            xxm = np.stack([np.pad(vmeshes[seed][0][i], pad, 'wrap') for i in ftname], axis=-1)
            #yym = np.stack([np.pad(vmeshes[seed][1]['pnncen'], pad, 'wrap'), np.pad(vmeshes[seed][1]['pnnsat'], pad, 'wrap')], axis=-1)
            yym = np.stack([vmeshes[seed][1][i] for i in tgname], axis=-1)
            print('xxm, yym shape = ', xxm.shape, yym.shape)
            preds[seed] = sess.run(samples, feed_dict={xx:np.expand_dims(xxm, 0), yy:np.expand_dims(yym, 0)})
            vmeshes[seed][0]['predict'] = preds[seed][:, :, :]


    ##############################
    ##Power spectrum
    shape = [nc,nc,nc]
    kk = tools.fftk(shape, bs)
    kmesh = sum(i**2 for i in kk)**0.5

    fig, axar = plt.subplots(2, 2, figsize = (8, 8))
    ax = axar[0]
    for seed in vseeds:
        for i, key in enumerate(['']):
            predict, hpmeshd = vmeshes[seed][0]['predict%s'%key] , vmeshes[seed][1]['pnn%s'%key], 
            k, pkpred = tools.power(predict/predict.mean(), boxsize=bs, k=kmesh)
            k, pkhd = tools.power(hpmeshd/hpmeshd.mean(), boxsize=bs, k=kmesh)
            k, pkhx = tools.power(hpmeshd/hpmeshd.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)    
        ##
            ax[0].semilogx(k[1:], pkpred[1:]/pkhd[1:], label=seed)
            ax[1].semilogx(k[1:], pkhx[1:]/(pkpred[1:]*pkhd[1:])**0.5)
            ax[0].set_title(key, fontsize=12)

    for axis in ax.flatten():
        axis.legend(fontsize=14)
        axis.set_yticks(np.arange(0, 1.2, 0.1))
        axis.grid(which='both')
        axis.set_ylim(0.,1.1)
    ax[0].set_ylabel('Transfer function', fontsize=14)
    ax[1].set_ylabel('Cross correlation', fontsize=14)
    #
    ax = axar[1]
    for i, key in enumerate([ '']):
        predict, hpmeshd = vmeshes[seed][0]['predict%s'%key] , vmeshes[seed][1]['pnn%s'%key], 
        vmin, vmax = 0, (hpmeshd[:, :, :].sum(axis=0)).max()
        im = ax[0].imshow(predict[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        im = ax[1].imshow(hpmeshd[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        ax[0].set_title(key, fontsize=15)
    ax[0].set_title('Prediction', fontsize=15)
    ax[1].set_title('Truth', fontsize=15)
    plt.savefig(savepath + '/vpredict%d.png'%max_steps)
    plt.show()
