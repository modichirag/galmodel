import numpy as np
import matplotlib.pyplot as plt
#
import sys, os
sys.path.append('../utils/')
import tools
import datatools as dtools
from time import time
from pixelcnn3d import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"
 #
import tensorflow as tf
from tensorflow.contrib.slim import add_arg_scope
from layers import wide_resnet
import tensorflow_hub as hub

import tensorflow.contrib.slim as slim
from layers import wide_resnet, wide_resnet_snorm, valid_resnet
from layers import  wide_resnet_snorm
import tensorflow_probability
import tensorflow_probability as tfp
tfd = tensorflow_probability.distributions
tfd = tfp.distributions
tfb = tfp.bijectors


import models
import logging
from datetime import datetime

#############################
seed_in = 3
from numpy.random import seed
seed(seed_in)
from tensorflow import set_random_seed
set_random_seed(seed_in)

bs = 400
nc, ncf = 128, 512
step, stepf = 5, 40
path = '../../data/z00/'
ftype = 'L%04d_N%04d_S%04d_%02dstep/'
ftypefpm = 'L%04d_N%04d_S%04d_%02dstep_fpm/'
numd = 1e-3
num = int(numd*bs**3)
R1 = 10
R2 = 3*1.2
kny = np.pi*nc/bs
kk = tools.fftk((nc, nc, nc), bs)
#seeds = [100]
seeds = [100, 200, 300, 400, 500, 600, 700]
vseeds = [100, 300, 800, 900]

#############################

suff = 'pad0-pixpmnnd-invf8'

normwt = 0.7
nfilter0 = 1
pad = int(0)
#fname = open('../models/n10/README', 'a+', 1)
#fname.write('%s \t :\n\tModel to predict halo position likelihood in halo_logistic with data supplemented by size=8, 16, 32, 64, 128; rotation with probability=0.5 and padding the mesh with 2 cells. Also reduce learning rate in piecewise constant manner. n_y=1 and high of quntized distribution to 3. Init field as 1 feature & high learning rate\n'%suff)
#fname.close()

savepath = '../modelsv2/galmodel/%s/'%suff
try : os.makedirs(savepath)
except: pass


fname = open(savepath + 'log', 'w+', 1)
#fname = None
num_cubes= 2000
cube_sizes = np.array([16, 32]).astype(int)
nsizes = len(cube_sizes)
pad = int(0)
cube_sizesft = (cube_sizes + 2*pad).astype(int)
max_offset = nc - cube_sizes
#ftname = ['ciclog']
#tgname = ['pcic']
ftname = ['pnnovd', 'mnnovd']
tgname = ['cic']
nchannels = len(ftname)
ntargets = len(tgname)

batch_size = 64
rprob = 0.5

print('Features are ', ftname, file=fname)
print('Pad with ', pad, file=fname)
print('Rotation probability = %0.2f'%rprob, file=fname)
fname.close()

#############################
##Read data and generate meshes



def get_meshes(seed, galaxies=False, inverse=True):
    mesh = {}
    mesh['s'] = tools.readbigfile(path + ftypefpm%(bs, nc, seed, step) + 'mesh/s/')
    partp = tools.readbigfile(path + ftypefpm%(bs, nc, seed, step) + 'dynamic/1/Position/')
    mesh['cic'] = tools.paintcic(partp, bs, nc)
    mesh['ciclog'] = np.log(1e-4 + mesh['cic'])
##    mesh['cicovd'] = mesh['cic']/mesh['cic'].mean()-1
##    mesh['decic'] = tools.decic(mesh['cic'], kk, kny)
##    mesh['R1'] = tools.fingauss(mesh['cic'], kk, R1, kny)
##    mesh['R2'] = tools.fingauss(mesh['cic'], kk, R2, kny)
##    mesh['GD'] = mesh['R1'] - mesh['R2']
##
    hmesh = {}
    hpath = path + ftype%(bs, ncf, seed, stepf) + 'galaxies_n05/galcat/'
    hposd = tools.readbigfile(hpath + 'Position/')
    massd = tools.readbigfile(hpath + 'Mass/').reshape(-1)*1e10
    galtype = tools.readbigfile(hpath + 'gal_type/').reshape(-1).astype(bool)
    hmesh['pnn'] = tools.paintnn(hposd, bs, nc)
    hmesh['pnnovd'] =  (hmesh['pnn'] - hmesh['pnn'].mean())/hmesh['pnn'].mean()
    hmesh['pcic'] = tools.paintcic(hposd, bs, nc)
    hmesh['pcicovd'] =  (hmesh['pcic'] - hmesh['pcic'].mean())/hmesh['pcic'].mean()
    hmesh['mnn'] = tools.paintnn(hposd, bs, nc, massd)
    hmesh['mnnovd'] =  (hmesh['mnn'] - hmesh['mnn'].mean())/hmesh['mnn'].mean()
    hmesh['mcic'] = tools.paintcic(hposd, bs, nc, massd)
    hmesh['mcicovd'] =  (hmesh['mcic'] - hmesh['mcic'].mean())/hmesh['mcic'].mean()
##    hmesh['mnn'] = tools.paintnn(hposd, bs, nc, massd)
##    hmesh['pnnsat'] = tools.paintnn(hposd[galtype], bs, nc)
##    hmesh['pnncen'] = tools.paintnn(hposd[~galtype], bs, nc)
##
##
    if inverse: return hmesh, mesh
    else: return mesh, hmesh



def generate_training_data():
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
                if size == 16: numcubes = 100
                features, target = dtools.randomvoxels(ftlistpad, targetmesh, numcubes, max_offset[i], 
                                                size, cube_sizesft[i], seed=seed, rprob=0)
            cube_features[i] = cube_features[i] + features
            cube_target[i] = cube_target[i] + target

     
    for i in range(cube_sizes.size):
        cube_target[i] = np.stack(cube_target[i],axis=0)
        cube_features[i] = np.stack(cube_features[i],axis=0)
        print(cube_features[i].shape, cube_target[i].shape)

    return meshes, cube_features, cube_target



#############################

def _mdn_pixmodel_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, 
                     cfilter_size=None, f_map=8):

    # Check for training mode                                                                                                   
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    def _module_fn():
        """                                                                                                                     
        Function building the module                                                                                            
        """

        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        conditional_im = wide_resnet(feature_layer, 16, activation_fn=tf.nn.leaky_relu, 
                                     keep_prob=dropout, is_training=is_training)
        conditional_im = wide_resnet(conditional_im, 16, activation_fn=tf.nn.leaky_relu, 
                                      keep_prob=dropout, is_training=is_training)
        conditional_im = wide_resnet(conditional_im, 1, activation_fn=tf.nn.leaky_relu, 
                                      keep_prob=dropout, is_training=is_training)
        conditional_im = tf.concat((feature_layer, conditional_im), -1)

        # Builds the neural network                                                                                             
        ul = [[obs_layer]]
        for i in range(10):
            ul.append(PixelCNN3Dlayer(i, ul[i], f_map=f_map, full_horizontal=True, h=None, 
                                      conditional_im=conditional_im, cfilter_size=cfilter_size, gatedact='sigmoid'))


        h_stack_in = ul[-1][-1]
        
        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, 1, 1], h_stack_in, orientation=None, gated=False, mask='b').output()

        with tf.variable_scope("fc_2"):
            fc2 = GatedCNN([1, 1, 1, n_mixture*3*n_y], fc1, orientation=None, 
                                gated=False, mask='b', activation=False).output()

        
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(fc2, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])

        loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=3,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale) + 1e-3

        # Form mixture of discretized logistic distributions. Note we shift the                                                 
        # logistic distribution by -0.5. This lets the quantization capture "rounding"                                          
        # intervals, `(x-0.5, x+0.5]`, and not "ceiling" intervals, `(x-1, x]`.                                                 
#         discretized_logistic_dist = tfd.QuantizedDistribution(
#             distribution=tfd.TransformedDistribution(
#                 distribution=tfd.Logistic(loc=loc, scale=scale),
#                 bijector=tfb.AffineScalar(shift=-0.5)),
#             low=0.,
#             high=2.**3-1)

        mixture_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=tfd.Normal(loc, scale))

        # Define a function for sampling, and a function for estimating the log likelihood                                      
        #sample = tf.squeeze(mixture_dist.sample())                                                                             
        sample = mixture_dist.sample()
        loglik = mixture_dist.log_prob(obs_layer)
        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer},
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'loc':loc, 'scale':scale, 'logits':logits})

    # Create model and register module if necessary                                                                     
    spec = hub.create_module_spec(_module_fn)
    module = hub.Module(spec, trainable=True)
    if isinstance(features,dict):
        predictions = module(features, as_dict=True)
    else:
        predictions = module({'features':features, 'labels':labels}, as_dict=True)

    if mode == tf.estimator.ModeKeys.PREDICT:
        hub.register_module_for_export(module, "likelihood")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loglik = predictions['loglikelihood']
    # Compute and register loss function                                                                                
    neg_log_likelihood = -tf.reduce_sum(loglik, axis=-1)
    neg_log_likelihood = tf.reduce_mean(neg_log_likelihood)

    tf.losses.add_loss(neg_log_likelihood)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer                                                                                                  
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([500, 1e3, 3e3, 1e4, 2e4, 3e4, 4e4]).astype(int))
            values = [1e-3, 1e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)
        tf.summary.scalar('loss', neg_log_likelihood)
    elif mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)




class MDNEstimator(tf.estimator.Estimator):
    """An estimator for distribution estimation using Mixture Density Networks.
    """

    def __init__(self,
                 nchannels,
                 n_y,
                 n_mixture,
                 optimizer=tf.train.AdamOptimizer,
                 dropout=None,
                 model_dir=None,
                 config=None):
        """Initializes a `MDNEstimator` instance.
        """


        def _model_fn(features, labels, mode):
            return _mdn_pixmodel_fn(features, labels, 
                                 nchannels, n_y, n_mixture, dropout,
                                            optimizer, mode, pad)


        super(self.__class__, self).__init__(model_fn=_model_fn,
                                             model_dir=model_dir,
                                             config=config)





def mapping_function(inds):
    def extract_batch(inds):
        
        #isize = np.random.choice(len(cube_sizes), 1, replace=True)[0]
        isize = np.random.choice(np.arange(1, len(cube_sizes)).astype('int'), 1, replace=True)[0]
        batch = int(batch_size*8/cube_sizes[isize])
        if cube_sizes[isize]==nc : batch = 1
        inds = inds[:batch]
        trainingsize = cube_features[isize].shape[0]
        inds[inds >= trainingsize] =  (inds[inds >= trainingsize])%trainingsize
        
        features = cube_features[isize][inds].astype('float32')
        targets = cube_target[isize][inds].astype('float32')
        
        for i in range(batch):
            nrotations=0
            while (np.random.random() < rprob) & (nrotations < 3):
                nrot, ax0, ax1 = np.random.randint(0, 3), *np.random.permutation((0, 1, 2))[:2]
                features[i] = np.rot90(features[i], nrot, (ax0, ax1))
                targets[i] = np.rot90(targets[i], nrot, (ax0, ax1))
                nrotations +=1
# #             print(isize, i, nrotations, targets[i].shape)
# #         print(inds)
        return features, targets
    
    ft, tg = tf.py_func(extract_batch, [inds],
                        [tf.float32, tf.float32])
    return ft, tg

def training_input_fn():
    """Serving input fn for training data"""

    dataset = tf.data.Dataset.range(len(np.array(cube_features)[0]))
    dataset = dataset.repeat().shuffle(1000).batch(batch_size)
    dataset = dataset.map(mapping_function)
    dataset = dataset.prefetch(16)
    return dataset

def testing_input_fn():
    """Serving input fn for testing data"""
    dataset = tf.data.Dataset.range(len(cube_features))
    dataset = dataset.batch(16)
    dataset = dataset.map(mapping_function)
    return dataset


        
#############################################################################
###save


def save_module(model, savepath, max_steps):

    print('\nSave module\n')

    features = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
    labels = tf.placeholder(tf.float32, shape=[None, None, None, None, ntargets], name='labels')
    exporter = hub.LatestModuleExporter("tf_hub", tf.estimator.export.build_raw_serving_input_receiver_fn({'features':features, 'labels':labels},
                                                                       default_batch_size=None))
    modpath = exporter.export(model, savepath + 'module', model.latest_checkpoint())
    modpath = modpath.decode("utf-8") 
    check_module(modpath)

    
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
            print('xxm :', xxm.mean(), xxm.std())
            print('yym :', yym.mean(), yym.std())
            preds[seed] = sess.run(samples, feed_dict={xx:np.expand_dims(xxm, 0), yy:0*np.expand_dims(yym, 0)})
            vmeshes[seed][0]['predict'] = np.squeeze(preds[seed])

            
    ##############################
    ##Power spectrum
    shape = [nc,nc,nc]
    kk = tools.fftk(shape, bs)
    kmesh = sum(i**2 for i in kk)**0.5

    fig, axar = plt.subplots(2, 2, figsize = (8, 8))
    ax = axar[0]
    for seed in vseeds:
        for i, key in enumerate(['']):
            predict, hpmeshd = vmeshes[seed][0]['predict%s'%key] , vmeshes[seed][1][tgname[0]], 
            if predict.mean() <1e-3 : predict += 1
            if hpmeshd.mean() <1e-3 : hpmeshd += 1
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
        predict, hpmeshd = vmeshes[seed][0]['predict%s'%key] , vmeshes[seed][1][tgname[0]], 
        vmin, vmax = 0, (hpmeshd[:, :, :].sum(axis=0)).max()
        #im = ax[0].imshow(predict[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        #im = ax[1].imshow(hpmeshd[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        im = ax[0].imshow(predict[:, :, :].sum(axis=0))
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(hpmeshd[:, :, :].sum(axis=0))
        plt.colorbar(im, ax=ax[1])
        ax[0].set_title(key, fontsize=15)
    ax[0].set_title('Prediction', fontsize=15)
    ax[1].set_title('Truth', fontsize=15)
    plt.savefig(savepath + '/vpredict%d.png'%max_steps)
    plt.show()

    plt.figure()
    plt.hist(vmeshes[100][0]['predict'].flatten(), range=(-5, 5), bins=100)
    plt.hist(vmeshes[100][1][tgname[0]].flatten(), alpha=0.5, range=(-5, 5), bins=100)
    plt.savefig(savepath + '/hist%d.png'%max_steps)
    plt.show()
    
    dosampletrue = False
    if max_steps in [50, 100, 500, 1000, 5000, 15000, 25000, 35000, 45000, 55000, 65000]:
        dosampletrue = True
        csize = 16
    if max_steps in [3000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]:
        dosampletrue = True
        csize = 32
    if dosampletrue: sampletrue(modpath, csize)


def sampletrue(modpath, csize):
    print('sampling true')
    tf.reset_default_graph()
    module = hub.Module(modpath + '/likelihood/')
    xx = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
    yy = tf.placeholder(tf.float32, shape=[None, None, None, None, ntargets], name='labels')
    samples = module(dict(features=xx, labels=yy), as_dict=True)['sample']
    loglik = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']

    index = np.where(cube_sizes == csize)[0][0]
    
    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        start = time()

        #sess.run(tf.initializers.global_variables())
        features = cube_features[index][0:1].astype('float32')
        targets = cube_target[index][0:1].astype('float32')
        xxm = features
        yym = targets
        print(xxm.shape, yym.shape)
        sample = np.zeros_like(yym)
        sample2 = sess.run(samples, feed_dict={xx:xxm, yy:yym*0})
        for i in range(yym.shape[1]):
            for j in range(yym.shape[2]):
                for k in range(yym.shape[3]):
                    data_dict = {xx:xxm, yy:sample}
                    next_sample = sess.run(samples, feed_dict=data_dict)
                    sample[:, i, j, k, :] = next_sample[:, i, j, k, :]
                        
        end = time()
        print('Taken : ', end-start)
        
    
    plt.figure(figsize = (12, 4))
    vmin, vmax = None, None
    plt.subplot(131)
    plt.imshow(yym[0,...,0].sum(axis=0), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(sample[0,...,0].sum(axis=0), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(sample2[0,...,0].sum(axis=0), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(savepath + '/sampletrue_im%d'%max_steps)

    plt.figure()
    plt.hist(sample2.flatten(), range=(-5, 5), bins=100, label='predict', alpha=0.8)
    plt.hist(yym[0,...,0].flatten(),  range=(-5, 5), bins=100, label='target', alpha=0.5)
    plt.hist(sample.flatten(), range=(-5, 5), bins=100, label='predicttrue', alpha=0.5)
    plt.legend()
    plt.savefig(savepath + '/truehist%d.png'%max_steps)
    plt.show()

    ##
    ii = 0
    k, ph = tools.power(yym[ii,...,], boxsize=bs/cube_sizes[index])
    k, pp1 = tools.power(sample[ii,...,], boxsize=bs/cube_sizes[index])
    k, pp1x = tools.power(sample[ii,...,], yym[ii,...,], boxsize=bs/cube_sizes[index])
    k, pp2 = tools.power(sample2[ii,...,], boxsize=bs/cube_sizes[index])
    k, pp2x = tools.power(sample2[ii,...,], yym[ii,...,], boxsize=bs/cube_sizes[index])
    
    
    plt.figure(figsize = (10, 4))
    plt.subplot(121)
    # plt.plot(k, ph, 'C%d-'%ii)
    plt.plot(k, pp1/ph, 'C%d-'%ii)
    plt.plot(k, pp2/ph, 'C%d--'%ii)
    plt.ylim(0, 1.5)
    plt.grid(which='both')
    plt.semilogx()
    # plt.loglog()
    
    plt.subplot(122)
    plt.plot(k, pp1x/(pp1*ph)**0.5, 'C%d-'%ii)
    plt.plot(k, pp2x/(pp2*ph)**0.5, 'C%d--'%ii)
    plt.ylim(0, 1)
    plt.grid(which='both')
    plt.semilogx()
    plt.savefig(savepath + '/sampletrue_2pt%d'%max_steps)
#


############################################################################
#############---------MAIN---------################

meshes, cube_features, cube_target = generate_training_data()
print('Features :', cube_features[0].mean(), cube_features[0].std())
print('Target :', cube_target[0].mean(), cube_target[0].std())

for i in cube_features: print(i.min(), i.max())
for i in cube_target: print(i.min(), i.max())
vmeshes = {}
for seed in vseeds: vmeshes[seed] = get_meshes(seed)

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
try: os.makedirs(savepath + '/logs/')
except: pass
logfile = datetime.now().strftime('logs/tflogfile_%H_%M_%d_%m_%Y.log')
fh = logging.FileHandler(savepath + logfile)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


#for max_steps in [50, 100, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]:
for max_steps in [50, 100, 500, 1000, 3000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000]:
    print('For max_steps = ', max_steps)
    tf.reset_default_graph()
    run_config = tf.estimator.RunConfig(save_checkpoints_steps = 2000)

    model =  MDNEstimator(nchannels=nchannels, n_y=ntargets, n_mixture=8, dropout=0.95,
                      model_dir=savepath + 'model', config = run_config)

    model.train(training_input_fn, max_steps=max_steps)
    f = open(savepath + 'model/checkpoint')
    lastpoint = int(f.readline().split('-')[-1][:-2])
    f.close()
    if lastpoint > max_steps:
        print('Don"t save')
        print(lastpoint)
    else:
        print("Have to save")
        save_module(model, savepath, max_steps)
