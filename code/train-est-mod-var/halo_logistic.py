import numpy as np
import matplotlib.pyplot as plt
#
import sys, os
sys.path.append('../utils/')
import tools
import datatools as dtools
from time import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 #
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import add_arg_scope
from layers import wide_resnet
import tensorflow_hub as hub
import tensorflow_probability
import tensorflow_probability as tfp
tfd = tensorflow_probability.distributions
tfd = tfp.distributions
tfb = tfp.bijectors

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
ncp = 128
step, stepf = 5, 40
path = '../../data/z00/'
ftype = 'L%04d_N%04d_S%04d_%02dstep/'
numd = 1e-3
num = int(numd*bs**3)
R1 = 3
R2 = 3*1.2
kny = np.pi*ncp/bs
kk = tools.fftk((ncp, ncp, ncp), bs)
seeds = [100, 200, 300, 400]
rprob = 0.5

#############################

suff = 'pad2_32'
fname = open('../models/n10/README', 'a+', 1)
fname.write('%s \t :\n\tModel to predict halo position likelihood in  trainestmodvarhalo.py with data supplemented by size=32 only; rotation with probability=0.5 and padding the mesh with 2 cells. Also reduce learning rate in piecewise constant manner. Changed teh n_y=1 and high of quntized distribution to 4\n'%suff)
fname.close()

savepath = '../models/n10/%s/'%suff
#if not os.path.exists(savepath):
#    os.makedirs(savepath)
try : os.makedirs(savepath)
except: pass

fname = open(savepath + 'log', 'w+', 1)
#fname = None
num_cubes= 500
#cube_sizes = np.array([8, 16, 32, 64, 128]).astype(int)
cube_sizes = np.array([32]).astype(int)
nsizes = len(cube_sizes)
pad = int(2)
cube_sizesft = (cube_sizes + 2*pad).astype(int)
max_offset = ncp - cube_sizes
ftname = ['cic']
tgname = ['pnn']
nchannels = len(ftname)
ntargets = len(tgname)
print('Features are ', ftname, file=fname)
print('Pad with ', pad, file=fname)
print('Rotation probability = %0.2f'%rprob, file=fname)

#############################
##Read data and generate meshes
#mesh = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'mesh/d/')



def generate_training_data():
    meshes = {}
    cube_features, cube_target = [[] for i in range(len(cube_sizes))], [[] for i in range(len(cube_sizes))]

    for seed in seeds:
        mesh = {}
        partp = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'dynamic/1/Position/')
        mesh['cic'] = tools.paintcic(partp, bs, ncp)
        #mesh['decic'] = tools.decic(mesh['cic'], kk, kny)
        mesh['R1'] = tools.fingauss(mesh['cic'], kk, R1, kny)
        #mesh['R2'] = tools.fingauss(mesh['cic'], kk, R2, kny)
        #mesh['GD'] = mesh['R1'] - mesh['R2']

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

        meshes[seed] = [mesh, hmesh]

        print('All the mesh have been generated for seed = %d'%seed)

        #Create training voxels
        ftlist = [mesh[i].copy() for i in ftname]
        ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
    #     targetmesh = hmesh['pnn']
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


meshes, cube_features, cube_target = generate_training_data()



##
##
##
###Save a snapshot of features
##fig, ax = plt.subplots(1, nchannels+1, figsize = (nchannels*4+4, 5))
##n = 10
##for i in range(nchannels):
##    ax[i].imshow(cube_features[n][:,:,:,i].sum(axis=0))
##    ax[i].set_title(ftname[i])
##ax[-1].imshow(cube_target[n][:,:,:,0].sum(axis=0))
##ax[-1].set_title('Target')
##plt.savefig('./figs/n%02d/features%s.png'%(numd*1e4, suff))
##
#############################
### Model
def _mdn_model_fn(features, labels, n_y, n_mixture, dropout, optimizer, mode):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        # Builds the neural network
        net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
        #net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.tanh)

        # Define the probabilistic layer 
        net = slim.conv3d(net, n_mixture*3*n_y, 1, activation_fn=None)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=3,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale)

        # Form mixture of discretized logistic distributions. Note we shift the
        # logistic distribution by -0.5. This lets the quantization capture "rounding"
        # intervals, `(x-0.5, x+0.5]`, and not "ceiling" intervals, `(x-1, x]`.
        discretized_logistic_dist = tfd.QuantizedDistribution(
            distribution=tfd.TransformedDistribution(
                distribution=tfd.Logistic(loc=loc, scale=scale),
                bijector=tfb.AffineScalar(shift=-0.5)),
            low=0.,
            high=2.**3-1)

        mixture_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=discretized_logistic_dist)

        # Define a function for sampling, and a function for estimating the log likelihood
        sample = tf.squeeze(mixture_dist.sample())
        loglik = mixture_dist.log_prob(obs_layer)
        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik})
    

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
            boundaries = [10000, 20000, 30000, 40000]
            values = [0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
                                        
        tf.summary.scalar('loss', neg_log_likelihood)
        tf.summary.scalar('rate', learning_rate)
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
                 n_y,
                 n_mixture,
                 optimizer=tf.train.AdamOptimizer,
                 dropout=None,
                 model_dir=None,
                 config=None):
        """Initializes a `MDNEstimator` instance.
        """

        def _model_fn(features, labels, mode):
            return _mdn_model_fn(features, labels, 
                 n_y, n_mixture, dropout,
                                 optimizer, mode)

        super(self.__class__, self).__init__(model_fn=_model_fn,
                                             model_dir=model_dir,
                                             config=config)



        
#############################################################################
###Train


batch_size=64
rprob = 0.5


def mapping_function(inds):
    def extract_batch(inds):
        
        isize = np.random.choice(len(cube_sizes), 1, replace=True)[0]
        batch = int(batch_size*8/cube_sizes[isize])
        if isize == cube_sizes[isize]==nc : batch = 1
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
#     sft = cube_features[isize].shape
#     stg = cube_target[isize].shape
#     ft.set_shape((None,)+sft[1:]) 
#     tg.set_shape((None,)+stg[1:])
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







def save_module(model, savepath, max_steps):

    print('Save module')

    features = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
    labels = tf.placeholder(tf.float32, shape=[None, None, None, None, ntargets], name='labels')
    exporter = hub.LatestModuleExporter("tf_hub", tf.estimator.export.build_raw_serving_input_receiver_fn({'features':features, 'labels':labels},
                                                                       default_batch_size=None))
    modpath = exporter.export(model, savepath + 'module', model.latest_checkpoint())
    modpath = modpath.decode("utf-8") 
    check_module(modpath)
    

#####
def check_module(modpath):
    print('Test module')
    tf.reset_default_graph()
    module = hub.Module(modpath + '/likelihood/')
    xx = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
    yy = tf.placeholder(tf.float32, shape=[None, None, None, None, ntargets], name='labels')
    samples = module(dict(features=xx, labels=yy), as_dict=True)['sample']
    loglik = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']

    preds = {}
    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())

        for seed in seeds:
            xxm = np.pad(meshes[seed][0]['cic'] , pad, 'wrap')
            #yym = np.stack([np.pad(meshes[seed][1]['pnncen'], pad, 'wrap'), np.pad(meshes[seed][1]['pnnsat'], pad, 'wrap')], axis=-1)
            yym = np.stack([meshes[seed][1][i] for i in tgname], axis=-1)
            print(xxm.shape, yym.shape)
            preds[seed] = sess.run(samples, feed_dict={xx:np.expand_dims(np.expand_dims(xxm,  -1), 0), yy:np.expand_dims(yym, 0)})
            meshes[seed][0]['predict'] = preds[seed][:, :, :]


    ##############################
    ##Power spectrum
    shape = [nc,nc,nc]
    kk = tools.fftk(shape, bs)
    kmesh = sum(i**2 for i in kk)**0.5

    fig, axar = plt.subplots(2, 2, figsize = (8, 8))
    ax = axar[0]
    for seed in seeds:
        for i, key in enumerate(['']):
            predict, hpmeshd = meshes[seed][0]['predict%s'%key] , meshes[seed][1]['pnn%s'%key], 
            k, pkpred = tools.power(predict/predict.mean(), boxsize=bs, k=kmesh)
            k, pkhd = tools.power(hpmeshd/hpmeshd.mean(), boxsize=bs, k=kmesh)
            k, pkhx = tools.power(hpmeshd/hpmeshd.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)    
        ##
            ax[0].semilogx(k[1:], pkpred[1:]/pkhd[1:], label=seed)
            ax[1].semilogx(k[1:], pkhx[1:]/(pkpred[1:]*pkhd[1:])**0.5)
            ax[0].set_title(key, fontsize=12)

    for axis in ax.flatten():
        axis.legend(fontsize=14)
        axis.set_yticks(np.arange(0, 1.1, 0.1))
        axis.grid(which='both')
        axis.set_ylim(0.,1.1)
    ax[0].set_ylabel('Transfer function', fontsize=14)
    ax[1].set_ylabel('Cross correlation', fontsize=14)
    #
    ax = axar[1]
    for i, key in enumerate([ '']):
        predict, hpmeshd = meshes[seed][0]['predict%s'%key] , meshes[seed][1]['pnn%s'%key], 
        vmin, vmax = 0, (hpmeshd[:, :, :].sum(axis=0)).max()
        im = ax[0].imshow(predict[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        im = ax[1].imshow(hpmeshd[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        ax[0].set_title(key, fontsize=15)
    ax[0].set_title('Prediction', fontsize=15)
    ax[1].set_title('Truth', fontsize=15)
    plt.savefig(savepath + '/predict%d.png'%max_steps)
    plt.show()

#


############################################################################
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


for max_steps in np.array([5e3, 1e4, 2e4, 3e4]).astype(int):
    print('For max_steps = ', max_steps)
    tf.reset_default_graph()
    run_config = tf.estimator.RunConfig(save_checkpoints_steps = 2000)

    model =  MDNEstimator(n_y=ntargets, n_mixture=8, dropout=0.95,
                      model_dir=savepath + 'model', config = run_config)

    model.train(training_input_fn, max_steps=max_steps)
    save_module(model, savepath, max_steps)
