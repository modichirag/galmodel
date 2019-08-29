###list of models:
##def _mdn_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad):
##def _mdn_mask_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad):
##def _mdn_vireg_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, fudge=1.0
##def _mdn_vireg_model_fn_simple(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, fu
####def _mdn_mass_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad):
##def _mdn_mass_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad):
##def _mdn_smooth_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad):
##def _mdn_inv_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, fudge=1.0, 
##def _mdn_specres_gm_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode):
##def _mdn_specres_poisson_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode):
##

import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import add_arg_scope
from layers import wide_resnet, wide_resnet_snorm, valid_resnet
from layers import  wide_resnet_snorm
from tfops import specnormconv3d, dynamic_deconv3d
import tensorflow_hub as hub
import tensorflow_probability
import tensorflow_probability as tfp
tfd = tensorflow_probability.distributions
tfd = tfp.distributions
tfb = tfp.bijectors


### Model
def _mdn_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, distribution='logistic'):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        # Builds the neural network
        if pad == 0:
            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
        elif pad == 2:
            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
        if pad == 4:
            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
            net = slim.conv3d(net, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
        #net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        if distribution == 'logistic': net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.tanh)
        else: net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.leaky_relu)

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
        if distribution == 'logistic':
            discretized_logistic_dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(loc=loc, scale=scale),
                    bijector=tfb.AffineScalar(shift=-0.5)),
                low=0.,
                high=2.**3-1)

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=discretized_logistic_dist)

        elif distribution == 'normal':

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=tfd.Normal(loc=loc, scale=scale))
            

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
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
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









# Model
def _mdn_specres_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, normwt=0.7, distribution='logistic'):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        nfilter = 16
        nfilter0 = 1
        num_iters = 1
        # Builds the neural network

        net = feature_layer
        subnet = specnormconv3d(net, nfilter, 3, name='l11', num_iters=num_iters, normwt=normwt)
        subnet = tf.nn.leaky_relu(subnet)
        net = net + specnormconv3d(subnet, nfilter0, 3, name='l12', num_iters=num_iters, normwt=normwt)
    #     net = tf.nn.dropout(net, 0.95)
        net = tf.nn.leaky_relu(net)

        subnet = specnormconv3d(net, nfilter, 3, name='l21', num_iters=num_iters, normwt=normwt)
        subnet = tf.nn.leaky_relu(subnet)
        net = net + specnormconv3d(subnet,nfilter0, 3, name='l22', num_iters=num_iters, normwt=normwt)
    #     net = tf.nn.dropout(net, 0.95)
        net = tf.nn.leaky_relu(net)
        
        subnet = specnormconv3d(net, nfilter, 3, name='l31', num_iters=num_iters, normwt=normwt)
        subnet = tf.nn.leaky_relu(subnet)
        net = net + specnormconv3d(subnet,nfilter0, 3, name='l32', num_iters=num_iters, normwt=normwt)
    #     net = tf.nn.dropout(net, 0.95)
        net = tf.nn.leaky_relu(net)
        
        subnet = specnormconv3d(net, nfilter, 3, name='l41', num_iters=num_iters, normwt=normwt)
        subnet = tf.nn.leaky_relu(subnet)
        net = net + specnormconv3d(subnet,nfilter0, 3, name='l42', num_iters=num_iters, normwt=normwt)
        net = tf.nn.leaky_relu(net)
        # Define the probabilistic layer 
        net = specnormconv3d(net, n_mixture*3*n_y, 1)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])
        loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=3,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale)+1e-3

        # Form mixture of discretized logistic distributions. Note we shift the
        # logistic distribution by -0.5. This lets the quantization capture "rounding"
        # intervals, `(x-0.5, x+0.5]`, and not "ceiling" intervals, `(x-1, x]`.
        if distribution == 'logistic':
            discretized_logistic_dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(loc=loc, scale=scale),
                    bijector=tfb.AffineScalar(shift=-0.5)),
                low=0.,
                high=2.**3-1)

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=discretized_logistic_dist)

        elif distribution == 'normal':

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=tfd.Normal(loc=loc, scale=scale))

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
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [1e-4, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
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





##
##def _mdn_mask_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, lr0=1e-3, dsitribution='logistic'):
##
##    # Check for training mode
##    is_training = mode == tf.estimator.ModeKeys.TRAIN
##        
##    def _module_fn():
##        """
##        Function building the module
##        """
##    
##        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
##        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
##        mask_layer = tf.clip_by_value(obs_layer, 0, 1)
##        #       
##        # Builds the neural network
##        if pad == 0:
##            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
##        elif pad == 2:
##            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
##        #net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
##        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
##        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
##        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
##        if distribution == 'logistic': net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.tanh)
##        else: net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.leaky_relu)
##
##        #Predicted mask
##        masknet = slim.conv3d(net, 8, 1, activation_fn=tf.nn.leaky_relu)
##        out_mask = slim.conv3d(masknet, 1, 1, activation_fn=None)
##        pred_mask = tf.nn.sigmoid(out_mask)
##
##        # Define the probabilistic layer 
##        likenet = slim.conv3d(net, 64, 1, activation_fn=tf.nn.leaky_relu)
##        net = slim.conv3d(likenet, n_mixture*3*n_y, 1, activation_fn=None)
##        cube_size = tf.shape(obs_layer)[1]
##        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])
###         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
##        loc, unconstrained_scale, logits = tf.split(net,
##                                                    num_or_size_splits=3,
##                                                    axis=-1)
##        scale = tf.nn.softplus(unconstrained_scale)
##
##        # Form mixture of discretized logistic distributions. Note we shift the
##        # logistic distribution by -0.5. This lets the quantization capture "rounding"
##        # intervals, `(x-0.5, x+0.5]`, and not "ceiling" intervals, `(x-1, x]`.
##        if distribution == 'logistic':
##            discretized_logistic_dist = tfd.QuantizedDistribution(
##                distribution=tfd.TransformedDistribution(
##                    distribution=tfd.Logistic(loc=loc, scale=scale),
##                    bijector=tfb.AffineScalar(shift=-0.5)),
##                low=0.,
##                high=2.**3-1)
##
##            mixture_dist = tfd.MixtureSameFamily(
##                mixture_distribution=tfd.Categorical(logits=logits),
##                components_distribution=discretized_logistic_dist)
##
##        elif distribution == 'normal':
##
##            mixture_dist = tfd.MixtureSameFamily(
##                mixture_distribution=tfd.Categorical(logits=logits),
##                components_distribution=tfd.Normal(loc=loc, scale=scale))
##
##        # Define a function for sampling, and a function for estimating the log likelihood
##        #sample = tf.squeeze(mixture_dist.sample())
##        rawsample = mixture_dist.sample()
##        sample = rawsample*pred_mask
##        loglik = mixture_dist.log_prob(obs_layer)
##
##        loss1 = - loglik* mask_layer 
##        loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,
##                                                labels=mask_layer) * 0.1
##        loss = loss1 + loss2
##
##        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
##                          outputs={'sample':sample, 'loglikelihood':loglik,
##                                   'loc':loc, 'scale':scale, 'logits':logits,
##                                   'rawsample':rawsample, 'pred_mask':pred_mask, 'out_mask':out_mask,
##                                   'loss':loss, 'loss1':loss1, 'loss2':loss2})
##
##
##    # Create model and register module if necessary
##    spec = hub.create_module_spec(_module_fn)
##    module = hub.Module(spec, trainable=True)
##    if isinstance(features,dict):
##        predictions = module(features, as_dict=True)
##    else:
##        predictions = module({'features':features, 'labels':labels}, as_dict=True)
##    
##    if mode == tf.estimator.ModeKeys.PREDICT:    
##        hub.register_module_for_export(module, "likelihood")
##        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
##    
##    loss = predictions['loss']
##    loss1 = tf.reduce_mean(predictions['loss1'])    
##    loss2 = tf.reduce_mean(predictions['loss2'])    
##    # Compute and register loss function
##    loss = tf.reduce_mean(loss)    
##    tf.losses.add_loss(loss)
##
##    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
##
##    train_op = None
##    eval_metric_ops = None
##
##    # Define optimizer
##    if mode == tf.estimator.ModeKeys.TRAIN:
##        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
##        with tf.control_dependencies(update_ops):
##            global_step=tf.train.get_global_step()
##            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
##            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
##            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
##            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
##            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
##            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "loss" : loss, 
##                "loss1" : loss1, "loss2" : loss2 }, every_n_iter=50)
##
##            tf.summary.scalar('rate', learning_rate)                            
##        tf.summary.scalar('loss', loss)
##    elif mode == tf.estimator.ModeKeys.EVAL:
##        
##        eval_metric_ops = { "log_p": neg_log_likelihood}
##
##    return tf.estimator.EstimatorSpec(mode=mode,
##                                      predictions=predictions,
##                                      loss=total_loss,
##                                      train_op=train_op,
##                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])
##




def _mdn_mask_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, lr0=1e-3, distribution='logistic', masktype='constant'):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 0.001)*1000
        #       
        # Builds the neural network
        if pad == 0:
            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
        elif pad == 2:
            net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
        #net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        if distribution == 'logistic': net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.tanh)
        else: net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.leaky_relu)

        #Predicted mask
        masknet = slim.conv3d(net, 8, 1, activation_fn=tf.nn.leaky_relu)
        out_mask = slim.conv3d(masknet, 1, 1, activation_fn=None)
        pred_mask = tf.nn.sigmoid(out_mask)

        # Define the probabilistic layer 
        likenet = slim.conv3d(net, 64, 1, activation_fn=tf.nn.leaky_relu)
        net = slim.conv3d(likenet, n_mixture*3*n_y, 1, activation_fn=None)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=3,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale) + 1e-3

        # Form mixture of discretized logistic distributions. Note we shift the
        # logistic distribution by -0.5. This lets the quantization capture "rounding"
        # intervals, `(x-0.5, x+0.5]`, and not "ceiling" intervals, `(x-1, x]`.
        if distribution == 'logistic':
            discretized_logistic_dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(loc=loc, scale=scale),
                    bijector=tfb.AffineScalar(shift=-0.5)),
                low=0.,
                high=2.**3-1)

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=discretized_logistic_dist)

        elif distribution == 'normal':

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=tfd.Normal(loc=loc, scale=scale))

        # Define a function for sampling, and a function for estimating the log likelihood
        #sample = tf.squeeze(mixture_dist.sample())
        rawsample = mixture_dist.sample()
        sample = rawsample*pred_mask
        rawloglik = mixture_dist.log_prob(obs_layer)
        print(rawloglik)
        print(out_mask)
        print(mask_layer)
        
        #loss1 = - rawloglik* out_mask #This can be constant mask as well if we use mask_layer instead
        if masktype == 'constant': loss1 = - rawloglik* mask_layer 
        elif masktype == 'vary': loss1 = - rawloglik* pred_mask 
        loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,
                                                labels=mask_layer) 
        loglik = - (loss1 + loss2)

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'loc':loc, 'scale':scale, 'logits':logits,
                                   'rawsample':rawsample, 'pred_mask':pred_mask, 'out_mask':out_mask,
                                   'rawloglik':rawloglik, 'loss1':loss1, 'loss2':loss2})


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
    
    rawloglik = tf.reduce_mean(predictions['rawloglik'])    
    loss1 = tf.reduce_mean(predictions['loss1'])    
    loss2 = tf.reduce_mean(predictions['loss2'])    

    # Compute and register loss function
    loglik = predictions['loglikelihood']
    # Compute and register loss function
    neg_log_likelihood = -tf.reduce_sum(loglik, axis=-1)
    neg_log_likelihood = tf.reduce_mean(neg_log_likelihood)
    
    tf.losses.add_loss(neg_log_likelihood)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    #loss = tf.reduce_mean(loss)    
    #tf.losses.add_loss(loss)
    #total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "negloglik" : neg_log_likelihood, 
                                                       "rawloglik" : rawloglik, "loss1" : loss1, "loss2" : loss2 }, every_n_iter=50)

            tf.summary.scalar('rate', learning_rate)                            
        tf.summary.scalar('loglik', neg_log_likelihood)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])





def _mdn_vireg_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, fudge=1.0):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        # Builds the neural network
        net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
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
        #sample = tf.squeeze(mixture_dist.sample())
        sample = mixture_dist.sample()
        
        loglik = mixture_dist.log_prob(obs_layer)
        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'loc':loc, 'scale':scale, 'logits':logits})



    def _inference_module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        # Builds the neural network
        net = slim.conv3d(obs_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
#         net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.tanh)

        # Define the probabilistic layer 
        net = slim.conv3d(net, 2*nchannels, 1, activation_fn=None)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, nchannels, 2])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale = tf.split(net, num_or_size_splits=2,
                                                    axis=-1)
        print('\nloc :\n', loc)
        scale = tf.nn.softplus(unconstrained_scale[...,0])
        
        distribution = tfd.MultivariateNormalDiag(loc=loc[...,0], scale_diag=scale)
        
        # Define a function for sampling, and a function for estimating the log likelihood
        sample = tf.squeeze(distribution.sample())
        print('\ninf dist sample :\n', distribution.sample())
        logfeature = tf.log1p(feature_layer)
        print('\nlogfeature :\n', logfeature)
        loglik = distribution.log_prob(logfeature[...])
        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik, 'sigma':scale, 'mean':loc})
    
    
    # Create model and register module if necessary
    spec = hub.create_module_spec(_module_fn)
    module = hub.Module(spec, trainable=True)
    spec_inf = hub.create_module_spec(_inference_module_fn)
    module_inf = hub.Module(spec_inf, trainable=True)
    
    if isinstance(features,dict):
        predictions = module(features, as_dict=True)
        features_ = features['features']
    else:
        predictions = module({'features':features, 'labels':labels}, as_dict=True)
        features_ = features
        
    samples = predictions['sample']
    print('\nsamples :\n', samples)

    inference = module_inf({'features':features_, 'labels':samples}, as_dict=True)
    
    if mode == tf.estimator.ModeKeys.PREDICT:    
        hub.register_module_for_export(module, "likelihood")
        hub.register_module_for_export(module_inf, "inference")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loglik = predictions['loglikelihood']
    reg_loglik = inference['loglikelihood']
    print('\nloglik :\n', loglik)
    print('\nreg_loglik :\n', reg_loglik)
    ####Compute and register loss function
    neg_log_likelihood = -tf.reduce_sum(loglik, axis=-1)
    neg_log_likelihood = tf.reduce_mean(neg_log_likelihood) 
    reg_log_likelihood = -tf.reduce_sum(reg_loglik, axis=-1)
    reg_log_likelihood = fudge* tf.reduce_mean(reg_log_likelihood)
    
    tf.losses.add_loss(neg_log_likelihood)
    tf.losses.add_loss(reg_log_likelihood)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    
    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4]).astype(int))
            values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)                            
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('loss', neg_log_likelihood)
        tf.summary.scalar('reg_loss', reg_log_likelihood)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)









def _mdn_vireg_model_fn_simple(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, fudge=1.0):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        # Builds the neural network
        net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
        #net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
        net = wide_resnet(net, 8, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = slim.conv3d(net, 8, 3, activation_fn=tf.nn.tanh)

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
        #sample = tf.squeeze(mixture_dist.sample())
        sample = mixture_dist.sample()
        
        loglik = mixture_dist.log_prob(obs_layer)
        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'loc':loc, 'scale':scale, 'logits':logits})



    def _inference_module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        # Builds the neural network
        net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
        #net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
        net = wide_resnet(net, 8, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = slim.conv3d(net, 8, 3, activation_fn=tf.nn.tanh)

        # Define the probabilistic layer 
        net = slim.conv3d(net, 2*nchannels, 1, activation_fn=None)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, nchannels, 2])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale = tf.split(net, num_or_size_splits=2,
                                                    axis=-1)
        print('\nloc :\n', loc)
        scale = tf.nn.softplus(unconstrained_scale[...,0])
        
        distribution = tfd.MultivariateNormalDiag(loc=loc[...,0], scale_diag=scale)
        
        # Define a function for sampling, and a function for estimating the log likelihood
        sample = tf.squeeze(distribution.sample())
        print('\ninf dist sample :\n', distribution.sample())
        logfeature = tf.log1p(feature_layer)
        print('\nlogfeature :\n', logfeature)
        loglik = distribution.log_prob(logfeature[...])
        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik, 'sigma':scale, 'mean':loc})
    
    
    # Create model and register module if necessary
    spec = hub.create_module_spec(_module_fn)
    module = hub.Module(spec, trainable=True)
    spec_inf = hub.create_module_spec(_inference_module_fn)
    module_inf = hub.Module(spec_inf, trainable=True)
    
    if isinstance(features,dict):
        predictions = module(features, as_dict=True)
        features_ = features['features']
    else:
        predictions = module({'features':features, 'labels':labels}, as_dict=True)
        features_ = features
        
    samples = predictions['sample']
    print('\nsamples :\n', samples)

    inference = module_inf({'features':features_, 'labels':samples}, as_dict=True)
    
    if mode == tf.estimator.ModeKeys.PREDICT:    
        hub.register_module_for_export(module, "likelihood")
        hub.register_module_for_export(module_inf, "inference")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loglik = predictions['loglikelihood']
    reg_loglik = inference['loglikelihood']
    print('\nloglik :\n', loglik)
    print('\nreg_loglik :\n', reg_loglik)
    ####Compute and register loss function
    neg_log_likelihood = -tf.reduce_sum(loglik, axis=-1)
    neg_log_likelihood = tf.reduce_mean(neg_log_likelihood) 
    reg_log_likelihood = -tf.reduce_sum(reg_loglik, axis=-1)
    reg_log_likelihood = fudge* tf.reduce_mean(reg_log_likelihood)
    
    tf.losses.add_loss(neg_log_likelihood)
    tf.losses.add_loss(reg_log_likelihood)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None



    
    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4]).astype(int))
            values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)                            
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('loss', neg_log_likelihood)
        tf.summary.scalar('reg_loss', reg_log_likelihood)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)











def _mdn_inv_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, logoffset=1e-3, log=True):
    '''Train inverse model i.e. go from the halo field to matter overdensity
    '''
    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        # Builds the neural network
        net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
        #net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = slim.conv3d(net, 32, 3, activation_fn=tf.nn.leaky_relu)


        # Define the probabilistic layer 
        net = slim.conv3d(net, 3*n_mixture*nchannels, 1, activation_fn=None)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, nchannels, n_mixture*3])

        logits, loc, unconstrained_scale = tf.split(net, num_or_size_splits=3,
                                                    axis=-1)
        print('\nloc :\n', loc)
        scale = tf.nn.softplus(unconstrained_scale[...]) + 1e-3
        
        distribution = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits[...]),
            #components_distribution=tfd.MultivariateNormalDiag(loc=loc[...,0], scale_diag=scale))
            components_distribution=tfd.Normal(loc=loc[...], scale=scale))
        print('\ngmm\n', distribution)

        
        # Define a function for sampling, and a function for estimating the log likelihood
        if log :
            print('Logged it')
            sample = tf.exp(distribution.sample()) - logoffset
            print('\ninf dist sample :\n', distribution.sample())
            logfeature = tf.log(tf.add(logoffset, obs_layer), 'logfeature')
            print('\nlogfeature :\n', logfeature)
            prob = distribution.prob(logfeature[...])
            loglik = distribution.log_prob(logfeature[...])
        else:
            print('UnLogged it')
            sample = distribution.sample()
            print('\ninf dist sample :\n', distribution.sample())
            loglik = distribution.log_prob(obs_layer[...])

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik, 'sigma':scale, 'mean':loc, 'logits':logits})
    
    
    # Create model and register module if necessary
    spec = hub.create_module_spec(_module_fn)
    module = hub.Module(spec, trainable=True)
    
    if isinstance(features,dict):
        predictions = module(features, as_dict=True)
        features_ = features['features']
    else:
        predictions = module({'features':features, 'labels':labels}, as_dict=True)
        features_ = features
        
    samples = predictions['sample']
    print('\nsamples :\n', samples)

    
    if mode == tf.estimator.ModeKeys.PREDICT:    
        hub.register_module_for_export(module, "likelihood")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loglik = predictions['loglikelihood']
    print('\nloglik :\n', loglik)
    ####Compute and register loss function
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
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4]).astype(int))
            values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)                            
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('loss', neg_log_likelihood)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)




def _mdn_invreg_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, logoffset=1e-3, log=True):
    '''Train inverse model i.e. go from the halo field to matter overdensity but with regression
    '''
    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        # Builds the neural network
        net = slim.conv3d(feature_layer, 16, 5, activation_fn=tf.nn.leaky_relu, padding='same')
        #net = wide_resnet(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
        net = wide_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = wide_resnet(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = slim.conv3d(net, 1, 3, activation_fn=tf.nn.leaky_relu)

        
        # Define a function for sampling, and a function for estimating the log likelihood
        if log :
            fit = tf.log(tf.add(logoffset, obs_layer), 'logfeature')
        else:
            fit = obs_layer

        sample = net
        loglik = -1* (net-fit)*(net-fit)
        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik})
    
    
    # Create model and register module if necessary
    spec = hub.create_module_spec(_module_fn)
    module = hub.Module(spec, trainable=True)
    
    if isinstance(features,dict):
        predictions = module(features, as_dict=True)
        features_ = features['features']
    else:
        predictions = module({'features':features, 'labels':labels}, as_dict=True)
        features_ = features
        
    samples = predictions['sample']
    print('\nsamples :\n', samples)

    
    if mode == tf.estimator.ModeKeys.PREDICT:    
        hub.register_module_for_export(module, "likelihood")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loglik = predictions['loglikelihood']
    print('\nloglik :\n', loglik)
    ####Compute and register loss function
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
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4]).astype(int))
            values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)                            
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('loss', neg_log_likelihood)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)










def _mdn_specres_poisson_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, normwt=0.7, nfilter0=1, nfilter=16):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        #nfilter0 = 4
        num_iters = 1
        
        if nfilter0 == 1: net = feature_layer
        else: net = specnormconv3d(feature_layer, nfilter0, 3, name='l00', num_iters=num_iters, normwt=normwt)

        # Builds the neural network
        subnet = specnormconv3d(net, nfilter, 3, name='l11', num_iters=num_iters, normwt=normwt)
        subnet = tf.nn.leaky_relu(subnet)
        net = net + specnormconv3d(subnet, nfilter0, 3, name='l12', num_iters=num_iters, normwt=normwt)
    #     net = tf.nn.dropout(net, 0.95)
        net = tf.nn.leaky_relu(net)

        subnet = specnormconv3d(net, nfilter, 3, name='l21', num_iters=num_iters, normwt=normwt)
        subnet = tf.nn.leaky_relu(subnet)
        net = net + specnormconv3d(subnet,nfilter0, 3, name='l22', num_iters=num_iters, normwt=normwt)
    #     net = tf.nn.dropout(net, 0.95)
        net = tf.nn.leaky_relu(net)
        
        subnet = specnormconv3d(net, nfilter, 3, name='l31', num_iters=num_iters, normwt=normwt)
        subnet = tf.nn.leaky_relu(subnet)
        net = net + specnormconv3d(subnet,nfilter0, 3, name='l32', num_iters=num_iters, normwt=normwt)
    #     net = tf.nn.dropout(net, 0.95)
        net = tf.nn.leaky_relu(net)
        
        subnet = specnormconv3d(net, nfilter, 3, name='l41', num_iters=num_iters, normwt=normwt)
        subnet = tf.nn.leaky_relu(subnet)
        net = net + specnormconv3d(subnet,nfilter0, 3, name='l42', num_iters=num_iters, normwt=normwt)
        net = tf.nn.leaky_relu(net)

        subnet = specnormconv3d(net, nfilter, 3, name='l51', num_iters=num_iters, normwt=normwt)
        subnet = tf.nn.leaky_relu(subnet)
        net = net + specnormconv3d(subnet,nfilter0, 3, name='l52', num_iters=num_iters, normwt=normwt)
        #net = tf.nn.leaky_relu(net)

##
##        net = wide_resnet_snorm(feature_layer, 8, activation_fn=tf.nn.leaky_relu, is_training=is_training)
##        net = wide_resnet_snorm(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
##        net = wide_resnet_snorm(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
###         net = wide_resnet_snorm(net, 32, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
##        net = wide_resnet_snorm(net, 16, activation_fn=tf.nn.leaky_relu)
##        net = wide_resnet_snorm(net, 1, activation_fn=None)
##        
        # Define the probabilistic layer
        if nfilter0 != 0: net = specnormconv3d(net, 1, 1, name='lfin', num_iters=num_iters, normwt=normwt)
        lbda = tf.nn.softplus(net, name='lambda') + 1e-5

        dist = tfd.Poisson(lbda)
        
        sample = tf.squeeze(dist.sample())
        loglik = dist.log_prob(obs_layer)
        difference = (tf.subtract(obs_layer, net))
    
        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik, 'lambda':lbda, 
                                   'difference':difference})
    
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
    
    # Compute and register loss function
    loglik = -predictions['loglikelihood']
#     diff = predictions['difference']
#     loglik = tf.square(diff)
    
    neg_log_likelihood = tf.reduce_sum(loglik, axis=-1)
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
            boundaries = [15000, 30000, 45000, 60000]
            values = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
                                        
        tf.summary.scalar('loss', neg_log_likelihood)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)









def _mdn_nozero_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, lr0=1e-3, dsitribution='logistic'):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 1)
        #       

        # Builds the neural network
        net = slim.conv3d(feature_layer, 16, 3, activation_fn=tf.nn.leaky_relu, padding='valid')
        subnet = tf.identity(net[:, 3:-3, 3:-3, 3:-3, :])
        net = valid_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = slim.conv3d(net, 16, 3, activation_fn=tf.nn.leaky_relu, padding='valid')
        net = net+subnet
        
        # Define the probabilistic layer 
        likenet = slim.conv3d(net, 64, 1, activation_fn=tf.nn.leaky_relu)
        net = slim.conv3d(likenet, n_mixture*3*n_y, 1, activation_fn=None)

        # Define the probabilistic layer 
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=3,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale) + 1e-3

        # Form mixture of discretized logistic distributions. Note we shift the
        # logistic distribution by -0.5. This lets the quantization capture "rounding"
        # intervals, `(x-0.5, x+0.5]`, and not "ceiling" intervals, `(x-1, x]`.
        if distribution == 'logistic':
            discretized_logistic_dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(loc=loc, scale=scale),
                    bijector=tfb.AffineScalar(shift=-0.5)),
                low=0.,
                high=2.**3-1)

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=discretized_logistic_dist)

        elif distribution == 'normal':

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=tfd.Normal(loc=loc, scale=scale))
            

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
    
    loss = -predictions['loglikelihood']
    # Compute and register loss function
    loss = tf.reduce_mean(loss)    
    tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)
            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "loss" : loss}, every_n_iter=50)

        tf.summary.scalar('loss', loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])





def _mdn_nozero_mask_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, lr0=1e-3):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 1)
        #       

        # Builds the neural network
        net = slim.conv3d(feature_layer, 16, 3, activation_fn=tf.nn.leaky_relu, padding='valid')
        subnet = tf.identity(net[:, 3:-3, 3:-3, 3:-3, :])
        net = valid_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = slim.conv3d(net, 16, 3, activation_fn=tf.nn.leaky_relu, padding='valid')
        net = net+subnet
        
        #Predicted mask
        masknet = slim.conv3d(net, 8, 1, activation_fn=tf.nn.leaky_relu)
        out_mask = slim.conv3d(masknet, 1, 1, activation_fn=None)
        pred_mask = tf.nn.sigmoid(out_mask)

        # Define the probabilistic layer 
        likenet = slim.conv3d(net, 64, 1, activation_fn=tf.nn.leaky_relu)
        net = slim.conv3d(likenet, n_mixture*3*n_y, 1, activation_fn=None)

        # Define the probabilistic layer 
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=3,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale) + 1e-3

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
        #sample = tf.squeeze(mixture_dist.sample())
        rawsample = mixture_dist.sample()
        sample = rawsample*pred_mask
        loglik = mixture_dist.log_prob(obs_layer)

        loss1 = - loglik* mask_layer 
        loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,
                                                labels=mask_layer)
        loss = loss1 + loss2

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'loc':loc, 'scale':scale, 'logits':logits,
                                   'rawsample':rawsample, 'pred_mask':pred_mask, 'out_mask':out_mask,
                                   'loss':loss, 'loss1':loss1, 'loss2':loss2})


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
    
    loss = predictions['loss']
    loss1, loss2 = tf.reduce_mean(predictions['loss1']), tf.reduce_mean(predictions['loss2'])
    # Compute and register loss function
    loss = tf.reduce_mean(loss)    
    tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)
            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "loss" : loss, 
                "loss1" : loss1, "loss2" : loss2 }, every_n_iter=50)

        tf.summary.scalar('loss', loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])






def _mdn_nozero_mask2_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, lr0=1e-3):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 1)
        #       

        # Builds the neural network
        net = slim.conv3d(feature_layer, 16, 3, activation_fn=tf.nn.leaky_relu, padding='valid')
        subnet = tf.identity(net[:, 3:-3, 3:-3, 3:-3, :])
        net = valid_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = slim.conv3d(net, 16, 3, activation_fn=tf.nn.leaky_relu, padding='valid')
        net = net+subnet
        
        #Predicted mask
        masknet = slim.conv3d(net, 8, 1, activation_fn=tf.nn.leaky_relu)
        out_mask = slim.conv3d(masknet, 1, 1, activation_fn=None)
        pred_mask = tf.nn.sigmoid(out_mask)

        # Define the probabilistic layer 
        likenet = slim.conv3d(net, 64, 1, activation_fn=tf.nn.leaky_relu)
        net = slim.conv3d(likenet, n_mixture*3*n_y, 1, activation_fn=None)

        # Define the probabilistic layer 
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=3,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale) + 1e-3

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
        #sample = tf.squeeze(mixture_dist.sample())
        rawsample = mixture_dist.sample()
        sample = rawsample*pred_mask
        loglik = mixture_dist.log_prob(obs_layer)

        loss1 = - loglik* pred_mask 
        loss2 = - loglik* mask_layer 
        loss3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,
                                                labels=mask_layer) 
        loss = loss1 + loss2 + loss3

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'loc':loc, 'scale':scale, 'logits':logits,
                                   'rawsample':rawsample, 'pred_mask':pred_mask, 'out_mask':out_mask,
                                   'loss':loss, 'loss1':loss1, 'loss2':loss2, 'loss3':loss3})


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
    
    loss = predictions['loss']
    loss1, loss2, loss3 = tf.reduce_mean(predictions['loss1']), tf.reduce_mean(predictions['loss2']), \
                          tf.reduce_mean(predictions['loss3'])
    # Compute and register loss function
    loss = tf.reduce_mean(loss)    
    tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)
            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "loss" : loss, 
                "loss1" : loss1, "loss2" : loss2, "loss3" : loss3 }, every_n_iter=50)

        tf.summary.scalar('loss', loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])












# Model
def _mdn_nozero_mass_reg_model_fn(features, labels, nchannels, n_y, dropout, optimizer, mode, pad):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 1)

        # Builds the neural network
        net = slim.conv3d(feature_layer, 16, 3, activation_fn=tf.nn.leaky_relu, padding='valid')
        subnet = tf.identity(net[:, 3:-3, 3:-3, 3:-3, :])
        net = valid_resnet(net, 16, activation_fn=tf.nn.leaky_relu, keep_prob=dropout, is_training=is_training)
        net = slim.conv3d(net, 16, 3, activation_fn=tf.nn.leaky_relu, padding='valid')
        net = net+subnet

        #Predicted mask
#         out_mask = slim.conv3d(net, 1, 1, activation_fn=tf.nn.tanh)
        out_mask = slim.conv3d(net, 1, 1, activation_fn=None)
        pred_mask = tf.nn.sigmoid(out_mask)
        
        net = slim.conv3d(net, 8, 1, activation_fn=tf.nn.leaky_relu)
        net = slim.conv3d(net, n_y, 1, activation_fn=None)

        rawsample = net
        diff = (net*pred_mask - obs_layer)
        sample = net*pred_mask
        loss = tf.square(diff)
        loglik = -loss

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer},
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'diff':diff,
                                   'rawsample':rawsample, 'pred_mask':pred_mask, 'out_mask':out_mask,
                                   'loss':loss})



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
    

    loss = predictions['loss']
    # Compute and register loss function
    loss = tf.reduce_mean(loss)    
    tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)                            
        tf.summary.scalar('loss', loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)







def _mdn_specres_nozero_mask_poisson_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad=4, lr0=1e-3, nfilter=32, nfilter0 = 2, normwt=0.7, num_iters=1):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    

        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 1)
        batch_size = tf.shape(obs_layer)[0]
        cube_size = tf.shape(obs_layer)[1]
        tileft = tf.tile(feature_layer, [1, 1, 1, 1, 2])
        #       

        # Builds the neural network       
        if nfilter0 == 2: net = tileft
        #if nfilter0 == 2: net = specnormconv3d(feature_layer, nfilter0, 3, name='l00', num_iters=num_iters, normwt=normwt)
        else: net = specnormconv3d(feature_layer, nfilter0, 3, name='l00', num_iters=num_iters, normwt=normwt)

        # Builds the neural network
        subnet = specnormconv3d(net, nfilter, 3, name='l11', num_iters=num_iters, normwt=normwt, padding='VALID')
        subnet = tf.nn.leaky_relu(subnet)
        subnet = specnormconv3d(subnet, nfilter0, 3, name='l12', num_iters=num_iters, normwt=normwt, padding='VALID')
        current_size = tf.shape(subnet)[1]
        short = tf.slice(net, [0, 2, 2, 2, 0], [batch_size, current_size, current_size, current_size, nfilter0])
        net = short + subnet
        net = tf.nn.leaky_relu(net)

        if pad == 4:
            subnet = specnormconv3d(net, nfilter, 3, name='l21', num_iters=num_iters, normwt=normwt, padding='VALID')
            subnet = tf.nn.leaky_relu(subnet)
            subnet = specnormconv3d(subnet, nfilter0, 3, name='l22', num_iters=num_iters, normwt=normwt, padding='VALID')
            current_size = tf.shape(subnet)[1]
            short = tf.slice(net, [0, 2, 2, 2, 0], [batch_size, current_size, current_size, current_size, nfilter0])
            net = short + subnet
            net = tf.nn.leaky_relu(net)
            
        
        # Define the probabilistic layer
        #if nfilter0 != 0: net = specnormconv3d(net, 1, 1, name='lfin', num_iters=num_iters, normwt=normwt)

        print(net)
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, nfilter0])
        print('net : ', net)
        prednet, masknet = tf.split(net, num_or_size_splits=2, axis=-1)
        print('prednet : ', prednet)
        print('masknet : ', masknet)
        
        #Predicted mask
        #out_mask = specnormconv3d(masknet[...,0], 1, 1, name='mask', normwt=normwt)
        #pred_mask = tf.squeeze(tf.nn.sigmoid(out_mask))
        out_mask = tf.identity(masknet[...,0], name='mask')
        pred_mask = tf.squeeze(tf.nn.sigmoid(out_mask))

        #Distribution        
        #lbda = specnormconv3d(prednet[...,0], 1, 1, name='pred', normwt=normwt)
        #lbda = tf.nn.softplus(lbda, name='lambda') + 1e-5
        lbda = tf.nn.softplus(prednet[...,0], name='lambda') + 1e-5
        dist = tfd.Poisson(lbda)
        
        rawsample = tf.squeeze(dist.sample())
        sample = rawsample*pred_mask
        loglik = dist.log_prob(obs_layer)
       
        # Define a function for sampling, and a function for estimating the log likelihood

        loss1 = - loglik* mask_layer 
        loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,
                                                labels=mask_layer)
        loss = loss1 + loss2

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'rate':lbda,
                                   'rawsample':rawsample, 'pred_mask':pred_mask, 'out_mask':out_mask,
                                   'loss':loss, 'loss1':loss1, 'loss2':loss2})


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
    
    loss = predictions['loss']
    loss1, loss2 = tf.reduce_mean(predictions['loss1']), tf.reduce_mean(predictions['loss2'])
    # Compute and register loss function
    loss = tf.reduce_mean(loss)    
    tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)
            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "loss" : loss, 
                "loss1" : loss1, "loss2" : loss2 }, every_n_iter=50)

        tf.summary.scalar('loss', loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])





def _mdn_specres_nozero_mask_poisson_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad=4, lr0=1e-3, nfilter=32, nfilter0 = 2, normwt=0.7, num_iters=1):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    

        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 1)
        batch_size = tf.shape(obs_layer)[0]
        cube_size = tf.shape(obs_layer)[1]
        tileft = tf.tile(feature_layer, [1, 1, 1, 1, 2])
        #       

        # Builds the neural network       
        if nfilter0 == 2: net = tileft
        #if nfilter0 == 2: net = specnormconv3d(feature_layer, nfilter0, 3, name='l00', num_iters=num_iters, normwt=normwt)
        else: net = specnormconv3d(feature_layer, nfilter0, 3, name='l00', num_iters=num_iters, normwt=normwt)

        # Builds the neural network
        subnet = specnormconv3d(net, nfilter, 3, name='l11', num_iters=num_iters, normwt=normwt, padding='VALID')
        subnet = tf.nn.leaky_relu(subnet)
        subnet = specnormconv3d(subnet, nfilter0, 3, name='l12', num_iters=num_iters, normwt=normwt, padding='VALID')
        current_size = tf.shape(subnet)[1]
        short = tf.slice(net, [0, 2, 2, 2, 0], [batch_size, current_size, current_size, current_size, nfilter0])
        net = short + subnet
        net = tf.nn.leaky_relu(net)

        if pad == 4:
            subnet = specnormconv3d(net, nfilter, 3, name='l21', num_iters=num_iters, normwt=normwt, padding='VALID')
            subnet = tf.nn.leaky_relu(subnet)
            subnet = specnormconv3d(subnet, nfilter0, 3, name='l22', num_iters=num_iters, normwt=normwt, padding='VALID')
            current_size = tf.shape(subnet)[1]
            short = tf.slice(net, [0, 2, 2, 2, 0], [batch_size, current_size, current_size, current_size, nfilter0])
            net = short + subnet
            net = tf.nn.leaky_relu(net)
            
        
        # Define the probabilistic layer
        #if nfilter0 != 0: net = specnormconv3d(net, 1, 1, name='lfin', num_iters=num_iters, normwt=normwt)

        print(net)
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, nfilter0])
        print('net : ', net)
        prednet, masknet = tf.split(net, num_or_size_splits=2, axis=-1)
        print('prednet : ', prednet)
        print('masknet : ', masknet)
        
        #Predicted mask
        #out_mask = specnormconv3d(masknet[...,0], 1, 1, name='mask', normwt=normwt)
        #pred_mask = tf.squeeze(tf.nn.sigmoid(out_mask))
        out_mask = tf.identity(masknet[...,0], name='mask')
        pred_mask = tf.squeeze(tf.nn.sigmoid(out_mask))

        #Distribution        
        #lbda = specnormconv3d(prednet[...,0], 1, 1, name='pred', normwt=normwt)
        #lbda = tf.nn.softplus(lbda, name='lambda') + 1e-5
        lbda = tf.nn.softplus(prednet[...,0], name='lambda') + 1e-5
        dist = tfd.Poisson(lbda)
        
        rawsample = tf.squeeze(dist.sample())
        sample = rawsample*pred_mask
        loglik = dist.log_prob(obs_layer)
       
        # Define a function for sampling, and a function for estimating the log likelihood

        loss1 = - loglik* mask_layer 
        loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,
                                                labels=mask_layer)
        loss = loss1 + loss2

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'rate':lbda,
                                   'rawsample':rawsample, 'pred_mask':pred_mask, 'out_mask':out_mask,
                                   'loss':loss, 'loss1':loss1, 'loss2':loss2})


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
    
    loss = predictions['loss']
    loss1, loss2 = tf.reduce_mean(predictions['loss1']), tf.reduce_mean(predictions['loss2'])
    # Compute and register loss function
    loss = tf.reduce_mean(loss)    
    tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)
            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "loss" : loss, 
                "loss1" : loss1, "loss2" : loss2 }, every_n_iter=50)

        tf.summary.scalar('loss', loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])










def _mdn_specres_nozero_poisson_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad=4, lr0=1e-3, nfilter=32, nfilter0 = 1, normwt=0.7, num_iters=1):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 1)
        batch_size = tf.shape(obs_layer)[0]
        cube_size = tf.shape(obs_layer)[1]
        #tileft = tf.tile(feature_layer, [1, 1, 1, 1, 2])
        #       

        # Builds the neural network       
        #if nfilter0 == 2: net = tileft
        #if nfilter0 == 2: net = specnormconv3d(feature_layer, nfilter0, 3, name='l00', num_iters=num_iters, normwt=normwt)
        #else:net = specnormconv3d(feature_layer, nfilter0, 3, name='l00', num_iters=num_iters, normwt=normwt)

        # Builds the neural network
        subnet = specnormconv3d(feature_layer, nfilter, 3, name='l11', num_iters=num_iters, normwt=normwt, padding='VALID')
        subnet = tf.nn.leaky_relu(subnet)
        subnet = specnormconv3d(subnet, nfilter0, 3, name='l12', num_iters=num_iters, normwt=normwt, padding='VALID')
        current_size = tf.shape(subnet)[1]
        short = tf.slice(feature_layer, [0, 2, 2, 2, 0], [batch_size, current_size, current_size, current_size, nfilter0])
        net = short + subnet
        net = tf.nn.leaky_relu(net)

        if pad == 4:
            subnet = specnormconv3d(net, nfilter, 3, name='l21', num_iters=num_iters, normwt=normwt, padding='VALID')
            subnet = tf.nn.leaky_relu(subnet)
            subnet = specnormconv3d(subnet, nfilter0, 3, name='l22', num_iters=num_iters, normwt=normwt, padding='VALID')
            current_size = tf.shape(subnet)[1]
            short = tf.slice(net, [0, 2, 2, 2, 0], [batch_size, current_size, current_size, current_size, nfilter0])
            net = short + subnet
            net = tf.nn.leaky_relu(net)
            

        print(net)
        lbda = tf.nn.softplus(net[...], name='lambda') + 1e-5
        dist = tfd.Poisson(lbda)
        
        sample = tf.squeeze(dist.sample())
        #sample = rawsample*pred_mask
        loglik = dist.log_prob(obs_layer)
       
        # Define a function for sampling, and a function for estimating the log likelihood
        loss = -loglik

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'rate':lbda, 'loss':loss})


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
    
    loss = predictions['loss']
    #loss1, loss2 = tf.reduce_mean(predictions['loss1']), tf.reduce_mean(predictions['loss2'])
    # Compute and register loss function
    loss = tf.reduce_mean(loss)    
    tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)
            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "loss" : loss }, every_n_iter=50)

        tf.summary.scalar('loss', loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])





def _mdn_nozero_poisson_model_fn2(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad=4, lr0=1e-3, nfilter=32, nfilter0 = 1, normwt=0.7, num_iters=1):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 1)
        batch_size = tf.shape(obs_layer)[0]
        cube_size = tf.shape(obs_layer)[1]
        #tileft = tf.tile(feature_layer, [1, 1, 1, 1, 2])
        #       

        # Builds the neural network       
        #if nfilter0 == 2: net = tileft
        #if nfilter0 == 2: net = specnormconv3d(feature_layer, nfilter0, 3, name='l00', num_iters=num_iters, normwt=normwt)
        #else:net = specnormconv3d(feature_layer, nfilter0, 3, name='l00', num_iters=num_iters, normwt=normwt)

        # Builds the neural network
        subnet = specnormconv3d(feature_layer, nfilter, 3, name='l11', num_iters=num_iters, normwt=normwt, padding='VALID', spectral_normed=False)
        subnet = tf.nn.leaky_relu(subnet)
        subnet = specnormconv3d(subnet, nfilter0, 3, name='l12', num_iters=num_iters, normwt=normwt, padding='VALID', spectral_normed=False)
        current_size = tf.shape(subnet)[1]
        short = tf.slice(feature_layer, [0, 2, 2, 2, 0], [batch_size, current_size, current_size, current_size, nfilter0])
        net = short + subnet
        net = tf.nn.leaky_relu(net)

        if pad == 4:
            subnet = specnormconv3d(net, nfilter, 3, name='l21', num_iters=num_iters, normwt=normwt, padding='VALID', spectral_normed=False)
            subnet = tf.nn.leaky_relu(subnet)
            subnet = specnormconv3d(subnet, nfilter0, 3, name='l22', num_iters=num_iters, normwt=normwt, padding='VALID', spectral_normed=False)
            current_size = tf.shape(subnet)[1]
            short = tf.slice(net, [0, 2, 2, 2, 0], [batch_size, current_size, current_size, current_size, nfilter0])
            net = short + subnet
            net = tf.nn.leaky_relu(net)
            

        print(net)
        lbda = tf.nn.softplus(net[...], name='lambda') + 1e-5
        dist = tfd.Poisson(lbda)
        
        sample = tf.squeeze(dist.sample())
        #sample = rawsample*pred_mask
        loglik = dist.log_prob(obs_layer)
       
        # Define a function for sampling, and a function for estimating the log likelihood
        loss = -loglik

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'rate':lbda, 'loss':loss})


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
    
    loss = predictions['loss']
    #loss1, loss2 = tf.reduce_mean(predictions['loss1']), tf.reduce_mean(predictions['loss2'])
    # Compute and register loss function
    loss = tf.reduce_mean(loss)    
    tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)
            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "loss" : loss }, every_n_iter=50)

        tf.summary.scalar('loss', loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])






def _mdn_nozero_poisson_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad=4, lr0=1e-3, nfilter=32, nfilter0 = 1):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """    
        
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        mask_layer = tf.clip_by_value(obs_layer, 0, 1)
        batch_size = tf.shape(obs_layer)[0]
        cube_size = tf.shape(obs_layer)[1]
        #tileft = tf.tile(feature_layer, [1, 1, 1, 1, 2])
        #       

        subnet = slim.conv3d(feature_layer, nfilter, 3, activation_fn=tf.nn.leaky_relu, padding='VALID')
        #subnet = tf.nn.leaky_relu(subnet)
        subnet = slim.conv3d(subnet, nfilter0, 3, activation_fn=None, padding='VALID')
        current_size = tf.shape(subnet)[1]
        short = tf.slice(feature_layer, [0, 2, 2, 2, 0], [batch_size, current_size, current_size, current_size, nfilter0])
        net = short + subnet
        net = tf.nn.leaky_relu(net)

        if pad == 4:
            subnet = slim.conv3d(net, nfilter, 3, activation_fn=tf.nn.leaky_relu, padding='VALID')
            subnet = tf.nn.leaky_relu(subnet)
            subnet = slim.conv3d(subnet, nfilter0, 3, activation_fn=None, padding='VALID')
            current_size = tf.shape(subnet)[1]
            short = tf.slice(net, [0, 2, 2, 2, 0], [batch_size, current_size, current_size, current_size, nfilter0])
            net = short + subnet
            net = tf.nn.leaky_relu(net)
            
        
        # Define the probabilistic layer
        #if nfilter0 != 0: net = specnormconv3d(net, 1, 1, name='lfin', num_iters=num_iters, normwt=normwt)

        print(net)
        lbda = tf.nn.softplus(net[...], name='lambda') + 1e-5
        dist = tfd.Poisson(lbda)
        
        sample = tf.squeeze(dist.sample())
        #sample = rawsample*pred_mask
        loglik = dist.log_prob(obs_layer)
       
        # Define a function for sampling, and a function for estimating the log likelihood
        loss = -loglik

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'loglikelihood':loglik,
                                   'rate':lbda, 'loss':loss})


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
    
    loss = predictions['loss']
    #loss1, loss2 = tf.reduce_mean(predictions['loss1']), tf.reduce_mean(predictions['loss2'])
    # Compute and register loss function
    loss = tf.reduce_mean(loss)    
    tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step=tf.train.get_global_step()
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [lr0, lr0/2, lr0/10, lr0/20, lr0/100, lr0/1000]
            #values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            train_op = optimizer(learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)
            tf.summary.scalar('rate', learning_rate)
            logging_hook = tf.train.LoggingTensorHook({"iter":global_step, "loss" : loss }, every_n_iter=50)

        tf.summary.scalar('loss', loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops, training_hooks = [logging_hook])












### Model
def _mdn_unetmodel_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode, pad, fsize=8, nsub=2, distribution='logistic'):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')

        # Builds the neural network

        if pad == 0:
            d00 = slim.conv3d(feature_layer, fsize, 5, activation_fn=tf.nn.leaky_relu, padding='same')
        elif pad == 2:
            d00 = slim.conv3d(feature_layer, fsize, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
        if pad == 4:
            d00 = slim.conv3d(feature_layer, fsize, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
            d00 = slim.conv3d(d00, fsize*2, 5, activation_fn=tf.nn.leaky_relu, padding='valid')
##        #downsample
##        dd = [[d00]]
##        cfsize = fsize
##        for i in range(nsub):
##            d0 = dd[-1][-1]
##            d1 = wide_resnet(d0, cfsize, activation_fn=tf.nn.leaky_relu)
##            d2 = wide_resnet(d1, cfsize, activation_fn=tf.nn.leaky_relu)
##            dsub = slim.max_pool3d(d2, kernel_size=3, stride=2, padding='SAME')
##            dd.append([d1, d2, dsub])
##            cfsize  *= 2
##
##        #lower layer
##        d0 = dd[-1][-1]
##        d1 = wide_resnet(d0, cfsize, activation_fn=tf.nn.leaky_relu)
##        d2 = wide_resnet(d1, cfsize, activation_fn=tf.nn.leaky_relu)
##
##        up = [[d1, d2]]
##        #upsample
##        for i in range(nsub):
##            cfsize = cfsize // 2 
##            usub = up[-1][-1]
##            dup = dd.pop()
##            u0 = dynamic_deconv3d('up%d'%i, usub, shape=[3,3,3,cfsize], activation=tf.nn.leaky_relu)
##            #u0 = slim.conv3d_transpose(usub, fsize, kernel_size=3, stride=2)
##            uc = tf.concat([u0, dup[1]], axis=-1)
##            u1 = wide_resnet(uc, cfsize, activation_fn=tf.nn.leaky_relu)
##            u2 = wide_resnet(u1, cfsize, activation_fn=tf.nn.leaky_relu)
##            up.append([u0, u1, u1c, u2])
##
##        u0 = up[-1][-1]
##        net = slim.conv3d(u0, 1, 3, activation_fn=tf.nn.tanh)
##
        #downsample #restructure code while doubling filter size
        cfsize = fsize
        d1 = wide_resnet(d00, cfsize, activation_fn=tf.nn.leaky_relu)
        d2 = wide_resnet(d1, cfsize, activation_fn=tf.nn.leaky_relu)
        dd = [d2]
        for i in range(nsub):
            cfsize *= 2
            print(i, cfsize)
            dsub = slim.max_pool3d(dd[-1], kernel_size=3, stride=2, padding='SAME')
            d1 = wide_resnet(dsub, cfsize, activation_fn=tf.nn.leaky_relu)
            d2 = wide_resnet(d1, cfsize, activation_fn=tf.nn.leaky_relu)
            dd.append(d2)

        print(len(dd))
        #upsample
        usub =  dd.pop()
        for i in range(nsub):
            u0 = dynamic_deconv3d('up%d'%i, usub, shape=[3,3,3,cfsize], activation=tf.identity)
            cfsize = cfsize // 2
            print(i, cfsize)
            u0 = slim.conv3d(u0, cfsize, 1, activation_fn=tf.identity, padding='same')
            #u0 = slim.conv3d_transpose(usub, fsize, kernel_size=3, stride=2)
            uc = tf.concat([u0, dd.pop()], axis=-1)
            u1 = wide_resnet(uc, cfsize, activation_fn=tf.nn.leaky_relu)
            u2 = wide_resnet(u1, cfsize, activation_fn=tf.nn.leaky_relu)
            usub = u2
            
        print(len(dd))
        net = slim.conv3d(usub, 1, 3, activation_fn=tf.nn.tanh)

        
        # Define the probabilistic layer 
        net = slim.conv3d(net, n_mixture*3*n_y, 1, activation_fn=None)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_y, n_mixture*3])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=3,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale) + 1e-3

        # Form mixture of discretized logistic distributions. Note we shift the
        # logistic distribution by -0.5. This lets the quantization capture "rounding"
        # intervals, `(x-0.5, x+0.5]`, and not "ceiling" intervals, `(x-1, x]`.
        if distribution == 'logistic':
            discretized_logistic_dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(loc=loc, scale=scale),
                    bijector=tfb.AffineScalar(shift=-0.5)),
                low=0.,
                high=2.**3-1)

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=discretized_logistic_dist)

        elif distribution == 'normal':

            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=tfd.Normal(loc=loc, scale=scale))
            
            
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
            boundaries = list(np.array([1e4, 2e4, 4e4, 5e4, 6e4]).astype(int))
            values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]
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

