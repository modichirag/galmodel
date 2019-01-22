import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
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


### Model
def _mdn_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode):

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
        net = slim.conv3d(net, 2*n_y, 1, activation_fn=None)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, nchannels, 2])
#         net = tf.reshape(net, [None, None, None, None, n_y, n_mixture*3])
        loc, unconstrained_scale = tf.split(net, num_or_size_splits=2,
                                                    axis=-1)
        print(loc)
        scale = tf.nn.softplus(unconstrained_scale[...,0])
        
        distribution = tfd.MultivariateNormalDiag(loc=loc[...,0], scale_diag=scale)
        
        # Define a function for sampling, and a function for estimating the log likelihood
        sample = tf.squeeze(distribution.sample())
        print('inf dist sample :', distribution.sample())
        logfeature = tf.log1p(feature_layer)
        loglik = distribution.log_prob(logfeature)
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
    print('samples', samples)

    inference = module_inf({'features':features_, 'labels':samples}, as_dict=True)
    
    if mode == tf.estimator.ModeKeys.PREDICT:    
        hub.register_module_for_export(module, "likelihood")
        hub.register_module_for_export(module_inf, "inference")
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loglik = predictions['loglikelihood']
    reg_loglik = inference['loglikelihood']
    print('loglik :', loglik)
    print('reg_loglik :', reg_loglik)
    ####Compute and register loss function
    neg_log_likelihood = -tf.reduce_sum(loglik, axis=-1)
    neg_log_likelihood = tf.reduce_mean(neg_log_likelihood) 
    reg_log_likelihood = - fudge* tf.reduce_mean(reg_loglik)
    
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







# Model
def _mdn_mass_model_fn(features, labels, nchannels, n_y, n_mixture, dropout, optimizer, mode):

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
        net = slim.conv3d(net, n_mixture*6, 1, activation_fn=None)
        cube_size = tf.shape(obs_layer)[1]
        net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, 1, n_mixture*6])
        #net = tf.reshape(net, [-1, cube_size, cube_size, cube_size, n_mixture*6])


        mu, unconstrained_sigma, probs, loc, unconstrained_scale, logits = tf.split(net,
                                                    num_or_size_splits=6,
                                                    axis=-1)
        scale = tf.nn.softplus(unconstrained_scale)
        sigma = tf.nn.relu(unconstrained_sigma) + 1e-2
        mu = tf.expand_dims(mu, -1)
        sigma = tf.expand_dims(sigma, -1)
        print('\nmu\n', mu)
        print('\nsigma\n', sigma)
        print('\nprobs\n', probs)
        print('\nloc\n', loc)
        print('\nscale\n', scale)
        print('\nlogits\n', logits)

        # Form mixture of discretized logistic distributions. Note we shift the
        # logistic distribution by -0.5. This lets the quantization capture "rounding"
        # intervals, `(x-0.5, x+0.5]`, and not "ceiling" intervals, `(x-1, x]`.
        discretized_logistic_dist = tfd.QuantizedDistribution(
            distribution=tfd.TransformedDistribution(
                distribution=tfd.Logistic(loc=loc, scale=scale),
                bijector=tfb.AffineScalar(shift=-0.5)),
            low=0.,
            high=2.**3-1)
        print('\ndiscretized\n', discretized_logistic_dist)

        mixture_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=discretized_logistic_dist)

        print('\mix_pos\n', mixture_dist)
        
        gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=probs),
            components_distribution=tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma))
            #components_distribution=tfd.Normal(loc=mu, scale=sigma))
        print('\gmm\n', gmm)


        
        #loss = - tf.reduce_mean(gmm.log_prob(obs_layer[...,0]) * obs_layer[...,1]) + \
        #       tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_mask,
        #                                                              labels=obs_layer[...,1]))

        # Define a function for sampling, and a function for estimating the log likelihood

        print('\nsample now\n')
        sample = tf.squeeze(mixture_dist.sample())
        print('\nsampled pos \n')
        masses = tf.squeeze(mixture_dist.sample())
        print('\nsampled mass\n')
        print(obs_layer[...,1], gmm.log_prob(obs_layer[...,0]))
        
        mloglik = gmm.log_prob(tf.expand_dims(obs_layer[...,0], -1))
        #mloglik = tf.multiply(mloglik, tf.expand_dims(obs_layer[...,1], -1))
        ploglik = mixture_dist.log_prob(tf.expand_dims(obs_layer[...,1], -1))
        print('\nmloglik\n')
        print(mloglik)
        print('\nmloglik\n')
        print(ploglik)
        loglik = tf.add(ploglik, ploglik)
        print(loglik)
        print('\nloglik\n')
        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                          outputs={'sample':sample, 'masses':masses,
                                   'mloglikelihood':mloglik, 'ploglikelihood':ploglik, 'loglikelihood':loglik})
    


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
    
    loglik = predictions['mloglikelihood']
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








