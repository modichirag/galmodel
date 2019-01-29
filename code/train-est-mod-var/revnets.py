import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability
import tensorflow_probability as tfp
import tensorflow_hub as hub
tfd = tensorflow_probability.distributions
tfd = tfp.distributions
tfb = tfp.bijectors
h = 0.1


import sys
sys.path.append('../utils/')
from layers import wide_resnet



  

class Squeeze3d(tfb.Reshape):
    """
    Borrowed from https://github.com/openai/glow/blob/master/tfops.py
    """
    
    def __init__(self,
                 event_shape_in,
                 factor=2,
                 is_constant_jacobian=True,
                 validate_args=False,
                 name=None):

        assert factor >= 1
        name = name or "squeeze"
        self.factor = factor
        event_shape_out = 1*event_shape_in
        event_shape_out[0] //=2
        event_shape_out[1] //=2
        event_shape_out[2] //=2
        event_shape_out[3] *=8
        self.event_shape_out = event_shape_out
        
        super(Squeeze3d, self).__init__(
            event_shape_out=event_shape_out,
            event_shape_in=event_shape_in,
        validate_args=validate_args,
        name=name)
    
    def _forward(self, x):        
        if self.factor == 1:
            return x
        factor = self.factor

        shape = tf.shape(x)
        height = shape[1]
        width = shape[2]
        length = shape[3]
        n_channels = x.get_shape()[4]

#         print(height, width, length, n_channels )
#         assert height % factor == 0 and width % factor == 0 and length % factor == 0
        x = tf.reshape(x, [-1, height//factor, factor,
                           width//factor, factor, length//factor, factor, n_channels])
        x = tf.transpose(x, [0, 1, 3, 5, 7, 2, 4, 6])
        x = tf.reshape(x, [-1, height//factor, width//factor, 
                               length//factor, n_channels*factor**3])
        return x
    
    def _inverse(self, x):        
        if self.factor == 1:
            return x
        factor = self.factor

        shape = tf.shape(x)
        height = shape[1]
        width = shape[2]
        length = shape[3]
        n_channels = int(x.get_shape()[4])
        
#         print(height, width, length, n_channels )
        assert n_channels >= 8 and n_channels % 8 == 0
        x = tf.reshape(
            x, [-1, height, width, length, int(n_channels/factor**3), factor, factor, factor])
        x = tf.transpose(x, [0, 1, 5, 2, 6, 3, 7, 4])
        x = tf.reshape(x, (-1, height*factor,
                           width*factor, height*factor, int(n_channels/factor**3)))
        return x
    

    



class iRevNetsimple(tfb.Bijector):

    def __init__(self,
       h=1.0,
       is_constant_jacobian=True,
       validate_args=False,
       name=None, kernel_size=3):
        self.h = h
        self.kernel_size = kernel_size
        name = name or "revnet"
        super(iRevNetsimple, self).__init__(
        forward_min_event_ndims=3,
        is_constant_jacobian=is_constant_jacobian,
        validate_args=validate_args,
        name=name)
        
    def _forward(self, x):
        # Split the input in half
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        fx2 = tf.layers.conv3d(x2, x2.shape[-1], self.kernel_size,  strides=1, #training=True, 
                          name=self.name+'/bottle', 
                               reuse=tf.AUTO_REUSE, padding='same' )
#         fx2 = wide_resnet(x2, x2.shape[-1],  #training=True, 
#                           scope=self.name+'/bottle')
        y1 = x1 + self.h*fx2
        return tf.concat([x2, y1], axis=-1)
    
    def _inverse(self, x, kernel_size=3):
        x2, y1 = tf.split(x, num_or_size_splits=2, axis=-1)
        Fx2 = - self.h*tf.layers.conv3d(x2, x2.shape[-1], self.kernel_size, strides=1, #training=True, 
                                   name=self.name+'/bottle', 
                                        reuse=tf.AUTO_REUSE , padding='same' )
#         Fx2 = - self.h*wide_resnet(x2, x2.shape[-1], scope=self.name+'/bottle', )
        x1 = Fx2 + y1
        return tf.concat([x1, x2], axis=-1)

    def _inverse_log_det_jacobian(self, y):
        return constant_op.constant(0., x.dtype.base_dtype)

    def _forward_log_det_jacobian(self, x):     
        return constant_op.constant(0., x.dtype.base_dtype)








def _mdn_model_fn(features, labels, nchannels, n_y, dropout, optimizer, mode, loss, softplus=False):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
        
    def _module_fn():
        """
        Function building the module
        """
    
        feature_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
        obs_layer = tf.placeholder(tf.float32, shape=[None, None, None, None, n_y], name='observations')
        shift = tf.Variable(1., dtype=tf.float32, name='shift')
        scale = tf.Variable(1., dtype=tf.float32, name='scale')

        # Builds the neural network
        # ! ny and nchannel need to be the same

        cube_size = tf.shape(feature_layer)[1]
#         print(cube_size)
        chain = tfb.Chain([tfp.bijectors.Affine(shift=shift, scale_identity_multiplier=scale), 
                    tfb.Invert(Squeeze3d(event_shape_in=[cube_size, cube_size, cube_size, nchannels])),
                   iRevNetsimple(name='layer1',h=h),
                   iRevNetsimple(name='layer1b',h=h),
                   iRevNetsimple(name='layer2',h=h),
                   iRevNetsimple(name='layer2b',h=h),
                   #tfb.Permute(np.arange(8)[::-1],axis=-1),
                   tfb.Permute(np.arange(8)[::-1],axis=-1),
                   iRevNetsimple(name='layer3',h=h),
                   iRevNetsimple(name='layer3b',h=h),
                   iRevNetsimple(name='layer4',h=h),
                   iRevNetsimple(name='layer4b',h=h),
                   tfb.Invert(Squeeze3d(event_shape_in=[cube_size//2,cube_size//2,cube_size//2,nchannels*8])), 
                   iRevNetsimple(name='layer5',h=h),
                   iRevNetsimple(name='layer5b',h=h),
                   iRevNetsimple(name='layer6',h=h),
                   iRevNetsimple(name='layer6b',h=h),
                   tfb.Permute(np.arange(64)[::-1],axis=-1),
                   iRevNetsimple(name='layer7',h=h),
                   iRevNetsimple(name='layer7b',h=h),
                   iRevNetsimple(name='layer8',h=h),
                   iRevNetsimple(name='layer8b',h=h),
                   tfb.Invert(Squeeze3d(event_shape_in=[cube_size//4,cube_size//4,cube_size//4,nchannels*64])), 
                   iRevNetsimple(name='layer9',h=h, kernel_size=1),
                   iRevNetsimple(name='layer9b',h=h, kernel_size=1),
                   iRevNetsimple(name='layer10',h=h, kernel_size=1),
                   iRevNetsimple(name='layer10b',h=h, kernel_size=1),
                   tfb.Permute(np.arange(64*8)[::-1],axis=-1),
                   iRevNetsimple(name='layer11',h=h, kernel_size=1),
                   iRevNetsimple(name='layer11b',h=h, kernel_size=1),
                   iRevNetsimple(name='layer12',h=h, kernel_size=1),
                   iRevNetsimple(name='layer12b',h=h, kernel_size=1),
                   Squeeze3d(event_shape_in=[cube_size//4,cube_size//4,cube_size//4,nchannels*64]), 
                   iRevNetsimple(name='layer13',h=h),
                   iRevNetsimple(name='layer13b',h=h),
                   iRevNetsimple(name='layer14',h=h),
                   iRevNetsimple(name='layer14b',h=h),
                   tfb.Permute(np.arange(64)[::-1],axis=-1),
                   iRevNetsimple(name='layer15',h=h),
                   iRevNetsimple(name='layer15b',h=h),
                   iRevNetsimple(name='layer16',h=h),
                   iRevNetsimple(name='layer16b',h=h),
                   Squeeze3d(event_shape_in=[cube_size//2,cube_size//2,cube_size//2,nchannels*8]), 
                   iRevNetsimple(name='layer17',h=h),
                   iRevNetsimple(name='layer17b',h=h),
                   iRevNetsimple(name='layer18',h=h),
                   iRevNetsimple(name='layer18b',h=h),
                   tfb.Permute(np.arange(8)[::-1],axis=-1),
                   iRevNetsimple(name='layer19',h=h),
                   iRevNetsimple(name='layer19b',h=h),
                   iRevNetsimple(name='layer20',h=h),
                   iRevNetsimple(name='layer20b',h=h),
                   Squeeze3d(event_shape_in=[cube_size, cube_size, cube_size, nchannels])])

        bijection = chain

        # Define the probabilistic layer 
        net = bijection.forward(feature_layer, name='lambda')
        if softplus:
            net = tf.nn.softplus(net, name='lambda') 
        dist = tfd.Poisson(net+1e-3)

        sample = tf.squeeze(dist.sample())
#         loglik = dist.log_prob(obs_layer+1)
        loglik = dist.log_prob(obs_layer)

        #l2 = tf.losses.mean_squared_error(obs_layer, net)
        l2 = (tf.square(tf.subtract(obs_layer, net)))
        l1 = (tf.abs(tf.subtract(obs_layer, net)))

        hub.add_signature(inputs={'features':feature_layer, 'labels':obs_layer}, 
                      outputs={'sample':sample, 'loglikelihood':loglik, 'lambda':net, 
                               'l2':l2, 'l1':l1 })
                               #,'shift':shift, 'scale':scale})
    
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
    

    if loss == 'loglikelihood':
         neg_log_likelihood = - predictions['loglikelihood']

    elif loss == 'l2':
        neg_log_likelihood = predictions['l2']
    elif loss == 'l1':
        neg_log_likelihood = predictions['l1']
    else:
        print('Loss not specified')
        
    neg_log_likelihood = tf.reduce_sum(neg_log_likelihood, axis=-1)
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
        #tf.summary.scalar('shift', predictions['shift'])
        #tf.summary.scalar('scale', predictions['scale'])
        tf.summary.scalar('loss', neg_log_likelihood)
    elif mode == tf.estimator.ModeKeys.EVAL:
        
        eval_metric_ops = { "log_p": neg_log_likelihood}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


