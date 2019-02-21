import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
import warnings


###Following spectral normed conv3d has been contributed by Francois Lanusse
NO_OPS = 'NO_OPS'
def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
  # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1
    
    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32),
                   u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
      )
    
    if update_collection is None:
        warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                  '. Please consider using a update collection instead.')
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([u.assign(u_final)]):
            W_bar = tf.reshape(W_bar, W_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)
        # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
        # has already been collected on the first call.
        if update_collection != NO_OPS:
            tf.add_to_collection(update_collection, u.assign(u_final))
    if with_sigma:
        return W_bar, sigma
    else:
        return W_bar


@add_arg_scope
def scope_has_variables(scope):
    return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0

@add_arg_scope
def specnormconv3d(input_, output_dim,
           kernel_size=3, stride=1, stddev=None,
           name="conv3d", spectral_normed=True, update_collection=None, with_w=False, 
             padding="SAME", reuse=tf.AUTO_REUSE):

    k_h, k_w, k_z = [kernel_size]*3
    d_h, d_w, d_z = [stride]*3 
    # Glorot intialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
    fan_in = k_h * k_w * k_z * input_.get_shape().as_list()[-1]
    fan_out = k_h * k_w * k_z * output_dim
    if stddev is None:
        stddev = tf.sqrt(2. / (fan_in))

    with tf.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()
        w = tf.get_variable("w", [k_h, k_w, k_z, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if spectral_normed:
            conv = tf.nn.conv3d(input_, spectral_normed_weight(w, update_collection=update_collection),
                              strides=[1, d_h, d_w, d_z, 1], padding=padding)
        else:
            conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_z, 1], padding=padding)

        biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if with_w:
            return conv, w, biases
        else:
            return conv



###Following have been taken from https://github.com/openai/glow/blob/master/tfops.py
###and modified to 3D. In addition, some variable names have been modified
###and assumptions on shape of input being constant relaxed
@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False, trainable=True):
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w

@add_arg_scope
# def actnorm3d(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, 
#             reverse=False, init=False, trainable=True):
#     if arg_scope([get_variable_ddi], trainable=trainable):
def actnorm3d(x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, 
            reverse=False, init=False, is_training=True, scope='actnorm'):
    name = scope
    if arg_scope([get_variable_ddi], trainable=is_training):
        if not reverse:
            x = actnorm_center3d(name+"_center", x, reverse)
            x = actnorm_scale3d(name+"_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
        else:
            x = actnorm_scale3d(name + "_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
            x = actnorm_center3d(name+"_center", x, reverse)
        if logdet != None:
            return x, logdet
        return x

# Activation normalization


@add_arg_scope
def actnorm_center3d(name, x, reverse=False):
    shape = x.get_shape()
    with tf.variable_scope(name):
        assert len(shape) == 5
        x_mean = tf.reduce_mean(x, [0, 1, 2, 3], keepdims=True)
        b = get_variable_ddi(
#             "b", (1, 1, 1, 1, int_shape(x)[4]), initial_value=-x_mean)
            "b", (1, 1, 1, 1, shape[4]), initial_value=-x_mean)

        if not reverse:
            x += b
        else:
            x -= b

        return x

# Activation normalization


@add_arg_scope
def actnorm_scale3d(name, x, scale=1., logdet=None, logscale_factor=3., 
                  batch_variance=False, reverse=False, init=False, trainable=True):
    shape = x.get_shape()
    with tf.variable_scope(name), arg_scope([get_variable_ddi], trainable=trainable):
        assert len(shape) == 5
        x_var = tf.reduce_mean(x**2, [0, 1, 2, 3], keepdims=True)
#         logdet_factor = int(shape[1])*int(shape[2])*int(shape[3])
        logdet_factor = (shape[1])*(shape[2])*(shape[3])
#         _shape = (1, 1, 1, 1, int_shape(x)[4])
        _shape = (1, 1, 1, 1, shape[4])

        if batch_variance:
            x_var = tf.reduce_mean(x**2, keepdims=True)

        if True:
            logs = get_variable_ddi("logs", _shape, initial_value=tf.log(
                scale/(tf.sqrt(x_var)+1e-6))/logscale_factor)*logscale_factor
            if not reverse:
                x = x * tf.exp(logs)
            else:
                x = x * tf.exp(-logs)
        else:
            # Alternative, doesn't seem to do significantly worse or better than the logarithmic version above
            s = get_variable_ddi("s", _shape, initial_value=scale /
                                 (tf.sqrt(x_var) + 1e-6) / logscale_factor)*logscale_factor
            logs = tf.log(tf.abs(s))
            if not reverse:
                x *= s
            else:
                x /= s

        if logdet != None:
            dlogdet = tf.reduce_sum(logs) * logdet_factor
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x
