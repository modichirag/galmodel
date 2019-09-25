""" Implementation of Cosmic RIM estimator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import flowpm

from galmodel.layers.recurrent_inference import ConvRIM3D
from galmodel.layers.convolutional_recurrent import ConvLSTM3DCell

def pm(lin, params):
  """ FastPM forward simulation
  """
  state = flowpm.lpt_init(lin, a0=params['a0'])
  final_state = flowpm.nbody(state, params['stages'], params['nc'])
  final_field = flowpm.cic_paint(tf.zeros_like(lin), final_state[0])
  return final_field

def get_likelihood_fn(params):
  def likelihood_fn(inputs, predictions):
    final_field = pm(predictions[..., 0], params)
    likelihood = tf.reduce_mean((final_field - inputs[...,0])**2, axis=[1, 2, 3])
    return likelihood
  return likelihood_fn

def model_fn(features, labels, mode, params):
  """
  Model function for the CosmicRIM.
  """
  is_training = mode == tf.estimator.ModeKeys.TRAIN

  # Build the RIM
  cell = ConvLSTM3DCell(filters=params["hidden_size"],
                        kernel_size=params["kernel_size"],
                        padding='SAME')
  output_layer = tf.keras.layers.Conv3D(filters=1,
                                        kernel_size=params["kernel_size"],
                                        padding='SAME')

  cosmic_rim = ConvRIM3D(cell, output_layer,
                         likelihood_fn=get_likelihood_fn(params),
                         niter=params["niter"], return_sequences=True)

  preds = cosmic_rim(features)
  predictions = {'preds':preds,
                 'features': features,
                 'labels':labels}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.tpu.TPUEstimatorSpec(mode=mode,
                                             predictions=predictions)

  batch_loss = tf.reduce_mean((preds - labels)**2, axis=[0, 2, 3, 4, 5])

  loss = tf.reduce_mean(batch_loss)

  train_op = None

  learning_rate = tf.train.exponential_decay(params["learning_rate"],
        tf.train.get_global_step(),
        decay_steps=100000,
        decay_rate=0.96)
  optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
  if params["use_tpu"]:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  # Define optimizer
  if mode == tf.estimator.ModeKeys.TRAIN:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss=loss,
                                     global_step=tf.train.get_global_step())

  return tf.estimator.tpu.TPUEstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op)

def get_cosmic_rim_estimator(hidden_size=128,
                             kernel_size=3,
                             niter=10,
                             learning_rate=0.001,
                             global_batch_size=16,
                             a0=0.1,
                             nstages=2,
                             nc=32,
                             use_tpu=False,
                             run_config=None):
  """
  Builds an Estimator
  """
  params = {'hidden_size':hidden_size,
            'kernel_size':kernel_size,
            'niter':niter,
            'learning_rate':learning_rate,
            'a0':a0,
            'stages':np.linspace(a0, 1.0, nstages, endpoint=True),
            'nc':nc,
            'use_tpu':use_tpu}

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=use_tpu,
      train_batch_size=global_batch_size,
      eval_batch_size=global_batch_size,
      predict_batch_size=global_batch_size,
      params=params,
      config=run_config)
  return estimator
