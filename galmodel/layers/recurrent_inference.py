"""Recurrent Inference Machine layers.
Adapted from the Keras tensorflow backend rnn function.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

import tensorflow as tf

from .convolutional_recurrent import ConvRNN3D

def rim(step_function,
        inputs,
        initial_states,
        niter,
        constants=None):
  """Runs a recurrent inference machine defined by an RNN cell for a number of
  steps.
  Arguments:
      step_function: RNN step function.
          Args;
              input; Tensor with shape `(samples, ...)` (no time dimension),
                  representing input for the batch of samples at a certain
                  time step.
              states; List of tensors.
          Returns;
              output; Tensor with shape `(samples, output_dim)`
                  (no time dimension).
              new_states; List of tensors, same length and shapes
                  as 'states'. The first state in the list must be the
                  output tensor at the previous timestep.
      inputs: Tensor of observed data `(samples, ...)`
      initial_states: Tensor with shape `(samples, state_size)`
          , containing the initial values for the states used
          in the step function. In the case that state_size is in a nested
          shape, the shape of initial_states will also follow the nested
          structure.
      likelihood_fn: Likelihood function used by the RIM, should accept as
          inputs (data, pred) and return a batched scalar.
      niter: Number of iterations to run the RIM for
      constants: List of constant values passed at each step.

  Returns:
      A tuple, `(last_output, outputs, new_states)`.
          last_output: the latest output of the rnn, of shape `(samples, ...)`
          outputs: tensor with shape `(samples, time, ...)` where each
              entry `outputs[s, t]` is the output of the step function
              at time `t` for sample `s`.
          new_states: list of tensors, latest states returned by
              the step function, of shape `(samples, ...)`.
  Raises:
      ValueError: if input dimension is less than 3.
  """
  batch = inputs.shape[0]

  if constants is None:
    constants = []

  # output_time_zero is used to determine the cell output shape and its dtype.
  # the value is discarded.
  output_time_zero, _ = step_function(inputs, tuple(initial_states) + tuple(constants))

  output_ta = tuple(
      tensor_array_ops.TensorArray(
          dtype=out.dtype,
          size=niter,
          tensor_array_name='output_ta_%s' % i)
      for i, out in enumerate(nest.flatten(output_time_zero)))

  states = tuple(initial_states)

  time = constant_op.constant(0, dtype='int32', name='time')

  while_loop_kwargs = {
      'cond': lambda time, *_: time < niter,
      'maximum_iterations': niter,
      'parallel_iterations': 32,
      'swap_memory': True,
  }

  def _step(time, output_ta_t, *states):
    """RNN step function.
    Arguments:
        time: Current timestep value.
        current_solution: current solution
        output_ta_t: TensorArray.
        *states: List of states.
    Returns:
        Tuple: `(time + 1, output_ta_t) + tuple(new_states)`
    """
    # The input to the RIM is the current solution,
    output, new_states = step_function(inputs,
                                       tuple(states) + tuple(constants))
    flat_state = nest.flatten(states)
    flat_new_state = nest.flatten(new_states)
    for state, new_state in zip(flat_state, flat_new_state):
      if isinstance(new_state, ops.Tensor):
        new_state.set_shape(state.shape)

    flat_output = nest.flatten(output)
    output_ta_t = tuple(
        ta.write(time, out) for ta, out in zip(output_ta_t, flat_output))
    new_states = nest.pack_sequence_as(initial_states, flat_new_state)
    return (time + 1, output_ta_t) + tuple(new_states)

  final_outputs = control_flow_ops.while_loop(
      body=_step,
      loop_vars=(time, output_ta) + states,
      **while_loop_kwargs)

  new_states = final_outputs[2:]
  output_ta = final_outputs[1]

  outputs = tuple(o.stack() for o in output_ta)
  last_output = tuple(o[-1] for o in outputs)

  outputs = nest.pack_sequence_as(output_time_zero, outputs)
  last_output = nest.pack_sequence_as(output_time_zero, last_output)

  # static shape inference
  def set_shape(output_):
    print(output_)
    if isinstance(output_, ops.Tensor):
      shape = output_.shape.as_list()
      shape[0] = niter
      shape[1] = batch
      output_.set_shape(shape)
    return output_

  outputs = nest.map_structure(set_shape, outputs)

  return last_output, outputs, new_states


class ConvRIM3D(ConvRNN3D):
  """Base class for convolutional recurrent inference machine layers.
  """

  def __init__(self,
               cell,          # Normal RNN cell
               output_layer,  # Output layer turning the RNN cell into the desired output dim
               likelihood_fn, # Likelihood function used for computing gradients
               niter,         # Number of iterations to run the RIM for
               **kwargs):

    super(self.__class__, self).__init__(cell, **kwargs)
    self.output_layer = output_layer
    self.likelihood_fn = likelihood_fn
    self.niter = niter
    self.input_spec = [InputSpec(ndim=5)]

  def get_initial_state(self, inputs):
    # (samples, timesteps, rows, cols, z, filters)
    initial_state = K.zeros_like(inputs)
    d = [1,2,1,1,1] if self.cell.data_format == 'channels_first' else [1,1,1,1,2]
    initial_state = K.tile(initial_state, d)
    shape = list(self.cell.kernel_shape)
    shape[-1] = self.cell.filters
    initial_state = self.cell.input_conv(initial_state,
                                         array_ops.zeros(tuple(shape)),
                                         padding=self.cell.padding)

    if hasattr(self.cell.state_size, '__len__'):
      return [initial_state for _ in self.cell.state_size]
    else:
      return [initial_state]


  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # Note input_shape will be list of shapes of initial states and
    # constants if these are passed in __call__.
    if self._num_constants is not None:
      constants_shape = input_shape[-self._num_constants:]  # pylint: disable=E1130
    else:
      constants_shape = None

    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    batch_size = input_shape[0] if self.stateful else None
    self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:6])

    # allow cell (if layer) to build before we set or validate state_spec
    if isinstance(self.cell, Layer):
      step_input_shape = list((input_shape[0],) + input_shape[1:])
      if self.cell.data_format == 'channels_first':
        d = 1
      else:
        d = 4
      step_input_shape = [2*c if i == d else c for i,c in enumerate (step_input_shape)]

      if constants_shape is not None:
        self.cell.build([step_input_shape] + constants_shape)
      else:
        self.cell.build(step_input_shape)

    # set or validate state_spec
    if hasattr(self.cell.state_size, '__len__'):
      state_size = list(self.cell.state_size)
    else:
      state_size = [self.cell.state_size]

    if self.state_spec is not None:
      # initial_state was passed in call, check compatibility
      if self.cell.data_format == 'channels_first':
        ch_dim = 1
      elif self.cell.data_format == 'channels_last':
        ch_dim = 4
      if [spec.shape[ch_dim] for spec in self.state_spec] != state_size:
        raise ValueError(
            'An initial_state was passed that is not compatible with '
            '`cell.state_size`. Received `state_spec`={}; '
            'However `cell.state_size` is '
            '{}'.format([spec.shape for spec in self.state_spec],
                        self.cell.state_size))
    else:
      if self.cell.data_format == 'channels_first':
        self.state_spec = [InputSpec(shape=(None, dim, None, None, None))
                           for dim in state_size]
      elif self.cell.data_format == 'channels_last':
        self.state_spec = [InputSpec(shape=(None, None, None, None, dim))
                           for dim in state_size]

    self.output_layer.build(self.state_spec[0].shape)

    if self.stateful:
      self.reset_states()
    self.built = True

  def call(self,
           inputs,
           mask=None,
           training=None,
           initial_state=None,
           constants=None):

    if isinstance(inputs, list):
      inputs = inputs[0]
    if initial_state is not None:
      pass
    elif self.stateful:
      initial_state = self.states
    else:
      initial_state = self.get_initial_state(inputs)

    if isinstance(mask, list):
      mask = mask[0]

    if len(initial_state) != len(self.states):
      raise ValueError('Layer has ' + str(len(self.states)) +
                       ' states but was passed ' +
                       str(len(initial_state)) +
                       ' initial states.')

    timesteps = self.niter

    kwargs = {}
    if generic_utils.has_arg(self.cell.call, 'training'):
      kwargs['training'] = training

    if constants:
      if not generic_utils.has_arg(self.cell.call, 'constants'):
        raise ValueError('RNN cell does not support constants')

      def step(inputs, states):
        constants = states[-self._num_constants:]
        states = states[:-self._num_constants]
        return self.cell.call(inputs, states, constants=constants,
                              **kwargs)
    else:
      def step(inputs, states):
        return self.cell.call(inputs, states, **kwargs)

    # Augment the RNN cell with the likelihood gradient
    def augmented_step(x, states):
      prediction = tf.stop_gradient(self.output_layer.call(states[0]))
      grad = tf.gradients(self.likelihood_fn(x, prediction), x)[0]
      return step(tf.concat([x, grad], axis=-1), states)

    last_output, outputs, states = rim(augmented_step,
                                      inputs,
                                      initial_state,
                                      constants=constants,
                                      niter=timesteps)
    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append(K.update(self.states[i], states[i]))
      self.add_update(updates, inputs=True)

    if self.return_sequences:
      output = outputs
      ni, nb, a,b,c, d = output.shape
      shape = [-1, a, b, c, d]
      output = self.output_layer.call(tf.reshape(output, shape))
      output = tf.reshape(output, [-1, nb, a, b, c, output.shape[-1]])
    else:
      output = last_output
      output = self.output_layer.call(output)

    if self.return_state:
      if not isinstance(states, (list, tuple)):
        states = [states]
      else:
        states = list(states)
      return [output] + states
    else:
      return output
