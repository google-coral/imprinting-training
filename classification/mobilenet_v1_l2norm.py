# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MobileNet v1 with L2-normalized embedding as well as L2-normalized 1x1 convolutional weights and zero bias."""

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from classification import conv2d_l2norm
from nets import mobilenet_v1

slim = tf.contrib.slim


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [
        min(shape[1], kernel_size[0]),
        min(shape[2], kernel_size[1])
    ]
  return kernel_size_out


def mobilenet_v1_l2norm(inputs,
                        num_classes=1000,
                        dropout_keep_prob=0.999,
                        is_training=True,
                        min_depth=8,
                        depth_multiplier=1.0,
                        conv_defs=None,
                        prediction_fn=tf.contrib.layers.softmax,
                        spatial_squeeze=True,
                        reuse=None,
                        scope='MobilenetV1',
                        global_pool=False,
                        initial_scale=10.):
  """Mobilenet v1 model for classification.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer (before dropout) are
      returned instead.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels) for
      all convolution ops. The value must be greater than zero. Typical usage
      will be to set this value in (0, 1) to reduce the number of parameters or
      computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
      of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window that
      reduces default-sized inputs to 1x1, while larger inputs lead to larger
      outputs. If true, any input size is pooled down to 1x1.
    initial_scale: normalized weights should be multiplied with a scale to
      alleviate the performance loss. This variable is trainable.

  Returns:
    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: Input rank is invalid.
  """
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))

  with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = mobilenet_v1.mobilenet_v1_base(
          inputs,
          scope=scope,
          min_depth=min_depth,
          depth_multiplier=depth_multiplier,
          conv_defs=conv_defs)
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
          net = slim.avg_pool2d(
              net, kernel_size, padding='VALID', scope='AvgPool_1a')
          end_points['AvgPool_1a'] = net

          net = slim.conv2d(
              net,
              num_outputs=1024,
              kernel_size=[1, 1],
              activation_fn=None,
              normalizer_fn=None,
              scope='Conv2d_1c_1x1_negative')
          net = tf.nn.l2_normalize(net, axis=-1, name='Normalize_1a')
          end_points['Normalize_1a'] = net

        if not num_classes:
          return net, end_points

        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')

        logits = conv2d_l2norm.conv2d(
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='Conv2d_1d_1x1_normalize',
            weight_norm=True)
        scale_factor = tf.get_variable(
            'scale_factor', [],
            dtype=tf.float32,
            initializer=tf.constant_initializer(initial_scale),
            trainable=True)
        slim.summaries.add_scalar_summary(scale_factor, 'scale_factor',
                                          'scale_factors')
        logits = tf.multiply(logits, scale_factor)

        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      if prediction_fn:
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points
