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
r"""Saves out a GraphDef containing the architecture of the model.

bazel build :export_inference_graph_l2norm
bazel-bin/export_inference_graph_l2norm \
--output_file=/tmp/mobilenet_v1_l2norm_inf_graph.pb
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf

from classification import mobilenet_v1_l2norm
from nets import mobilenet_v1

slim = tf.contrib.slim

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer('image_size', 224, 'The image size to use.')

tf.app.flags.DEFINE_integer('num_classes', 1001,
                            'Number of classes to distinguish')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string('output_file', '',
                           'Where to save the resulting file to.')

tf.app.flags.DEFINE_bool('quantize', False,
                         'Whether to use quantized graph or not.')

tf.app.flags.DEFINE_bool('write_text_graphdef', False,
                         'Whether to write a text version of graphdef.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.output_file:
    raise ValueError('You must supply the path to save to with --output_file')
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as graph:
    image_size = FLAGS.image_size
    input_shape = [FLAGS.batch_size, image_size, image_size, 3]
    placeholder = tf.placeholder(
        name='input', dtype=tf.float32, shape=input_shape)
    scope = mobilenet_v1.mobilenet_v1_arg_scope(
        is_training=False, weight_decay=0.0)
    with slim.arg_scope(scope):
      mobilenet_v1_l2norm.mobilenet_v1_l2norm(
          placeholder, is_training=False, num_classes=FLAGS.num_classes)

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    graph_def = graph.as_graph_def()
    if FLAGS.write_text_graphdef:
      tf.io.write_graph(
          graph_def,
          os.path.dirname(FLAGS.output_file),
          os.path.basename(FLAGS.output_file),
          as_text=True)
    else:
      with tf.gfile.GFile(FLAGS.output_file, 'wb') as f:
        f.write(graph_def.SerializeToString())


if __name__ == '__main__':
  tf.app.run()
