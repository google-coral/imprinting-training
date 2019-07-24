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
"""Tests for export_inference_graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from classification import export_inference_graph_l2norm


class ExportInferenceGraphTest(tf.test.TestCase):

  def testExportInferenceGraph(self):
    tmpdir = self.get_temp_dir()
    output_file = os.path.join(tmpdir, 'eval_graph.pb')
    flags = tf.app.flags.FLAGS
    flags.output_file = output_file
    export_inference_graph_l2norm.main(None)
    self.assertTrue(tf.gfile.Exists(output_file))


if __name__ == '__main__':
  tf.test.main()
