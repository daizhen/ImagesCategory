# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=g-bad-import-order,unused-import
"""Tests the graph freezing tool."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import os
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
import freeze_graph
import numpy as np
import sys
sys.path.append('../src/util/')
import ModelFreezeUtil;

class FreezeGraphTest(test_util.TensorFlowTestCase):
    def testFreezeGraph(self):
        
        model_folder = '../models/producttype/'
        checkpoint_prefix = os.path.join(model_folder, "saved_checkpoint")
        checkpoint_state_name = "checkpoint_state"
        input_graph_name = "input_graph.pb"
        output_graph_name = "output_graph.pb"

        output_graph_path = os.path.join(model_folder, output_graph_name)
        # We'll create an input graph that has a single variable containing 1.0,
        # and that then multiplies it by 2.
        
        #variable_node = tf.Variable(1.0, name="variable_node")
        '''
        with tf.Graph().as_default():
        
            variable_node = tf.Variable(1.0, name="variable_node")
            output_node = tf.mul(variable_node, 3.0, name="output_node")
            output_node2 = tf.mul(output_node, 4.0, name="output_node2")
            placeholder_1 = tf.placeholder(dtype=tf.float32,shape=[2], name='placeholder_1')
            placeholder_2 = tf.placeholder(dtype=tf.float32,shape=[2], name='placeholder_2')
            
            add_node = tf.add(placeholder_1,placeholder_2, name="Add2")
            sess = tf.Session()
            init = tf.initialize_all_variables()
            sess.run(init)
            output = sess.run(output_node)
            self.assertNear(3.0, output, 0.00001)
            saver = tf.train.Saver()
            saver.save(sess, checkpoint_prefix, global_step=0,
                        latest_filename=checkpoint_state_name)
            tf.train.write_graph(sess.graph.as_graph_def(), self.get_temp_dir(),
                                input_graph_name)

        # We save out the graph to disk, and then call the const conversion
        # routine.
        input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
        input_saver_def_path = ""
        input_binary = False
        input_checkpoint_path = checkpoint_prefix + "-0"
        output_node_names = "output_node,Add2"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        clear_devices = False

        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                input_binary, input_checkpoint_path,
                                output_node_names, restore_op_name,
                                filename_tensor_name, output_graph_path,
                                clear_devices)
        '''                            
        # Now we make sure the variable is now a constant, and that the graph still
        # produces the expected result.
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            #self.assertEqual(7, len(output_graph_def.node))
            for node in output_graph_def.node:
                self.assertNotEqual("Variable", node.op)
                print(node.name)
                
            with tf.Session() as sess:
                conv1_weights = sess.graph.get_tensor_by_name("conv1_weights:0")
                check_data_node = sess.graph.get_tensor_by_name("check_data_node:0")
                check_prediction = sess.graph.get_tensor_by_name("check_prediction:0")
                #placeholder_2_1 = sess.graph.get_tensor_by_name("placeholder_2:0")
            
                #add_node = sess.graph.get_tensor_by_name("Add2:0")
                
                output1 = sess.run(check_prediction,feed_dict={check_data_node: np.reshape(np.ones(10000),(1,100,100,1))})
                #print(conv1_weights)
                #output = sess.run(output1)
                print(output1)
                #self.assertNear(2.0, output, 0.00001)
                    
if __name__ == "__main__":
  tf.test.main()