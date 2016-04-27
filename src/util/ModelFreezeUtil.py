from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import os
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
import freeze_graph

def SaveModel(config):
    # We'll create an input graph that has a single variable containing 1.0,
    # and that then multiplies it by 2.
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

def RestoreModel(config):
    pass