import os
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

import numpy as np
import sys
sys.path.append('../util/')
import ImageUtil;
import freeze_graph
def validate():
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
                
            prediction_result = sess.run(check_prediction,feed_dict={check_data_node: np.reshape(np.ones(10000),(1,100,100,1))})
            #print(conv1_weights)
            #output = sess.run(output1)
            print(prediction_result)
            prediction_index = np.argmax(prediction_result, 1)[0];
            print(prediction_index)
            #self.assertNear(2.0, output, 0.00001)