import tensorflow as tf

import numpy as np
import math
from tensorflow.python.platform import gfile

from tensorflow.python.tools import freeze_graph

IMAGE_SIZE = 100
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 3
VALIDATION_SIZE = 200  # Size of the validation set.
SEED = 6478  # Set to None for random seed.
BATCH_SIZE = 60
NUM_EPOCHS = 10

conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED), name='conv1_weights')
conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')
	
conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED), name='conv2_weights')
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='conv2_biases')
	
fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE / 4 * IMAGE_SIZE / 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED), name='fc1_weights')
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]), name='fc1_biases')
fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED), name='fc2_weights')
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='fc2_biases')

batch = tf.Variable(0)

varlist = [conv1_weights,conv1_biases,conv2_weights,conv2_biases,fc1_weights,fc1_biases,fc2_weights,fc2_biases]

sess = tf.Session()
saver=tf.train.Saver();
#saver.restore(sess,save_path='../models/producttype/train_result')


def Import(sess):
    with gfile.FastGFile("../models/producttype/graph.pb",'rb') as f:
        graph_def = tf.GraphDef()
        content = f.read()
        graph_def.ParseFromString(content)
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')



Import(sess)


sess.run(tf.initialize_all_variables())

saver.restore(sess,save_path='../models/producttype/train_result')
#this is placeholder

check_data_node = sess.graph.get_tensor_by_name('check_data_node:0')
check_prediction = sess.graph.get_tensor_by_name('check_prediction:0')

#sess.run(init)
feed_dict={check_data_node: np.reshape(np.ones(10000),[1,100,100,1]).astype(np.float32)}
print sess.run(check_prediction, feed_dict= feed_dict)
print sess.run(fc1_weights)
sess.close()



input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
input_saver_def_path = ""
input_binary = False
input_checkpoint_path = checkpoint_prefix + "-0"
output_node_names = "output_node"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = os.path.join(self.get_temp_dir(), output_graph_name)
clear_devices = False
    