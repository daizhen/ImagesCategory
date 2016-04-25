import tensorflow as tf

import numpy as np
import math
from tensorflow.python.platform import gfile


def Import(sess):
    with gfile.FastGFile("my_graph.pb",'rb') as f:
        graph_def = tf.GraphDef()
        content = f.read()
        graph_def.ParseFromString(content)
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
#Start session again
sess = tf.Session()
Import(sess)

pool_va = sess.graph.get_tensor_by_name('pool:0')

#init = tf.initialize_all_variables()
saver=tf.train.Saver();
saver.restore(sess,save_path='./train_result')

#sess.run(init)

print sess.run(pool_va)
sess.close()

    