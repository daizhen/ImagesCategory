import tensorflow as tf

import numpy as np
import math
from tensorflow.python.platform import gfile



a = tf.Variable(10)
b = tf.Variable(2)

pool_va = tf.Variable([1,2,2,1], name="pool")


sess = tf.Session();
init = tf.initialize_all_variables()
sess.run(init)

change_value = pool_va.assign([1,3,3,1])

sess.run(change_value)

print 'trainning done..'


tf.train.write_graph(sess.graph_def, '', 'my_graph.pb', as_text=False)


sess.close()


def Import(sess):
    with gfile.FastGFile("my_graph.pb",'rb') as f:
        graph_def = tf.GraphDef()
        content = f.read()
        graph_def.ParseFromString(content)
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

sess = tf.Session();
Import(sess)