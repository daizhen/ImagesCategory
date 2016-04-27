import tensorflow as tf

import numpy as np
import math


#a = tf.Variable(10)
def Import():
    tf.import_meta_graph('graph.save')
#Start session again
sess = tf.Session()
#Import()

#pool_va = sess.graph.get_tensor_by_name('pool:0')

#init = tf.initialize_all_variables()
saver=tf.train.Saver();
saver.restore(sess,save_path='./train_result')

#sess.run(init)

sess.close()

    