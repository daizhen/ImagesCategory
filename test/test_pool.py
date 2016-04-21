import tensorflow as tf

import numpy as np

data = np.ones((1,12,13,1), dtype='float32')


pool_1 = tf.nn.max_pool(data, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')

pool_2 = tf.nn.max_pool(data, ksize=[1,4,3,1],strides=[1,4,3,1], padding='VALID')

pool_shape_1 = pool_1.get_shape().as_list()
pool_shape_2 = pool_2.get_shape().as_list()
reshape_1 = tf.reshape(
            [pool_1,pool_2],
            [pool_shape_1[0], pool_shape_1[1] * pool_shape_1[2] * pool_shape_1[3] + pool_shape_2[1] * pool_shape_2[2] * pool_shape_2[3]])


sess = tf.Session()

x = sess.run(reshape_1)
y = sess.run(pool_2).shape
print x
print y