import tensorflow as tf

import numpy as np
import math


a = tf.Variable(10)
b = tf.Variable(2)




c =tf.div(a,b);
sess = tf.Session();
init = tf.initialize_all_variables()
sess.run(init)

print b.eval(sess)
print sess.run(c)