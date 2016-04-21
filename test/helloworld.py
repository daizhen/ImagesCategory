import tensorflow as tf

import numpy as np
import math

hello = tf.constant("Hello");
sess = tf.Session()
print sess.run(hello);


# run a + b
a = tf.Variable(1.2)
b = tf.Variable(2.1)

x1 = list()
x1.append([1,2,3])
x1.append([4,3])
x = [[1,2,3,4],[1,2,3,4]]
print x
print x1


print sess.run(tf.div(a,b))





v = tf.Variable(x)
init = tf.initialize_all_variables()
sess.run(init)
c = sess.run(a+b)
v2 = sess.run(v)
print v2
print c
a = a+1
b = b+1

c = sess.run(a+b)


print c
