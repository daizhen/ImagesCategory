import tensorflow as tf

import numpy as np

hello = tf.constant("Hello");
sess = tf.Session()
print sess.run(hello);


# run a + b
a = tf.Variable(1)
b = tf.Variable(2)
x1 = list()
x1.append([1,2,3])
x1.append([4,3])
x = [[1,2,3,4],[1,2,3,4]]
print x
print x1

d = tf.constant(x1, shape=tf.TensorShape([]), dtype=tf.float32)
init = tf.initialize_all_variables()
sess.run(init)
c = sess.run(a+b)

print c
a = a+1
b = b+1

c = sess.run(a+b)


print c
