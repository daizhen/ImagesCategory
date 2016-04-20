import tensorflow as tf

import numpy as np

hello = tf.constant("Hello");
sess = tf.Session()
print sess.run(hello);


# run a + b
a = tf.Variable(1)
b = tf.Variable(2)
x = np.ndarray([(1,2,3,4),(1,2,3,4)])
d = tf.constant(x)
init = tf.initialize_all_variables()
sess.run(init)
c = sess.run(a+b)

print c
a = a+1
b = b+1

c = sess.run(a+b)


print c
