import tensorflow as tf

import numpy as np
import math


a = tf.Variable(10)
b = tf.Variable(2)

pool_va = tf.Variable([1,2,2,1], name="pool")


sess = tf.Session();
init = tf.initialize_all_variables()
sess.run(init)

change_value = pool_va.assign([1,3,3,1])

sess.run(change_value)

tf.train.export_meta_graph(filename='graph.save', as_text=True)

saver=tf.train.Saver();
saver.save(sess,save_path='./train_result')
#close the session 
sess.close();
