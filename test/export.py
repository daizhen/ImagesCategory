import tensorflow as tf

import numpy as np
import math


a = tf.Variable(10)
b = tf.Variable(2)

pool_va = tf.Variable([1,2,2,1], name="pool")

store_list = [a,b,pool_va]


saver=tf.train.Saver(store_list);
sess = tf.Session();

init = tf.initialize_all_variables()
sess.run(init)

change_value = pool_va.assign([1,3,3,1])

sess.run(change_value)

#tf.train.export_meta_graph(filename='graph.save', as_text=True)

saver=tf.train.Saver(store_list);
saver.save(sess,save_path='./train_result')
#close the session 
sess.close();

sess = tf.Session();
saver=tf.train.Saver(store_list);
saver.restore(sess,save_path='./train_result')
pool =sess.graph.get_tensor_by_name('pool:0')
print sess.run(pool)
sess.close();
