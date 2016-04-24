import tensorflow as tf

import numpy as np
import math


a = tf.Variable(10)
b = tf.Variable(2)



pool_va = tf.Variable([1,2,2,1])

c =tf.div(a,b);
sess = tf.Session();
init = tf.initialize_all_variables()
sess.run(init)

x= tf.placeholder(tf.float32, shape=[1,None,None,1])
k = tf.placeholder(tf.int32, shape=[4])


change_value = pool_va.assign([1,3,3,1])

sess.run(change_value)

eval_value = pool_va.eval(sess).tolist()
print eval_value

pool_1 = tf.nn.max_pool(x,eval_value,[1,1,1,1],padding='VALID');

feed_dict = {x: np.reshape([1,2,3,4,5,6,7,8],(1,2,4,1)),}
                         
print sess.run(pool_1,feed_dict = feed_dict)

sess.run(pool_va.assign([1,3,3,1]))

pool_2 = tf.nn.max_pool(x,eval_value,[1,1,1,1],padding='VALID');

def TestDynamicPool():
    pass
sess.run(c)
 
saver = tf.train.Saver()
saver.save(sess,save_path='./train_data')
print b.eval(sess)
print sess.run(c)


saver.export_meta_graph


saver.restore(sess, save_path='../train_result.meta')
#saver.restore(sess, save_path='../train_result')
#xx = sess.run(conv1_weights)



sess.close()   



