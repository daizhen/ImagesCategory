import tensorflow as tf
import numpy as np
import math

'''
a = tf.Variable(1)
b = tf.Variable(2)
pool_va = tf.Variable([1,2,2,1], name="pool")

store_list = [a,b,pool_va]

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

print sess.run(a)
print sess.run(pool_va)
#sess.run(init)

sess.close()
'''

class TestImport:
    a = tf.Variable(1)
    b = tf.Variable(2)
    pool_va = tf.Variable([1,2,2,1], name="pool")
    store_list = [a,b,pool_va]
    def import_test(self):
        sess = tf.Session()
        saver=tf.train.Saver();
        saver.restore(sess,save_path='./train_result')
        print sess.run(self.pool_va)
    '''
    def import(self):
        sess = tf.Session()
        saver=tf.train.Saver();
        saver.restore(sess,save_path='./train_result')
        print sess.run(self.a)
        '''
if __name__ == "__main__":
    test = TestImport()
    test.import_test()