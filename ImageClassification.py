import tensorflow as tf

import numpy as np
import math



def ClassifyCategory(imageData):
    
    tf.train.import_meta_graph('./models/producttype/graph.save')
    saver=tf.train.Saver();
    with tf.Session() as sess:
        test_prediction = sess.graph.get_tensor_by_name('test_prediction:0')
        saver.restore(sess,save_path='./models/producttype/train_result')
        sess.run(test_prediction)
if __name__ == "__main__":  
    #ResizeImages("../sample_data/gray_images","../sample_data/100_100",(100,100))
    ClassifyCategory("")   

    