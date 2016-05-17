import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append('../util/')

import DataUtil as DataUtil
import TextVectorUtil as TextVectorUtil
import ModelUtil as ModelUtil


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden,istrain): 
    if istrain:
        X = tf.nn.dropout(X, p_drop_input)
    h = tf.nn.relu(tf.matmul(X, w_h))
    if istrain:
        h = tf.nn.dropout(h, p_drop_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    if istrain:
        h2 = tf.nn.dropout(h2, p_drop_hidden)

    return tf.matmul(h2, w_o)
    
def train():
    tokenDict = TextVectorUtil.GetAllTokenDict('../../data/all_trainning_tokens.csv')
    all_labels = DataUtil.LoadAllLabels('../../category_name_id_map.csv')
    train_data,train_labels = DataUtil.LoadTextTokenList('../../data/trainning_data.csv')
    
    validation_data,validation_labels = DataUtil.LoadTextTokenList('../../data/validation_data.csv')
    
    train_size = train_data.shape[0]
    validation_size = validation_data.shape[0]
    
    
    train_labels = [item[0] for item in train_labels]
    # Convert labels to softmax matrix
    label_list = np.ndarray(shape=[train_size], dtype=np.float32)
    for index in range(train_size):
        label_list[index] = np.where(all_labels == train_labels[index])[0][0]
    
    train_labels = (np.arange(len(all_labels)) == label_list[:, None]).astype(np.float32)
    
    validation_labels = [item[0] for item in validation_labels]
    # Convert labels to softmax matrix
    label_list = np.ndarray(shape=[validation_size], dtype=np.float32)
    for index in range(validation_size):
        label_list[index] = np.where(all_labels == validation_labels[index])[0][0]
    
    validation_labels = (np.arange(len(all_labels)) == label_list[:, None]).astype(np.float32)

    w_h = init_weights([len(tokenDict), 400])
    w_h2 = init_weights([400, 400])
    w_o = init_weights([400, len(all_labels)])

    X = tf.placeholder("float", [None, len(tokenDict)])
    Y = tf.placeholder("float", [None, len(all_labels)])
    
    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    
    py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden,True)
    validation_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden,False)
    
    def CaculateErrorRate(session,dataList,labels):
        data_size = dataList.shape[0]
        errorCount = 0;
        for step in xrange(int(data_size / 100)):
            offset = (step * 100)
            batch_data = dataList[offset:(offset + 100)]
            batch_text_data_vector = TextVectorUtil.BuildText2DimArray(batch_data,tokenDict)
            batch_labels = labels[offset:(offset + 100)]
            feed_dict={X: batch_text_data_vector, Y: batch_labels,p_keep_input: 1.0,p_keep_hidden: 1.0}
            # Run the graph and fetch some of the nodes.
            #print batch_data.shape
            #print batch_labels.shape
            #print train_labels
            validation_prediction_result = session.run(validation_x,feed_dict=feed_dict)
            errorCount += ModelUtil.error_count(validation_prediction_result,batch_labels)
        return  errorCount *100.0/ data_size
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.RMSPropOptimizer(0.01, 0.95).minimize(cost)
    #predict_op = tf.argmax(py_x, 1)
    predict_op = tf.nn.softmax(py_x)
    
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    
    batch_size = 100
    for x in range(10):
        for i in range(train_size / batch_size):
            offset = (i * batch_size) % (train_size - batch_size)
            current_batch_data = train_data[offset:(offset + batch_size)]
            batch_text_data_vector = TextVectorUtil.BuildText2DimArray(current_batch_data,tokenDict)
            batch_labels = train_labels[offset:(offset + batch_size)]
            #print batch_text_data_vector
            loss, prediction,_ = sess.run([cost,predict_op,train_op], feed_dict={X: batch_text_data_vector, 
                                                            Y: batch_labels,
                                                            p_keep_input: 1.0,
                                                            p_keep_hidden: 1.0})
            #print prediction     
            #print batch_labels
            print 'Loss, %.3f' % loss
            print 'predict %.3f' % ModelUtil.error_rate(prediction,batch_labels)
            
            if (x*(train_size / batch_size) + i) % 200 ==0 and (x*(train_size / batch_size) + i)!=0:
                print 'validation error:%.4f' % CaculateErrorRate(sess,validation_data,validation_labels)
    sess.close()       
if __name__ == "__main__":  
    #ResizeImages("../sample_data/gray_images","../sample_data/100_100",(100,100))
    #Process_OCR("../../data/gray_images")
    train()