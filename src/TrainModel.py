import os
import sys
import urllib
import tensorflow.python.platform
import numpy
import tensorflow as tf
import csv
import random
import math
from PIL import Image

from util.freeze_graph import freeze_graph

import util.DataUtil as DataUtil
import util.TextVectorUtil as TextVectorUtil
import util.ModelUtil as ModelUtil

IMAGE_SIZE = 100
NUM_CHANNELS = 1
PIXEL_DEPTH = 255

BATCH_SIZE = 300
NUM_EPOCHS = 20

class TrainModel:
    stddev_value = 0.1
    init_learn_rate = 0.01
    decay_rate = 0.95
    SEED = None
    
    model_save_dir = ''
    model_save_file_name = 'train_result'
    # Image infomation
    imageInfo={'WIDTH':100,'HEIGHT':100,'CHANNELS':1}
    
    train_csv_file = ''
    name_id_mapping_file = ''
    image_dir = ''
    
    validation_csv_file = ''
    test_csv_file = ''
    
    text_tokens_csv = ''
    tokenDict = None
     
    train_size = 0
    validation_size = 0
    test_size = 0
    tokenCount = 0
    labelCount = 0
    
    train_data= None
    train_tokens_list = None
    train_labels = None
    #DataUtil.LoadCategoryData('../data/trainning_data.csv','../'+NAME_ID_MAPPING_NAME,'../data/100_100',imageInfo)
    validation_data = None
    validation_tokens_list = None
    validation_labels = None
    # = DataUtil.LoadCategoryData('../data/validation_data.csv','../'+NAME_ID_MAPPING_NAME,'../data/100_100',imageInfo)
    test_data = None
    test_tokens_list = None
    test_labels = None
    # = DataUtil.LoadCategoryData('../data/test_data.csv','../'+NAME_ID_MAPPING_NAME,'../data/100_100',imageInfo)

    # Model parameters
    
    conv1_weights = None
    conv1_biases = None
    conv2_weights = None
    
    conv2_biases = None
    
    conv3_weights = None
    conv3_biases = None
    
    fc1_weights = None
    fc1_biases = None

    fc2_weights = None
    fc2_biases = None
    
    def InitVars(self):
        # Set parameters
        self.conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, self.imageInfo['CHANNELS'], 32],  # 5x5 filter, depth 32.
                            stddev=self.stddev_value,
                            seed=self.SEED), name='conv1_weights')
        self.conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')
        
        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 32],
                                stddev=self.stddev_value,
                                seed=self.SEED), name='conv2_weights')
        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[32]), name='conv2_biases')
        
        self.conv3_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],
                                stddev=self.stddev_value,
                                seed=self.SEED), name='conv3_weights') 
        self.conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='conv3_biases')
        
        self.fc1_weights = tf.Variable(  # fully connected, depth 1024.
            tf.truncated_normal([int(self.imageInfo['WIDTH'] / 8) * int(self.imageInfo['HEIGHT'] / 8) * 64 + self.tokenCount, 300],
                                stddev=self.stddev_value,
                                seed=self.SEED), name='fc1_weights')
        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[300]), name='fc1_biases')
        
        
        self.fc2_weights = tf.Variable(
            tf.truncated_normal([300, self.labelCount],
                                stddev=self.stddev_value,
                                seed=self.SEED), name='fc2_weights')
        self.fc2_biases = tf.Variable(tf.constant(self.stddev_value, shape=[self.labelCount]), name='fc2_biases')

    def LoadData(self):
        self.train_data,  self.train_tokens_list, self.train_labels = DataUtil.LoadCategoryData(self.train_csv_file,self.name_id_mapping_file,self.image_dir,self.imageInfo)
        self.validation_data, self.validation_tokens_list,self.validation_labels = DataUtil.LoadCategoryData(self.validation_csv_file,self.name_id_mapping_file,self.image_dir,self.imageInfo)
        self.test_data, self.test_tokens_list,self.test_labels = DataUtil.LoadCategoryData(self.test_csv_file,self.name_id_mapping_file,self.image_dir,self.imageInfo)
        self.validation_size = validation_data.shape[0]
        self.test_size = test_data.shape[0]
        self.train_size = train_data.shape[0]

        self.tokenDict = TextVectorUtil.GetAllTokenDict(self.text_tokens_csv)
    
        self.tokenCount = len(self.tokenDict)
    
        self.labelCount = self.train_labels.shape[1]
        
        # Set parameters
        self.conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, self.imageInfo['CHANNELS'], 32],  # 5x5 filter, depth 32.
                            stddev=self.stddev_value,
                            seed=self.SEED), name='conv1_weights')
        self.conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')
        
        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 32],
                                stddev=self.stddev_value,
                                seed=self.SEED), name='conv2_weights')
        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[32]), name='conv2_biases')
        
        self.conv3_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],
                                stddev=self.stddev_value,
                                seed=self.SEED), name='conv3_weights') 
        self.conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='conv3_biases')
        
        self.fc1_weights = tf.Variable(  # fully connected, depth 1024.
            tf.truncated_normal([int(self.imageInfo['WIDTH'] / 8) * int(self.imageInfo['HEIGHT'] / 8) * 64 + self.tokenCount, 300],
                                stddev=self.stddev_value,
                                seed=self.SEED), name='fc1_weights')
        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[300]), name='fc1_biases')
        
        
        self.fc2_weights = tf.Variable(
            tf.truncated_normal([300, self.labelCount],
                                stddev=self.stddev_value,
                                seed=self.SEED), name='fc2_weights')
        self.fc2_biases = tf.Variable(tf.constant(self.stddev_value, shape=[self.labelCount]), name='fc2_biases')
        
    def CreateModel(self,data,text_data, train=False):
        conv = tf.nn.conv2d(data,
                            self.conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            self.conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        print pool.get_shape().as_list()
        conv = tf.nn.conv2d(pool,
                            self.conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv3_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='VALID')
                                                            
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        print pool_shape
        print self.fc1_weights.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        #Add text vector into account before fully connected layer
        
        reshape = tf.concat(1,[reshape,text_data])
        
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden1 = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.

        if train:
            hidden1 = tf.nn.dropout(hidden1, 0.8, seed=SEED)
        return tf.matmul(hidden1, self.fc2_weights) + self.fc2_biases
    
    def TrainModel(self):
        train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, self.imageInfo['WIDTH'], self.imageInfo['HEIGHT'], self.imageInfo['CHANNELS']))
            
        train_text_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, self.tokenCount))
        
        train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, self.labelCount))
        
        validation_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, self.imageInfo['WIDTH'], self.imageInfo['HEIGHT'], self.imageInfo['CHANNELS']))
            
        validation_text_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, self.tokenCount))
            
        validation_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, self.labelCount))
        
        logits = self.CreateModel(train_data_node,train_text_node, True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                        tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
        # Add the regularization term to the loss.
        loss += 5e-8 * regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            self.init_learn_rate,# Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            self.train_size,     # Decay step.
            self.decay_rate,     # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=batch)
        # Predictions for the minibatch, validation set and test set.
        train_prediction = tf.nn.softmax(logits)
        validation_prediction = tf.nn.softmax(self.CreateModel(validation_data_node,validation_text_node))
        
        #vars to be saved
        store_list = [self.conv1_weights,self.conv1_biases,self.conv2_weights,self.conv2_biases,
                        self.conv3_weights,self.conv3_biases,self.fc1_weights,self.fc2_biases,self.fc2_weights,self.fc2_biases]

        # Create saver
        saver=tf.train.Saver(store_list);
        def CaculateErrorRate(session,dataList,tokenList,labels):
            data_size = dataList.shape[0]
            errorCount = 0;
            for step in xrange(int(data_size / BATCH_SIZE)):
                offset = (step * BATCH_SIZE)
                batch_data = dataList[offset:(offset + BATCH_SIZE), :, :, :]
                batch_text_data = tokenList[offset:(offset + BATCH_SIZE)]
                batch_text_data_vector = TextVectorUtil.BuildText2DimArray(batch_text_data,self.tokenDict)
                batch_labels = labels[offset:(offset + BATCH_SIZE)]
                feed_dict = {validation_data_node: batch_data,
                            validation_text_node: batch_text_data_vector,
                            validation_labels_node: batch_labels}
                # Run the graph and fetch some of the nodes.
                #print batch_data.shape
                #print batch_labels.shape
                #print train_labels
                validation_prediction_result = session.run(validation_prediction,feed_dict=feed_dict)
                errorCount += ModelUtil.error_count(validation_prediction_result,batch_labels)
            return  errorCount *100.0/ data_size  
        
        with tf.Session() as s:
        
            tf.initialize_all_variables().run()
            print 'Initialized!'
            # Loop through training steps.
            for step in xrange(int(NUM_EPOCHS * self.train_size / BATCH_SIZE)):
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                batch_data = self.train_data[offset:(offset + BATCH_SIZE), :, :, :]
                batch_text_data = self.train_tokens_list[offset:(offset + BATCH_SIZE)]
                batch_text_data_vector = TextVectorUtil.BuildText2DimArray(batch_text_data,self.tokenDict)
                batch_labels = self.train_labels[offset:(offset + BATCH_SIZE)]
                # This dictionary maps the batch data (as a numpy array) to the
                # node in the graph is should be fed to.
                #print batch_data.shape
                feed_dict = {train_data_node: batch_data,
                            train_text_node: batch_text_data_vector,
                            train_labels_node: batch_labels}
                # Run the graph and fetch some of the nodes.
                #print batch_data.shape
                #print batch_labels.shape
                #print train_labels
                _, l, lr, predictions = s.run(
                    [optimizer, loss, learning_rate, train_prediction],
                    feed_dict=feed_dict)

                if step % 1 == 0:
                    #print s.run(conv1_weights);
                    #print s.run(conv2_weights);
                    
                    #saver.save(s,save_path='../models/producttype/train_result')
                    
                    print 'Epoch %.2f' % (float(step) * BATCH_SIZE / self.train_size)
                    print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                    print 'Minibatch error: %.1f%%' % ModelUtil.error_rate(predictions,batch_labels)

                if step % 100 == 0 and step != 0 :    
                    saver.save(s,save_path=os.path.join(self.model_save_dir,self.model_save_file_name))                            
                    print 'Validation error: %.1f%%' % CaculateErrorRate(s,self.validation_data,self.validation_tokens_list,self.validation_labels)
                
                sys.stdout.flush()
                
            saver.save(s,save_path=os.path.join(self.model_save_dir,self.model_save_file_name))                    
            #saver.save(s,save_path='../models/producttype/train_result')
            # Finally print the result!
            test_error = CaculateErrorRate(s,self.test_data,self.test_tokens_list,self.test_labels)
            print 'Test error: %.1f%%' % test_error

    def RestoreParameters(self,session):
        #vars to be saved
        store_list = [self.conv1_weights,self.conv1_biases,self.conv2_weights,self.conv2_biases,
                        self.conv3_weights,self.conv3_biases,self.fc1_weights,self.fc2_biases,self.fc2_weights,self.fc2_biases]
        restorer = tf.train.Saver(store_list)
        restorer.restore(session,save_path=os.path.join(self.model_save_dir,self.model_save_file_name))      
        pass           

    def Predict(image_data, token_list):
        check_data_node = tf.placeholder(tf.float32, shape=(1, self.imageInfo['WIDTH'], self.imageInfo['HEIGHT'], self.imageInfo['CHANNELS']))  
        text_node = tf.placeholder(tf.float32, shape=(1, self.tokenCount))
        prediction = tf.nn.softmax(self.CreateModel(check_data_node,text_node))
        with tf.Session() as s:
            RestoreParameters(s)
            feed_dict = {check_data_node: batch_data,
                         text_node: batch_text_data_vector}
            prediction_result = s.run(prediction,feed_dict = feed_dict)
        
        