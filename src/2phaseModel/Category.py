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


sys.path.append('../util/')
from freeze_graph import freeze_graph
import DataUtil as DataUtil
import TextVectorUtil as TextVectorUtil
import ModelUtil as ModelUtil

IMAGE_SIZE = 100
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 300
NUM_EPOCHS = 20

NUM_LABELS = 3

NAME_ID_MAPPING_NAME = 'category_name_id_map.csv'
MODEL_FOLDER = '../models/category/'

'''
tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS
'''

imageInfo={'WIDTH':100,'HEIGHT':100,'CHANNELS':1}

tokenDict = TextVectorUtil.GetAllTokenDict('../../data/all_trainning_tokens.csv')    
tokenCount = len(tokenDict)

# Variable def for 1st phase training and will be used for 2nd phase
conv1_weights = tf.Variable(
    tf.truncated_normal([5, 5, imageInfo['CHANNELS'], 32],  stddev=0.01,seed=SEED), 
    name='conv1_weights')
conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')
	
conv2_weights = tf.Variable(
    tf.truncated_normal([5, 5, 32, 64],stddev=0.01, seed=SEED), 
    name='conv2_weights')
conv2_biases = tf.Variable(tf.constant(0.01, shape=[64]), name='conv2_biases')
'''    
conv3_weights = tf.Variable(
    tf.truncated_normal([5, 5, 64, 128],stddev=0.01,seed=SEED), 
    name='conv3_weights') 
conv3_biases = tf.Variable(tf.constant(0.1, shape=[128]), name='conv3_biases')
'''
fc_weights = tf.Variable(  # fully connected, depth 1024.
    tf.truncated_normal([int(imageInfo['WIDTH'] / 4) * int(imageInfo['HEIGHT'] / 4) * 64 , NUM_LABELS],stddev=0.1,seed=SEED), name='fc_weights')
fc_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='fc_biases')

batch_1 = tf.Variable(0, name='batch_1')

store_list = [conv1_weights,conv1_biases,conv2_weights,conv2_biases,fc_weights,fc_biases,batch_1]

'''
Following defined the variables for 2nd-phase 
'''

fc1_weights = tf.Variable(
    f.truncated_normal([int(imageInfo['WIDTH'] / 4) * int(imageInfo['HEIGHT'] / 4) * 64 + tokenCount, 800],
                            stddev=0.1,
                            seed=SEED), name='fc1_weights')
fc1_biases = tf.Variable(tf.constant(0.1, shape=[800]), name='fc1_biases')

fc2_weights = tf.Variable(
    tf.truncated_normal([800, 800],stddev=0.1,seed=SEED), name='fc2_weights')
fc2_biases = tf.Variable(tf.constant(0.1, shape=[800]), name='fc2_biases')
    
fc3_weights = tf.Variable(
    tf.truncated_normal([800, NUM_LABELS],stddev=0.1,seed=SEED), name='fc3_weights')
fc3_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='fc3_biases')

def model_1(data):
    conv = tf.nn.conv2d(data,conv1_weights,strides=[1, 1, 1, 1],padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    
    conv = tf.nn.conv2d(pool,conv2_weights,strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    
    '''
    print pool.get_shape().as_list()
    conv = tf.nn.conv2d(pool,conv3_weights,strides=[1, 1, 1, 1],padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
    '''
                                                            
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    print pool_shape
    print fc_weights.get_shape().as_list()
    reshape = tf.reshape(pool,[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        
    return tf.matmul(reshape, fc_weights) + fc_biases

def model_2(data, isTrain = False):
    hidden1 = tf.nn.relu(tf.matmul(data, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.

    if train:
        hidden1 = tf.nn.dropout(hidden1, 0.5, seed=SEED)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, fc2_weights) + fc2_biases)
    
    if train:
        hidden2 = tf.nn.dropout(hidden2, 0.5, seed=SEED)
        
    return tf.matmul(hidden2, fc3_weights) + fc3_biases
    
def trainModel_1():
    train_data, train_tokens_list,train_labels = DataUtil.LoadCategoryData('../../data/trainning_data.csv','../../'+NAME_ID_MAPPING_NAME,'../../data/100_100',imageInfo)
    validation_data, validation_tokens_list,validation_labels = DataUtil.LoadCategoryData('../../data/validation_data.csv','../../'+NAME_ID_MAPPING_NAME,'../../data/100_100',imageInfo)
    test_data, test_tokens_list,test_labels = DataUtil.LoadCategoryData('../../data/test_data.csv','../../'+NAME_ID_MAPPING_NAME,'../../data/100_100',imageInfo)
    
    validation_size = validation_data.shape[0]
    test_size = test_data.shape[0]
    train_size = train_data.shape[0]
    
    labelCount = train_labels.shape[1]
    
    train_data_node = tf.placeholder(tf.float32, shape=[BATCH_SIZE, imageInfo['WIDTH'], imageInfo['HEIGHT'], imageInfo['CHANNELS']])
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, labelCount))

    validation_data_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, imageInfo['WIDTH'], imageInfo['HEIGHT'], imageInfo['CHANNELS']))
    validation_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, labelCount))
    
    logits = model_1(train_data_node)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = tf.nn.l2_loss(fc_weights) 
    # Add the regularization term to the loss.
    loss += 5e-8 * regularizers
    
        # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch_1 * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=batch_1)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    validation_prediction = tf.nn.softmax(model_1(validation_data_node))
    
    
    def CaculateErrorRate(session,dataList,labels):
        data_size = dataList.shape[0]
        errorCount = 0;
        for step in xrange(int(data_size / BATCH_SIZE)):
            offset = (step * BATCH_SIZE)
            batch_data = dataList[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {validation_data_node: batch_data,
                         validation_labels_node: batch_labels}
            validation_prediction_result = session.run(validation_prediction,feed_dict=feed_dict)
            errorCount += ModelUtil.error_count(validation_prediction_result,batch_labels)
        return  errorCount *100.0/ data_size     
    saver=tf.train.Saver(store_list);
    with tf.Session() as s:        
        tf.initialize_all_variables().run()
        saver.save(s,save_path='./train_result')
        for step in xrange(int(NUM_EPOCHS * train_size / BATCH_SIZE)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            #print batch_data.shape
            feed_dict = {train_data_node: batch_data,train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = s.run([optimizer, loss, learning_rate, train_prediction],feed_dict=feed_dict)
            saver.save(s,save_path='./train_result')
            if step % 1 == 0:
                print 'Epoch %.2f' % (float(step) * BATCH_SIZE / train_size)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                print 'Minibatch error: %.1f%%' % ModelUtil.error_rate(predictions,batch_labels)
                sys.stdout.flush()
            if step % 100 == 0 and step != 0 :                                
                print 'Validation error: %.1f%%' % CaculateErrorRate(s,validation_data,validation_labels)
                sys.stdout.flush()
                    
        # Finally print the result!
        test_error = CaculateErrorRate(s,test_data,test_labels)
        print 'Test error: %.1f%%' % test_error

def trainModel_2():
    train_data, train_tokens_list,train_labels = DataUtil.LoadCategoryData('../../data/trainning_data.csv','../../'+NAME_ID_MAPPING_NAME,'../../data/100_100',imageInfo)
    validation_data, validation_tokens_list,validation_labels = DataUtil.LoadCategoryData('../../data/validation_data.csv','../../'+NAME_ID_MAPPING_NAME,'../../data/100_100',imageInfo)
    test_data, test_tokens_list,test_labels = DataUtil.LoadCategoryData('../../data/test_data.csv','../../'+NAME_ID_MAPPING_NAME,'../../data/100_100',imageInfo)
    
    validation_size = validation_data.shape[0]
    test_size = test_data.shape[0]
    train_size = train_data.shape[0]
    labelCount = train_labels.shape[1]
    
    train_data_node = tf.placeholder(tf.float32, shape=[BATCH_SIZE, imageInfo['WIDTH'], imageInfo['HEIGHT'], imageInfo['CHANNELS']])
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, labelCount))

    validation_data_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, imageInfo['WIDTH'], imageInfo['HEIGHT'], imageInfo['CHANNELS']))
    validation_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, labelCount))
    
    logits = model_1(train_data_node)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = tf.nn.l2_loss(fc_weights) 
    # Add the regularization term to the loss.
    loss += 5e-8 * regularizers
    
        # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch_1 * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=batch_1)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    validation_prediction = tf.nn.softmax(model_1(validation_data_node))
    
    
    def CaculateErrorRate(session,dataList,labels):
        data_size = dataList.shape[0]
        errorCount = 0;
        for step in xrange(int(data_size / BATCH_SIZE)):
            offset = (step * BATCH_SIZE)
            batch_data = dataList[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {validation_data_node: batch_data,
                         validation_labels_node: batch_labels}
            validation_prediction_result = session.run(validation_prediction,feed_dict=feed_dict)
            errorCount += ModelUtil.error_count(validation_prediction_result,batch_labels)
        return  errorCount *100.0/ data_size     
    saver=tf.train.Saver(store_list);
    with tf.Session() as s:        
        tf.initialize_all_variables().run()
        saver.save(s,save_path='./train_result')
        for step in xrange(int(NUM_EPOCHS * train_size / BATCH_SIZE)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            #print batch_data.shape
            feed_dict = {train_data_node: batch_data,train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = s.run([optimizer, loss, learning_rate, train_prediction],feed_dict=feed_dict)
            saver.save(s,save_path='./train_result')
            if step % 1 == 0:
                print 'Epoch %.2f' % (float(step) * BATCH_SIZE / train_size)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                print 'Minibatch error: %.1f%%' % ModelUtil.error_rate(predictions,batch_labels)
                sys.stdout.flush()
            if step % 100 == 0 and step != 0 :                                
                print 'Validation error: %.1f%%' % CaculateErrorRate(s,validation_data,validation_labels)
                sys.stdout.flush()
                    
        # Finally print the result!
        test_error = CaculateErrorRate(s,test_data,test_labels)
        print 'Test error: %.1f%%' % test_error
def main_1(argv=None):  # pylint: disable=unused-argument
    
    
    train_data, train_tokens_list,train_labels = DataUtil.LoadCategoryData('../data/trainning_data.csv','../'+NAME_ID_MAPPING_NAME,'../data/100_100',imageInfo)
    validation_data, validation_tokens_list,validation_labels = DataUtil.LoadCategoryData('../data/validation_data.csv','../'+NAME_ID_MAPPING_NAME,'../data/100_100',imageInfo)
    test_data, test_tokens_list,test_labels = DataUtil.LoadCategoryData('../data/test_data.csv','../'+NAME_ID_MAPPING_NAME,'../data/100_100',imageInfo)

    print train_labels
    validation_size = validation_data.shape[0]
    test_size = test_data.shape[0]
    train_size = train_data.shape[0]

    print "train_labels",train_labels.shape
    
    tokenDict = TextVectorUtil.GetAllTokenDict('../data/all_trainning_tokens.csv')
    
    tokenCount = len(tokenDict)
    
    labelCount = train_labels.shape[1]
    
    num_epochs = NUM_EPOCHS
   
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(tf.float32, shape=[None, imageInfo['WIDTH'], imageInfo['HEIGHT'], imageInfo['CHANNELS']])
    
    train_text_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, tokenCount))
    
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, labelCount))

    validation_data_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, imageInfo['WIDTH'], imageInfo['HEIGHT'], imageInfo['CHANNELS']))
    validation_text_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, tokenCount))
        
    validation_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, labelCount))
    
    test_data_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, imageInfo['WIDTH'], imageInfo['HEIGHT'], imageInfo['CHANNELS']))
    test_text_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, tokenCount))
        
    check_data_node = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='check_data_node')
    check_text_node = tf.placeholder(tf.float32,shape=(1, tokenCount))
    
    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, imageInfo['CHANNELS'], 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED), name='conv1_weights')
    conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')
	
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED), name='conv2_weights')
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='conv2_biases')
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='conv2_biases')   
    
    conv3_weights = tf.Variable(
        tf.truncated_normal([5, 5, 64, 128],
                            stddev=0.1,
                            seed=SEED), name='conv3_weights') 
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[128]), name='conv3_biases')
    
    fc1_weights = tf.Variable(  # fully connected, depth 1024.
        tf.truncated_normal([int(imageInfo['WIDTH'] / 8) * int(imageInfo['HEIGHT'] / 8) * 128 + tokenCount, 800],
                            stddev=0.1,
                            seed=SEED), name='fc1_weights')
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[800]), name='fc1_biases')
    fc2_weights = tf.Variable(
        tf.truncated_normal([800, 800],
                            stddev=0.1,
                            seed=SEED), name='fc2_weights')
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[800]), name='fc2_biases')
    
    fc3_weights = tf.Variable(
        tf.truncated_normal([800, labelCount],
                            stddev=0.1,
                            seed=SEED), name='fc3_weights')
    fc3_biases = tf.Variable(tf.constant(0.1, shape=[labelCount]), name='fc3_biases')
    
    # Var list to save
    #varlist = [conv1_weights,conv1_biases,conv2_weights,conv2_biases,fc1_weights,fc1_biases,fc2_weights,fc2_biases]

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data,text_data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        print pool.get_shape().as_list()
        conv = tf.nn.conv2d(pool,
                            conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='VALID')
                                                            
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        print pool_shape
        print fc1_weights.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        #Add text vector into account before fully connected layer
        
        reshape = tf.concat(1,[reshape,text_data])
        
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        '''
        if train:
            hidden1 = tf.nn.dropout(hidden1, 0.5, seed=SEED)
        '''
        hidden2 = tf.nn.relu(tf.matmul(hidden1, fc2_weights) + fc2_biases)
        '''
        if train:
            hidden2 = tf.nn.dropout(hidden2, 0.5, seed=SEED)
        '''
        return tf.matmul(hidden2, fc3_weights) + fc3_biases      
    def FreezeGraph(sess):
        checkpoint_prefix = os.path.join(MODEL_FOLDER, "saved_checkpoint")
        checkpoint_state_name = "checkpoint_state"
        input_graph_name = "input_graph.pb"
        output_graph_name = "output_graph.pb"

        # We'll create an input graph that has a single variable containing 1.0,
        # and that then multiplies it by 2.
        saver = tf.train.Saver()
        saver.save(sess, checkpoint_prefix, global_step=0,
                        latest_filename=checkpoint_state_name)
        tf.train.write_graph(sess.graph.as_graph_def(), MODEL_FOLDER,input_graph_name)

        # We save out the graph to disk, and then call the const conversion
        # routine.
        input_graph_path = os.path.join(MODEL_FOLDER, input_graph_name)
        input_saver_def_path = ""
        input_binary = False
        input_checkpoint_path = checkpoint_prefix + "-0"
        output_node_names = "check_data_node,check_prediction"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_graph_path = os.path.join(MODEL_FOLDER, output_graph_name)
        clear_devices = False

        freeze_graph(input_graph_path, input_saver_def_path,
                                input_binary, input_checkpoint_path,
                                output_node_names, restore_op_name,
                                filename_tensor_name, output_graph_path,
                                clear_devices)
    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node,train_text_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    #loss += 5e-8 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.003,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    validation_prediction = tf.nn.softmax(model(validation_data_node,validation_text_node))
    test_prediction = tf.nn.softmax(model(test_data_node,test_text_node))
    
    check_prediction = tf.nn.softmax(model(check_data_node,check_text_node), name="check_prediction")
    # Create a local session to run this computation.
    saver=tf.train.Saver();
    #Save the graph model
    #tf.train.export_meta_graph(filename='./models/producttype/graph.save', as_text=True)
    with tf.Session() as s:
    
        ckpt = tf.train.get_checkpoint_state(os.path.join(MODEL_FOLDER,'with_text'))
        tf.initialize_all_variables().run()
        if ckpt and ckpt.model_checkpoint_path:
            print "find the checkpoing file"
            saver.restore(s, ckpt.model_checkpoint_path)
        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()
        #Save the graph model
        tf.train.write_graph(s.graph_def, '', os.path.join(MODEL_FOLDER,'with_text/graph.pb'), as_text=False)

        print 'Initialized!'
        # Loop through training steps.
        for step in xrange(int(num_epochs * train_size / BATCH_SIZE)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
            batch_text_data = train_tokens_list[offset:(offset + BATCH_SIZE)]
            batch_text_data_vector = TextVectorUtil.BuildText2DimArray(batch_text_data,tokenDict)
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
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
                
                print 'Epoch %.2f' % (float(step) * BATCH_SIZE / train_size)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                print 'Minibatch error: %.1f%%' % ModelUtil.error_rate(predictions,batch_labels)
            if step % 100 == 0 and step != 0 :                                
                print 'Validation error: %.1f%%' % CaculateErrorRate(s,validation_data,validation_tokens_list,validation_labels)
                sys.stdout.flush()
                
        FreezeGraph(s)
        #saver.save(s,save_path='../models/producttype/train_result')
        # Finally print the result!
        test_error = CaculateErrorRate(s,test_data,test_tokens_list,test_labels)
        print 'Test error: %.1f%%' % test_error
        if FLAGS.self_test:
            print 'test_error', test_error
            assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (test_error,)


if __name__ == '__main__':
    trainModel_1()
    #LoadCategoryData("sample_data/100_100")