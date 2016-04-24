"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.8%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to exectute a short self-test.
"""
import gzip
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

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 100
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 3
VALIDATION_SIZE = 200  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 60
NUM_EPOCHS = 10


tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

def LoadPossibleLabels():
    fileName='subcategory_name_id_map.csv'
    csvfile = file(fileName, 'rb')
    reader = csv.reader(csvfile)
    index = 0
	
    label_list = list()
    for line in reader:
        if index != 0:
            currentLabel = line[0]
            if not currentLabel in label_list:
                
                label_list.append(currentLabel);
        index +=1
    csvfile.close()
    return numpy.array(label_list)

def LoadData(imageDir):
    
    all_classes = LoadPossibleLabels()
    fullFileName = 'all_subcategory_data.csv'
    csvfile = file(fullFileName, 'rb')
    reader = csv.reader(csvfile)
    index = 0
	
    data_list = list()
    for line in reader:
        if index != 0:
            fullImageFile = os.path.join(imageDir,line[0])
            if(os.path.exists(fullImageFile)):
                data_list.append(line);
                #print line[0]
        index +=1
    csvfile.close()
	
	#label_list=data_list[,1]
    
    '''shuffle the list '''
    random.shuffle(data_list)
    print len(data_list)
    
    image_count =20000
    #len(data_list)
    image_list = numpy.ndarray( 
        shape=(image_count, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        dtype=numpy.float32)
        
    label_list = numpy.ndarray(shape=[image_count], dtype=numpy.float32)
    label_list_result = numpy.ndarray(shape=(image_count, NUM_LABELS), dtype=numpy.float32)
    
    for index in range(image_count):
        dataItem = data_list[index]

        image = Image.open(os.path.join(imageDir,dataItem[0]))   # image is a PIL image 
        array = numpy.array(image)        # array is a numpy array
        image_list[index,:,:,:] = numpy.reshape(array,(IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS));
        
        try:
        
            label_list[index] = numpy.where(all_classes == dataItem[1])[0][0]
        except:
            print "error:",  dataItem
    print 'here'
    label_list_result = (numpy.arange(NUM_LABELS) == label_list[:, None]).astype(numpy.float32)
    image_list = (image_list - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    #print image_list
    #print label_list_result
    return image_list,label_list_result
    
def fake_data(num_images):
    """Generate a fake dataset that matches the dimensions of MNIST."""
    data = numpy.ndarray(
        shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        dtype=numpy.float32)
    labels = numpy.zeros(shape=(num_images, NUM_LABELS), dtype=numpy.float32)
    for image in xrange(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image, label] = 1.0
    return data, labels


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])


def main(argv=None):  # pylint: disable=unused-argument
    
    # 70 persent for Train
    train_prop = 70
    # 20 persent for validation
    validation_prop = 20
    # 10 persent for test
    test_prop = 10
    
    if FLAGS.self_test:
        print 'Running self-test.'
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(16)
        test_data, test_labels = fake_data(256)
        num_epochs = 1
    else:
        all_data, all_labels = LoadData("data/100_100")
        '''
        train_size = int(train_prop * len(all_data)/100)
        validation_size = int(validation_prop * len(all_data)/100)
        test_size = int(test_prop * len(all_data)/100)
        '''
        
        train_size = len(all_data) - 1000
        
        validation_size = 500
        test_size = 500
        
        
        # Extract it into numpy arrays.
        train_data = all_data[:train_size - 1,:,:,:]

        train_labels = all_labels[:train_size - 1]
        
        validation_data = all_data[train_size:train_size+validation_size -1,:,:,:]
        validation_labels = all_labels[train_size:train_size+validation_size -1]
        
        test_data = all_data[train_size:train_size+validation_size,:,:,:]
        test_labels = all_labels[train_size:train_size+validation_size]
        
        print "train_labels",train_labels.shape
        num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    # For the validation and test data, we'll just hold the entire dataset in
    # one constant node.
    validation_data_node = tf.constant(validation_data)
    test_data_node = tf.constant(test_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
	
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
	
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE / 4 * IMAGE_SIZE / 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
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
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
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
    validation_prediction = tf.nn.softmax(model(validation_data_node))
    test_prediction = tf.nn.softmax(model(test_data_node))

    # Create a local session to run this computation.
    saver=tf.train.Saver();
    #Save the graph model
    tf.train.export_meta_graph(filename='./models/subcategory/graph.save', as_text=True)
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print 'Initialized!'
        # Loop through training steps.
        for step in xrange(int(num_epochs * train_size / BATCH_SIZE)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            #print batch_data.shape
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            #print batch_data.shape
            #print batch_labels.shape
            #print train_labels
            _, l, lr, predictions = s.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            if step % 100 == 0:
                saver.save(s,save_path='./models/subcategory/train_result')
                print 'Epoch %.2f' % (float(step) * BATCH_SIZE / train_size)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                print 'Minibatch error: %.1f%%' % error_rate(predictions,
                                                             batch_labels)
                print 'Validation error: %.1f%%' % error_rate(
                    validation_prediction.eval(), validation_labels)
                sys.stdout.flush()
        saver.save(s,save_path='./models/subcategory/train_result')
        # Finally print the result!
        test_error = error_rate(test_prediction.eval(), test_labels)
        print 'Test error: %.1f%%' % test_error
        if FLAGS.self_test:
            print 'test_error', test_error
            assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
                test_error,)


if __name__ == '__main__':
    tf.app.run()
    #LoadCategoryData("sample_data/100_100")
