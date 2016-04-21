import os
import sys
import urllib
#import tensorflow.python.platform
import numpy
import csv
from PIL import Image
import random
import math
import tensorflow as tf

NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 3
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 1
NUM_EPOCHS = 10


def test():
    #os.path.join()
	pass
    
def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])
        
def LoadCategoryData(imageDir):
    image_list = list()
    label_list = list()
    fullFileName = 'all_category_data.csv'
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
    
    for index in range(20):
        dataItem = data_list[index]

        image = Image.open(os.path.join(imageDir,dataItem[0]))   # image is a PIL image 
        array = numpy.array(image)        # array is a numpy array
        image_list.append(array);
        label_list.append(dataItem[1]);
    return image_list,label_list

'''
x,y = LoadCategoryData('sample_data/gray_images')
print x[0].shape
'''
def Train():
	pass

def create_labels(original_labels,possible_labels):
     return (original_labels == possible_labels[:, None]).astype(numpy.float32)
     
def main(argv=None):

    train_prop = 70
    validation_prop = 20
    test_prop = 10
    pyramid_1= 1 # pooling the image into 1x1 
    pyramid_2= 10 # pooling the image into 10x10
    pyramid_3= 50 # pooling the image into 50x50
    
    # Get the data.
    all_data, all_labels = LoadCategoryData('sample_data/gray_images')
    train_size = int(train_prop * len(all_data)/100)
    validation_size = int(validation_prop * len(all_data)/100)
    test_size = int(test_prop * len(all_data)/100)
    
    train_data = all_data[:train_size - 1]
    train_labels = all_labels[:train_size - 1]
    
    validation_data = all_data[train_size:train_size+validation_size -1]
    validation_labels = all_labels[train_size:train_size+validation_size - 1]
    
    test_data = all_data[train_size+validation_size:]
    test_labels = all_labels[train_size+validation_size:]
    
    const_pyramid_1 = tf.constant(pyramid_1,dtype=tf.float32)
    const_pyramid_2 = tf.constant(pyramid_2,dtype=tf.float32)
    const_pyramid_3 = tf.constant(pyramid_3,dtype=tf.float32)
    
    num_epochs = 10
    
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(tf.float32,shape=(1,None, None, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,shape=[NUM_LABELS])
    
    # For the validation and test data, we'll just hold the entire dataset in
    # one constant node.
    
    
    validation_data_node = tf.placeholder(tf.float32,shape=[1,None, None, NUM_CHANNELS])
    test_data_node = tf.placeholder(tf.float32,shape=[1,None, None, NUM_CHANNELS])

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
	
    image_width = tf.Variable(100.0)
    image_height = tf.Variable(100.0)
    
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([pyramid_1*pyramid_1 + pyramid_2*pyramid_2 + pyramid_3*pyramid_3, 512],
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
    def model(data, sess, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1,1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        
        pool = tf.nn.max_pool(relu,
                              ksize=[1,2, 2, 1],
                              strides=[1,2, 2, 1],
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1,1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        
        #Pyramid pooling to make the pool to the same size
        #data_width = relu.get_shape().as_list()[1]
        #data_height = relu.get_shape().as_list()[2] 
        data_width = math.ceil(image_width.eval(sess)/2)
        data_height = math.ceil(image_height.eval(sess)/2)
        
        pool_window_width_1 = math.ceil(data_width/pyramid_1)
        pool_window_height_1 = math.ceil(data_height/pyramid_1)
        
        pool_stride_width_1 = math.floor(data_width/pyramid_1)
        pool_stride_height_1 = math.floor(data_height / pyramid_1)    
        
        pool_window_width_2 = math.ceil(data_width/pyramid_2)
        pool_window_height_2 = math.ceil(data_height/pyramid_2)        
        pool_stride_width_2 = math.floor(data_width/pyramid_2)
        pool_stride_height_2 = math.floor(data_height / pyramid_2)   
        
        pool_window_width_3 = math.ceil(data_width/pyramid_3)
        pool_window_height_3 = math.ceil(data_height/pyramid_3)
        pool_stride_width_3 = math.floor(data_width/pyramid_3)
        pool_stride_height_3 = math.floor(data_height / pyramid_3)  
        
        pool_1 = tf.nn.max_pool(relu,
                              ksize=[1, pool_window_width_1, pool_window_height_1, 1],
                              strides=[1, pool_stride_width_1, pool_stride_height_1, 1],
                              padding='VALID')
                                                         
        pool_2 = tf.nn.max_pool(relu,
                              ksize=[1, pool_window_width_2, pool_window_height_2, 1],
                              strides=[1, pool_stride_width_2, pool_stride_height_2, 1],
                              padding='VALID')
        pool_3 = tf.nn.max_pool(relu,
                              ksize=[1, pool_window_width_3, pool_window_height_3, 1],
                              strides=[1, pool_stride_width_3, pool_stride_height_3, 1],
                              padding='VALID')
        pool_shape_1 =  pool_1.get_shape().as_list()
        pool_shape_2 =  pool_2.get_shape().as_list()
        pool_shape_3 =  pool_3.get_shape().as_list()
        
        reshaped_pool_1 = tf.reshape(pool_1,[pyramid_1])
        reshaped_pool_2 = tf.reshape(pool_2,[pyramid_2*pyramid_2])
        reshaped_pool_3 = tf.reshape(pool_3,[pyramid_3*pyramid_3])
        
        # Reshape the feature map cuboid into a 1D matrix to feed it to the fully connected layers.
        # result_pool is a 1-D data.
        result_pool = tf.concat(0,[reshaped_pool_1,reshaped_pool_2,reshaped_pool_3])
        result_pool = tf.reshape(result_pool,[1,pyramid_1*pyramid_1 + pyramid_2*pyramid_2 + pyramid_3*pyramid_3])

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(result_pool, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases
    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print 'Initialized!'
        # Training computation: logits + cross-entropy loss.
        logits = model(train_data_node,s, True)
        
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
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
            batch,  # Current index into the dataset.
            train_size,          # Decay step.
            0.95,                # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=batch)

        # Predictions for the minibatch, validation set and test set.
        train_prediction = tf.nn.softmax(logits)
        # We'll compute them only once in a while by calling their {eval()} method.
        validation_prediction = tf.nn.softmax(model(validation_data_node,s))
        test_prediction = tf.nn.softmax(model(test_data_node,s))

        # Loop through training steps.
        for step in xrange(int(num_epochs * train_size)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = step % train_size
            
            datashape = train_data[offset].shape
            batch_data =np.reshape(train_data[offset],(1,datashape[0],datashape[1],1))
            batch_labels = train_labels[offset]
            
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = s.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            if step % 100 == 0:
                print 'Epoch %.2f' % (float(step) * BATCH_SIZE / train_size)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                print 'Minibatch error: %.1f%%' % error_rate(predictions,
                                                             batch_labels)
                print 'Validation error: %.1f%%' % error_rate(
                    validation_prediction.eval(), validation_labels)
                sys.stdout.flush()
        # Finally print the result!
        test_error = error_rate(test_prediction.eval(), test_labels)
        print 'Test error: %.1f%%' % test_error
        '''
        if FLAGS.self_test:
            print 'test_error', test_error
            assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (test_error,)
        '''

if __name__ == '__main__':
    tf.app.run()