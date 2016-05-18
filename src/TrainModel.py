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

class TrainModel():
    stddev_value = 0.1
    SEED = None
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
        pass
    def RestoreParameters(self) 
        
        pass           
            