import os
import sys
import urllib
#import tensorflow.python.platform
import numpy
#import tensorflow as tf
import csv

def test():
    #os.path.join()
	pass

def LoadCategoryData():
	image_list = list()
	label_list = list()
	fullFileName = os.path.join('data','all_category_data.csv')
	csvfile = file(fullFileName, 'rb')
	reader = csv.reader(csvfile)
	index = 0
	
	data_list = list()
	for line in reader:
		if index != 0 :
			data_list.append(line);
			print line[0]
		index +=1
	csvfile.close()
	
	label_list=data_list[,1]
	for dataItem in data_list:
		image = Image.open(dataItem[0])   # image is a PIL image 
		array = numpy.array(image)        # array is a numpy array
		image_list.append(array);
	return image_list,label_list
	
def Train():
	pass