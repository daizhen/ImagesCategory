import CSVUtil
import sys
import numpy as np
from PIL import Image
import TextVectorUtil
import re
import os

def LoadAllLabels(fileName):
    name_id_mappings = CSVUtil.ReadCSV(fileName)
    name_list = [item[0] for item in name_id_mappings]
    return np.array(name_list)

'''
General function to load data
    csvFilePath: csv file contains: image name, category, subcatetory, product type, product and text tokens
    mappingFilePath: csv file to contain mappint between label name and label id
    imageDir: Dir stores the images
    imageInfo: width, height, channels
RETURN:
    image_list: images np.ndarray(shape=(image_count, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    text tokens: np.array(), and for each item it's a list of tokens 
    label_list: np.ndarray(shape=(image_count, NUM_LABELS), dtype=np.float32)
    
'''    
def _loadData(csvFilePath,imageDir,imageInfo):
    plainData = CSVUtil.ReadCSV(csvFilePath)
    #labels = LoadAllLabels(mappingFilePath)
    data_count = len(plainData)
    
    image_list = np.ndarray( 
        shape=(data_count, imageInfo['WIDTH'], imageInfo['HEIGHT'], imageInfo['CHANNELS']),
        dtype=np.float32)
    tokens_list = []
    '''
    label_list = np.ndarray(shape=[image_count], dtype=np.float32)
    label_list_result = np.ndarray(shape=(data_count, NUM_LABELS), dtype=np.float32)
    '''
    regEx = re.compile('\\W*')
    for index in range(data_count):
        dataItem = plainData[index]
        image = Image.open(os.path.join(imageDir,dataItem[0]))   # image is a PIL image 
        array = np.array(image)        # array is a numpy array
        image_list[index,:,:,:] = np.reshape(array,(imageInfo['WIDTH'], imageInfo['HEIGHT'], imageInfo['CHANNELS']))
        currentWordList = regEx.split(dataItem[-1])
        tokens_list.append(currentWordList)

    image_list = (image_list - (255 / 2.0)) / 255        
    return image_list, np.array(tokens_list)

def LoadDataByType(csvFilePath,mappingFilePath,imageDir,imageInfo,dataType):
    columnIndex = 1
    if dataType == 'Category':
        columnIndex = 1
    elif dataType == 'Subcategory':
        columnIndex = 2
    elif dataType == 'ProductType':
        columnIndex = 3
    elif dataType == 'Product':
        columnIndex = 4
    else:
        print "Not valid data type:",dataType
          
    plainData = CSVUtil.ReadCSV(csvFilePath)
    image_list, tokens_list = _loadData(csvFilePath,imageDir,imageInfo)
    
    data_count = image_list.shape[0]
    all_classes = LoadAllLabels(mappingFilePath)
    label_list = np.ndarray(shape=[data_count], dtype=np.float32)
    label_list_result = np.ndarray(shape=(data_count, len(all_classes)), dtype=np.float32)
    
    for index in range(data_count):
        dataItem = plainData[index]
        try:
            label_list[index] = np.where(all_classes == dataItem[columnIndex])[0][0]
        except:
            print "error:",  dataItem 
            
    label_list_result = (np.arange(len(all_classes)) == label_list[:, None]).astype(np.float32)
    return image_list, tokens_list,label_list_result
    
def LoadCategoryData(csvFilePath,mappingFilePath,imageDir,imageInfo):
    return LoadDataByType(csvFilePath,mappingFilePath,imageDir,imageInfo,'Category')
def LoadSubcategoryData(csvFilePath,mappingFilePath,imageDir,imageInfo):
    return LoadDataByType(csvFilePath,mappingFilePath,imageDir,imageInfo,'Subcategory')     
def LoadProductTypeData(csvFilePath,mappingFilePath,imageDir,imageInfo):
    return LoadDataByType(csvFilePath,mappingFilePath,imageDir,imageInfo,'ProductType')  
def LoadProductData(csvFilePath,mappingFilePath,imageDir,imageInfo):
    return LoadDataByType(csvFilePath,mappingFilePath,imageDir,imageInfo,'Product')  

if __name__ == "__main__":
    imageInfo={}
    imageInfo['WIDTH'] = 100
    imageInfo['HEIGHT'] = 100
    imageInfo['CHANNELS'] = 1
    
    image_list, tokens_list,label_list_result = LoadDataByType('../../data/trainning_data.csv','../../category_name_id_map.csv','../../data/100_100',imageInfo,'Category')
    print image_list.shape
    print image_list