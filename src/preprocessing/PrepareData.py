'''
Prepare trainning, validation and test data set 

'''

import sys
import numpy as np
sys.path.append('../util/')
import CSVUtil;
import TextVectorUtil
import random
import os

def PrepareDataSet():
    
    # 70 persent for Train
    train_prop = 70
    # 20 persent for validation
    validation_prop = 20
    # 10 persent for test
    test_prop = 10
    
    
    imageTextList = CSVUtil.ReadCSV('./ocr_result.csv')
    imageTokensDict = {}
    for item in imageTextList:
        imageTokensDict[item[0]] = TextVectorUtil.GetTokenList(item[1])
    print len(imageTokensDict)
    
    #Image category
    imageCategoryDict = {}
    imageCategoryList = CSVUtil.ReadCSV('../../all_category_data.csv')   
    for item in imageCategoryList:
        imageCategoryDict[item[0]] = item[1]
    
    #Image sub category
    imageSubcategoryDict = {}
    imageSubcategoryList = CSVUtil.ReadCSV('../../all_subcategory_data.csv')   
    for item in imageSubcategoryList:
        imageSubcategoryDict[item[0]] = item[1]
    #Image product type    
    imageProductTypeDict = {}
    productTypeList = CSVUtil.ReadCSV('../../all_producttype_data.csv')    
    for item in productTypeList:
        imageProductTypeDict[item[0]] = item[1]
    #print len(imageProductTypeDict)
    
    #Image product     
    imageProductDict = {}
    productList = CSVUtil.ReadCSV('../../all_product_data.csv')    
    for item in productList:
        imageProductDict[item[0]] = item[1]
    
    
    resultList = []
    demo_data_list = []
    for imageName in imageTokensDict.keys():
        if imageName in imageCategoryDict and imageName in imageSubcategoryDict and imageName in imageProductTypeDict and imageName in imageProductDict and os.path.exists(os.path.join('../../data/100_100/',imageName)):
            resultList.append((imageName,imageCategoryDict[imageName],imageSubcategoryDict[imageName],imageProductTypeDict[imageName],imageProductDict[imageName],' '.join(imageTokensDict[imageName])))
        else:
            #print imageName
            pass
        if imageName in imageCategoryDict and imageName in imageSubcategoryDict and imageName in imageProductTypeDict and imageName in imageProductDict and os.path.exists(os.path.join('../../data/gray_images/',imageName)):
            demo_data_list.append((imageName,imageCategoryDict[imageName],imageSubcategoryDict[imageName],imageProductTypeDict[imageName],imageProductDict[imageName],' '.join(imageTokensDict[imageName])))

    print len(resultList)
    
    random.shuffle(resultList)
    
    train_size = int(train_prop * len(resultList)/100)
    validation_size = int(validation_prop * len(resultList)/100)
    test_size = int(test_prop * len(resultList)/100)
    
    train_data = resultList[:train_size]
    validation_data = resultList[train_size:train_size+validation_size]
    test_data = resultList[train_size+validation_size:train_size+validation_size + test_size]
    
    '''
    CSVUtil.WriteCSV('../../data/trainning_data.csv',train_data)
    CSVUtil.WriteCSV('../../data/validation_data.csv',validation_data)
    CSVUtil.WriteCSV('../../data/test_data.csv',test_data)
    '''
    CSVUtil.WriteCSV('../../data/demo_data.csv',demo_data_list)
if __name__ == "__main__":
    PrepareDataSet()
    