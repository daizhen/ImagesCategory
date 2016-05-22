import util.CSVUtil as CSVUtil
import util.ImageUtil as ImageUtil
import util.TextVectorUtil as TextVectorUtil
import TrainModel

import tensorflow as tf
import numpy as np
import os
import sys

class ModelPrediction:
    image_path=None
    category = None
    subcategory = None
    producttype= None
    product = None
    token_list = None
    
    session_category = None
    session_subcategory = None
    session_producttype =None
    session_product = None
    
    model_category = None
    model_subcategory = None
    model_producttype =None
    model_product = None
    base_dir = '/home/daizhen/projects/ImagesCategory';
    
    def Predict(self):
    
        data = CSVUtil.ReadCSV(os.path.join(self.base_dir,'data/test_data.csv'))
        
        test_item = data[47]
        self.image_path = os.path.join(self.base_dir,'data/jpg_images/'+test_item[0])
        self.token_list = TextVectorUtil.GetTokenList(test_item[5])
        
        cmd_str = 'python ModelPrediction.py category "'+self.image_path+'" '+ (' '.join(self.token_list))
     
        #return
        result = os.popen(cmd_str).readlines()
        predict_category = result[-1].strip()
        cmd_str = 'python ModelPrediction.py subcategory "'+self.image_path+'" '+ (' '.join(self.token_list))
        result = os.popen(cmd_str).readlines()
        predict_subcategory = result[-1].strip()
        
        cmd_str = 'python ModelPrediction.py producttype "'+self.image_path+'" '+ (' '.join(self.token_list))
        result = os.popen(cmd_str).readlines()
        predict_producttype = result[-1].strip()
        cmd_str = 'python ModelPrediction.py product "'+self.image_path+'" '+ (' '.join(self.token_list))
        result = os.popen(cmd_str).readlines()
        predict_product = result[-1].strip()
        print predict_category,predict_subcategory,predict_producttype,predict_product
        return predict_category,predict_subcategory,predict_producttype,predict_product
    
    def PredictCategory(self):
        if self.model_category == None:
        #if True:
            self.model_category = TrainModel.TrainModel()
            self.model_category.model_save_dir = os.path.join(self.base_dir,'trained_models/category')
            self.model_category.model_save_file_name='category_model'
            self.model_category.text_tokens_csv = os.path.join(self.base_dir,'data/all_trainning_tokens.csv')
            self.model_category.name_id_mapping_file = os.path.join(self.base_dir,'category_name_id_map.csv')
            self.model_category.InitVars()
            self.session_category = tf.Session()
            self.model_category.RestoreParameters(self.session_category)
            
        #image_file = os.path.join(base_dir,'data/jpg_images/sd21531814samm msg.jpeg')
        image = ImageUtil.PreprocessImage(self.image_path,(100,100))
        image_data = ImageUtil.ReadImageToArray(image)

        if self.token_list == None:
            raw_text =OCRUtil.ExtractText(self.image_path)
            self.token_list = TextVectorUtil.GetTokenList(raw_text)
        #image_data = array
        #self.model_category.Predict(image_data,self.token_list,self.session_category)
        return self.model_category.Predict(image_data,self.token_list,self.session_category)
    
    def PredictSubCategory(self):
        if self.model_subcategory == None:
            self.model_subcategory = TrainModel.TrainModel()
            self.model_subcategory.model_save_dir = os.path.join(self.base_dir,'trained_models/subcategory')
            self.model_subcategory.model_save_file_name='subcategory_model'
            self.model_subcategory.text_tokens_csv = os.path.join(self.base_dir,'data/all_trainning_tokens.csv')
            self.model_subcategory.name_id_mapping_file = os.path.join(self.base_dir,'subcategory_name_id_map.csv')
            self.model_subcategory.InitVars()
            self.session_subcategory = tf.Session()
            self.model_subcategory.RestoreParameters(self.session_subcategory)
        #image_file = os.path.join(base_dir,'data/jpg_images/sd21531814samm msg.jpeg')
        image = ImageUtil.PreprocessImage(self.image_path,(100,100))
        image_data = ImageUtil.ReadImageToArray(image)

        if self.token_list == None:
            raw_text =OCRUtil.ExtractText(self.image_path)
            self.token_list = TextVectorUtil.GetTokenList(raw_text)
        #image_data = array
        return self.model_subcategory.Predict(image_data,self.token_list,self.session_subcategory)
    def PredictProductType(self):
        if self.model_producttype == None:
            self.model_producttype = TrainModel.TrainModel()
            self.model_producttype.model_save_dir = os.path.join(self.base_dir,'trained_models/producttype')
            self.model_producttype.model_save_file_name='producttype_model'
            self.model_producttype.text_tokens_csv = os.path.join(self.base_dir,'data/all_trainning_tokens.csv')
            self.model_producttype.name_id_mapping_file = os.path.join(self.base_dir,'producttype_name_id_map.csv')
            self.model_producttype.InitVars()
        #image_file = os.path.join(base_dir,'data/jpg_images/sd21531814samm msg.jpeg')
        image = ImageUtil.PreprocessImage(self.image_path,(100,100))
        image_data = ImageUtil.ReadImageToArray(image)

        if self.token_list == None:
            raw_text =OCRUtil.ExtractText(self.image_path)
            self.token_list = TextVectorUtil.GetTokenList(raw_text)
        #image_data = array
        return self.model_producttype.Predict(image_data,self.token_list)
    
    def PredictProduct(self):
        if self.model_product == None:
            self.model_product = TrainModel.TrainModel()
            self.model_product.model_save_dir = os.path.join(self.base_dir,'trained_models/product')
            self.model_product.model_save_file_name='product_model'
            self.model_product.text_tokens_csv = os.path.join(self.base_dir,'data/all_trainning_tokens.csv')
            self.model_product.name_id_mapping_file = os.path.join(self.base_dir,'product_name_id_map.csv')
            self.model_product.InitVars()
        #image_file = os.path.join(base_dir,'data/jpg_images/sd21531814samm msg.jpeg')
        image = ImageUtil.PreprocessImage(self.image_path,(100,100))
        image_data = ImageUtil.ReadImageToArray(image)

        if self.token_list == None:
            raw_text =OCRUtil.ExtractText(self.image_path)
            self.token_list = TextVectorUtil.GetTokenList(raw_text)
        #image_data = array
        return self.model_product.Predict(image_data,self.token_list)
    def Release(self):
        if(self.session_category):
            self.session_category.close()
        if(self.session_subcategory):
            self.session_subcategory.close()
        if(self.session_producttype):
            self.session_producttype.close()
        if(self.session_product):
            self.session_product.close()

if __name__ == "__main__": 
    predict_type = ''
    print sys.argv
    if len(sys.argv) >1:
        predict_type = sys.argv[1]
    
    model = ModelPrediction()
    model.image_path = sys.argv[2]
    model.token_list = sys.argv[3:]
    '''
    data = CSVUtil.ReadCSV(os.path.join(model.base_dir,'data/test_data.csv'))
    
    test_item = data[0]
    model.image_path = os.path.join(model.base_dir,'data/jpg_images/'+test_item[0])
    model.category = test_item[1]
    model.subcategory = test_item[2]
    model.producttype = test_item[3]
    model.product = test_item[4]
    model.token_list = TextVectorUtil.GetTokenList(test_item[5])
    '''
    if predict_type =='category':
        result_category = model.PredictCategory();
        print result_category
    elif predict_type =='subcategory':
        result_category = model.PredictSubCategory();
        print result_category
    elif predict_type =='producttype':
        result_category = model.PredictProductType();
        print result_category
    elif predict_type =='product':
        result_category = model.PredictProduct();
        print result_category
    #result_category = model.PredictSubCategory();
    
    #result_subcategory = model.PredictSubCategory();

    #result_productype = model.PredictProductType();
    #result_product = model.PredictProduct();
    
    #print 'Original: %20s, prediction:%20s' % (model.subcategory,result_subcategory)
    #print 'Original: %20s, prediction:%20s' % (model.producttype,result_productype)
    #print 'Original: %20s, prediction:%20s' % (model.product,result_product)
