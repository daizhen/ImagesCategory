import sys
import os
sys.path.append('/home/daizhen/projects/ImagesCategory/src')
import ModelPrediction
import util.CSVUtil as CSVUtil
import util.TextVectorUtil as TextVectorUtil

def Process():
   
    result_list = [] 
    data = CSVUtil.ReadCSV('demo_data.csv')
    base_dir= '/home/daizhen/projects/ImagesCategory/'
    model = ModelPrediction.ModelPrediction()
    
    for item in data:
        model.image_path = os.path.join(base_dir,'data/100_100/'+item[0])
        model.token_list = TextVectorUtil.GetTokenList(item[5])
        
        predict_category = model.PredictCategory()
        item.append(predict_category)
        result_list.append(item)
    model.Release()
    CSVUtil.WriteCSV('demo_data.csv',result_list)
def Process_1():
   
    result_list = [] 
    data = CSVUtil.ReadCSV('demo_data.csv')
    base_dir= '/home/daizhen/projects/ImagesCategory/'
    model = ModelPrediction.ModelPrediction()
    
    for item in data:
        model.image_path = os.path.join(base_dir,'data/100_100/'+item[0])
        model.token_list = TextVectorUtil.GetTokenList(item[5])
        
        predict_subcategory = model.PredictSubCategory()
        item.append(predict_subcategory)
        result_list.append(item)
    model.Release()
    CSVUtil.WriteCSV('demo_data.csv',result_list)

def Process_2():
   
    result_list = [] 
    data = CSVUtil.ReadCSV('demo_data.csv')
    base_dir= '/home/daizhen/projects/ImagesCategory/'
    model = ModelPrediction.ModelPrediction()
    
    for item in data:
        model.image_path = os.path.join(base_dir,'data/100_100/'+item[0])
        model.token_list = TextVectorUtil.GetTokenList(item[5])
        
        predict_producttype = model.PredictProductType()
        item.append(predict_producttype)
        result_list.append(item)
    model.Release()
    CSVUtil.WriteCSV('demo_data.csv',result_list)

def Process_3():
    result_list = [] 
    data = CSVUtil.ReadCSV('demo_data.csv')
    base_dir= '/home/daizhen/projects/ImagesCategory/'
    model = ModelPrediction.ModelPrediction()
    
    for item in data:
        model.image_path = os.path.join(base_dir,'data/100_100/'+item[0])
        model.token_list = TextVectorUtil.GetTokenList(item[5])
        
        predict_product = model.PredictProduct()
        item.append(predict_product)
        result_list.append(item)
    model.Release()
    CSVUtil.WriteCSV('demo_data.csv',result_list)        
if __name__ == "__main__":
    Process_2()
    