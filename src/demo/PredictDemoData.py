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

def ProcessResult():
     data = CSVUtil.ReadCSV('demo_data.csv')
     category_acc =0.0
     subcategory_acc =0.0
     producttype_acc =0.0
     productt_acc =0.0
     for item in data:
        if item[1] == item[-4]:
            category_acc+=1
            
        if item[2] == item[-3]:
            subcategory_acc+=1
            
        if item[3] == item[-2]:
            producttype_acc+=1
            
        if item[4] == item[-1]:
            productt_acc+=1
     category_acc = category_acc*100.0/len(data)
     subcategory_acc = subcategory_acc*100.0/len(data)
     producttype_acc = producttype_acc*100.0/len(data)
     productt_acc = productt_acc*100.0/len(data)       
     print category_acc,subcategory_acc,producttype_acc,productt_acc
     '''
     79.71 90.54 75.41 53.49
     '''
if __name__ == "__main__":
    ProcessResult()
    