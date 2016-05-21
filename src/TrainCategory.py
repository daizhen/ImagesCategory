import TrainModel
import os
from PIL import Image
import util.TextVectorUtil as TextVectorUtil
import util.ImageUtil as ImageUtil
import util.OCRUtil as OCRUtil
import numpy as np

def train():                                        
    base_dir = '/home/daizhen/projects/ImagesCategory';
    model = TrainModel.TrainModel()
    model.model_save_dir = os.path.join(base_dir,'trained_models/category')                         
    model.model_save_file_name='category_model'
    model.train_csv_file = os.path.join(base_dir,'data/trainning_data.csv')
    model.name_id_mapping_file = os.path.join(base_dir,'category_name_id_map.csv')
    model.image_dir =  os.path.join(base_dir,'data/100_100') 
    
    model.validation_csv_file = os.path.join(base_dir,'data/validation_data.csv')
    model.test_csv_file = os.path.join(base_dir,'data/test_data.csv')
    model.text_tokens_csv = os.path.join(base_dir,'data/all_trainning_tokens.csv')
    model.LoadData()
    model.TrainModel()

def Test():
    base_dir = '/home/daizhen/projects/ImagesCategory';
    model = TrainModel.TrainModel()
    model.model_save_dir = os.path.join(base_dir,'trained_models/category')                         
    model.model_save_file_name='category_model'
    model.text_tokens_csv = os.path.join(base_dir,'data/all_trainning_tokens.csv')
    model.name_id_mapping_file = os.path.join(base_dir,'category_name_id_map.csv')
    
    model.InitVars()
    image_file = os.path.join(base_dir,'data/jpg_images/sd21531814samm msg.jpeg')
    image = ImageUtil.PreprocessImage(image_file,(100,100))
    image_data = ImageUtil.ReadImageToArray(image)
    print image_data
    #image = Image.open(image_file)   # image is a PIL image 
    #array = np.array(image)
    raw_text =OCRUtil.ExtractText(image_file)
    
    token_list = TextVectorUtil.GetTokenList(raw_text)
    
    'request for service,application,user access issue,user,control pleas own sod creat mgrs approv obtain restrict via are sap compani your previous use develop for usertool support feedback due click their copyright includ statement privaci main email userld then return back mainten modif hewlett not amp ticket modifi packard term hpsc ccount button page permit the manag'
    
    print token_list
    #image_data = array
    model.Predict(image_data,token_list)
      
if __name__ == "__main__":  
    train()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         