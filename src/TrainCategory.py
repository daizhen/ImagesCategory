import TrainModel
import os

def train():                                        
    base_dir = '/home/daizhen/projects/ImagesCategory';
    model = TrainModel.TrainModel()
    model.model_save_dir = os.path.join(base_dir,'trained_models/category')                         
    model.model_save_file_name='category_model'
    model.train_csv_file = os.path.join(base_dir,'data/trainning_data.csv')
    model.name_id_mapping_file = os.path.join(base_dir,'category_name_id_map.csv')
    model.image_dir =  os.                                                      path.join(base_dir,'data/100_100') 
    
    model.validation_csv_file = os.path.join(base_dir,'data/validation_data.csv')
    model.test_csv_file = os.path.join(base_dir,'data/test_data.csv')
    model.text_tokens_csv = os.path.join(base_dir,'data/all_trainning_tokens.csv')
    model.LoadData()
    model.TrainModel()
if __name__ == "__main__":  
    train()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         