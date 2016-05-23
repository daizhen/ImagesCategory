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
        result_list.append(item.append(predict_category))
    model.Release()
    CSVUtil.WriteCSV(result_list,'demo_data_result.csv')
if __name__ == "__main__":
    Process()