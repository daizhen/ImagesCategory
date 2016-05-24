from django.http import HttpResponse
from django.shortcuts import render_to_response 
from django.template.context import RequestContext
from PIL import Image
import sys
sys.path.append('/home/daizhen/projects/ImagesCategory/src')
import util.CSVUtil as CSVUtil
import ModelPrediction

demo_set = CSVUtil.ReadCSV('/home/daizhen/projects/ImagesCategory/src/demo/demo_data_result.csv')

def predict(request):
    if request.FILES and 'imagefile' in  request.FILES:
        reqfile = request.FILES['imagefile']
        #return render_to_response('TestPage.html', context_instance=RequestContext(request))
        img = Image.open(reqfile)
        img.save("/home/daizhen/projects/ImagesCategory/data/gray_images/__upload.jpeg","jpeg")
        
        model = ModelPrediction.ModelPrediction()
        model.image_path = '/home/daizhen/projects/ImagesCategory/data/gray_images/__upload.jpeg'
        predict_category, predict_subcategory, predict_producttype,predictd_product = model.Predict()
        
        value_dict = {}
        value_dict['filename'] = '__upload.jpeg'
        value_dict['predict_category'] = predict_category
        value_dict['predict_subcategory'] = predict_subcategory
        value_dict['predict_producttype'] = predict_producttype
        value_dict['predict_product'] = predictd_product
        return render_to_response('TestPage.html',value_dict, context_instance=RequestContext(request))

    return render_to_response('TestPage.html', context_instance=RequestContext(request))