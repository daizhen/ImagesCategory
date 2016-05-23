from django.http import HttpResponse
from django.shortcuts import render_to_response

import sys
sys.path.append('/home/daizhen/projects/ImagesCategory/src/util')
import CSVUtil as CSVUtil

demo_set = CSVUtil.ReadCSV('/home/daizhen/projects/ImagesCategory/src/demo/demo_data_result.csv')
def default(request):
    index =  request.GET.get('index')
    index = int(index)
    #return HttpResponse(index)
    #return HttpResponse(type(testset))
    value_dict = {}
    value_dict['filename'] = demo_set[index][0]
    value_dict['index'] = index
    value_dict['actual_category'] =  demo_set[index][1]
    value_dict['actual_subcategory'] = demo_set[index][2]
    value_dict['actual_producttype'] = demo_set[index][3]
    value_dict['actual_product'] = demo_set[index][4]
    value_dict['predict_category'] = demo_set[index][6]
    value_dict['predict_subcategory'] = demo_set[index][7]
    value_dict['predict_producttype'] = demo_set[index][8]
    value_dict['predict_product'] = demo_set[index][9]
    
    return render_to_response('default.html',value_dict)
    #return HttpResponse('<h1>main pange</h1>')