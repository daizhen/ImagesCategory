from django.http import HttpResponse
from django.shortcuts import render_to_response

import sys
sys.path.append('/home/daizhen/projects/ImagesCategory/src/util')
import CSVUtil as CSVUtil

testset = CSVUtil.ReadCSV('/home/daizhen/projects/ImagesCategory/data/test_data.csv')
def default(request):
    index =  request.GET.get('index')
    index = int(index)
    #return HttpResponse(index)
    #return HttpResponse(type(testset))
    return render_to_response('default.html',{'index':index, 'filename':testset[index][0]})
    #return HttpResponse('<h1>main pange</h1>')