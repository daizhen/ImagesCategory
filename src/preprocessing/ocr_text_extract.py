import csv
import os
import sys
import numpy as np
from havenondemand.hodclient import *

def LoadProgress():
    fileName='progress.csv'
    progressDict = {};
    if not os.path.exists(fileName):
        return {}
    csvfile = file(fileName, 'rb')
    reader = csv.reader(csvfile)
	
    for line in reader:
        if len(line) > 0:
            currentFileName = line[0]
            progressDict[currentFileName] = True
    csvfile.close()
    return progressDict

def UpdateProgress(fileName):
    progressFileName='./progress.csv'
    csvfile = file(progressFileName, 'ab')
    writer = csv.writer(csvfile)
	
    writer.writerow([fileName])
    csvfile.close()
    
def StoreResult(fileName,response):
    progressFileName='./ocr_result.csv'
    csvfile = file(progressFileName, 'ab')
    writer = csv.writer(csvfile)
	
    responseText = ''
    for i in range(len(response)):
        responseText = responseText + '\n' + response[i]['text']
    
    writer.writerow([fileName,responseText.encode('utf-8')])
    csvfile.close()   

   
def Process_OCR(in_dir):
    fileList = os.listdir(in_dir)
    api_key = 'd92436dc-74a0-416d-9a8b-1d03cd493af3'
    progressDict = LoadProgress()
    client = HODClient(api_key, version="v1")
    
    for imageFileName in fileList:
        try:
            if not imageFileName in progressDict:
                #print imageFileName
                in_full_name = os.path.join(in_dir,imageFileName.encode('utf-8'))
                params = {'file': in_full_name.decode('utf-8')}
                result = client.post_request(params, HODApps.OCR_DOCUMENT, async=False)
                #print result['text_block']
                #Update progress
                StoreResult(imageFileName,result['text_block'])
                progressDict[imageFileName] = True;
                UpdateProgress(imageFileName)
        except  Exception as e:
            print imageFileName
            
if __name__ == "__main__":  
    #ResizeImages("../sample_data/gray_images","../sample_data/100_100",(100,100))
    #Process_OCR("../../data/gray_images")
    Process_OCR("/media/sf_VM_Share/jpg_images")
    
    '''
    progress = LoadProgress()
    print progress
    UpdateProgress('123.jpeg');
    UpdateProgress('456.jpeg');
    '''       
    