import os
import sys
import numpy as np
from havenondemand.hodclient import *

def ExtractText(image_path):
    api_key = 'd92436dc-74a0-416d-9a8b-1d03cd493af3'
    client = HODClient(api_key, version="v1")
    params = {'file': image_path.decode('utf-8')}
    result = client.post_request(params, HODApps.OCR_DOCUMENT, async=False)
    text_block = result['text_block']
    responseText = ''
    for i in range(len(text_block)):
        responseText = responseText + '\n' + text_block[i]['text']
    return responseText