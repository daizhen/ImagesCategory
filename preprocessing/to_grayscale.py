# -*- coding: utf-8 -*-
from PIL import Image
#from pylab import *
import os
import sys

def ConvertToGray(in_dir,out_dir):
    #读取图片,灰度化，并转为数组
    file_list = os.listdir(in_dir)
    for file in file_list:
        in_file = os.path.join(in_dir,file)
        out_file = os.path.join(out_dir,file);
        im = Image.open(in_file).convert('L')
        im.save(out_file);


ConvertToGray("../sample_data/original_images","../sample_data/gray_images")