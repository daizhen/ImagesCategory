from PIL import Image
import os
import sys
import numpy as np


def LoadImage(imagePath):
    return Image.open(imagePath)

'''
Read Fixed size (pre-processed) image into array.
'''
def ReadImageToArray(image, width, height):
    #image = Image.open(imagePath)
    array = np.array(image)
    return array;

def PreprocessImage(imagePath,resultSize):
    image = Image.open(imagePath)
    grayImage = image.convert('L')
    return grayImage.resize(resultSize,Image.ANTIALIAS)