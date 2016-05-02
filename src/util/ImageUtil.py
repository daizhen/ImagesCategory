from PIL import Image
import os
import sys
import numpy as np


def LoadImage(imagePath):
    return Image.open(imagePath)

'''
Read Fixed size (pre-processed) image into array.
'''
def ReadImageToArray(image):
    #image = Image.open(imagePath)
    array = np.array(image)
    array = (array - (255 / 2.0)) / 255
    return array;

def PreprocessImage(imagePath,resultSize):
    image = Image.open(imagePath)
    grayImage = image.convert('L')
    return grayImage.resize(resultSize,Image.ANTIALIAS)