#import BaseUtil  
import os  
from PIL import Image
def ResizeImages(in_dir, out_dir,size):
    fileList = os.listdir(in_dir)
    for imageFile in fileList:
        in_full_name = os.path.join(in_dir,imageFile)
        out_full_name = os.path.join(out_dir,imageFile)
        img = Image.open(in_full_name)
        targetImage = img.resize(size,Image.ANTIALIAS)
        targetImage.save(out_full_name)

if __name__ == "__main__":  
    #ResizeImages("../sample_data/gray_images","../sample_data/100_100",(100,100))
    ResizeImages("../data/gray_images","../data/100_100",(100,100))   