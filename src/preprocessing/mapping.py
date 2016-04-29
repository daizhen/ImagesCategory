import csv
import os
import sys
import numpy as np

def LoadLabels(fileName):
    csvfile = file(fileName, 'rb')
    reader = csv.reader(csvfile)
    index = 0
	
    label_list = list()
    for line in reader:
        if index != 0:
            currentLabel = line[1]
            if not currentLabel in label_list:
                
                label_list.append(currentLabel);
                #print line[0]
        index +=1
    csvfile.close()
    return label_list
    
    

def CreateCategoryMappings():

    categoryList = LoadLabels('../../all_category_data.csv')
    
    mapping_1 = [(categoryList[i],i) for i in range(len(categoryList))]
    mapping_2 = [(i,categoryList[i]) for i in range(len(categoryList))]
    
    csvfile = file('../../category_name_id_map.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['name', 'id'])

    writer.writerows(mapping_1)
    csvfile.close()
    
    csvfile = file('../../category_id_name_map.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'name'])

    writer.writerows(mapping_2)
    csvfile.close()
    

def CreateSubCategoryMappings():
    categoryList = LoadLabels('../../all_subcategory_data.csv')
    
    mapping_1 = [(categoryList[i],i) for i in range(len(categoryList))]
    mapping_2 = [(i,categoryList[i]) for i in range(len(categoryList))]
    
    csvfile = file('../../subcategory_name_id_map.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['name', 'id'])

    writer.writerows(mapping_1)
    csvfile.close()
    
    csvfile = file('../../subcategory_id_name_map.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'name'])

    writer.writerows(mapping_2)
    csvfile.close()

def CreateProductTypeMappings():
    categoryList = LoadLabels('../../all_producttype_data.csv')
    
    mapping_1 = [(categoryList[i],i) for i in range(len(categoryList))]
    mapping_2 = [(i,categoryList[i]) for i in range(len(categoryList))]
    
    csvfile = file('../../producttype_name_id_map.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['name', 'id'])

    writer.writerows(mapping_1)
    csvfile.close()
    
    csvfile = file('../../producttype_id_name_map.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'name'])

    writer.writerows(mapping_2)
    csvfile.close()

def CreateProductMappings():
    categoryList = LoadLabels('../../all_product_data.csv')
    
    mapping_1 = [(categoryList[i],i) for i in range(len(categoryList))]
    mapping_2 = [(i,categoryList[i]) for i in range(len(categoryList))]
    
    csvfile = file('../../product_name_id_map.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['name', 'id'])

    writer.writerows(mapping_1)
    csvfile.close()
    
    csvfile = file('../../product_id_name_map.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'name'])

    writer.writerows(mapping_2)
    csvfile.close()
if __name__ == "__main__":
    CreateCategoryMappings()     
    CreateSubCategoryMappings()  
    CreateProductTypeMappings()
    CreateProductMappings()