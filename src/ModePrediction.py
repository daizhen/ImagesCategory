
class ModelPrediction:
    image_path=None
    category = None
    subcategory = None
    producttype= None
    product = None
    token_list = None
    
    session_category = None
    session_subcategory = None
    session_producttype =None
    session_product = None
    
    model_category = None
    model_subcategory = None
    model_producttype =None
    model_product = None
    base_dir = '/home/daizhen/projects/ImagesCategory';
    
    def Predict(self):
        pass
    
    def PredictCategory(self):
        if self.model_category == None:
            self.model_category = TrainModel.TrainModel()
            self.model_category.model_save_dir = os.path.join(self.base_dir,'trained_models/category')
            self.model_category.model_save_file_name='category_model'
            self.model_category.text_tokens_csv = os.path.join(self.base_dir,'data/all_trainning_tokens.csv')
            self.model_category.name_id_mapping_file = os.path.join(self.base_dir,'category_name_id_map.csv')
            self.model_category.InitVars()
        #image_file = os.path.join(base_dir,'data/jpg_images/sd21531814samm msg.jpeg')
        image = ImageUtil.PreprocessImage(self.image_path,(100,100))
        image_data = ImageUtil.ReadImageToArray(image)

        if self.token_list == None:
            raw_text =OCRUtil.ExtractText(self.image_path)
            self.token_list = TextVectorUtil.GetTokenList(raw_text)
        #image_data = array
        return self.model_category.Predict(image_data,self.token_list)
    
    def PredictSubCategory(self):
        if self.self. == None:
            self.model_subcategory = TrainModel.TrainModel()
            self.model_subcategory.model_save_dir = os.path.join(self.base_dir,'trained_models/subcategory')
            self.model_subcategory.model_save_file_name='subcategory_model'
            self.model_subcategory.text_tokens_csv = os.path.join(self.base_dir,'data/all_trainning_tokens.csv')
            self.model_subcategory.name_id_mapping_file = os.path.join(self.base_dir,'subcategory_name_id_map.csv')
            self.model_subcategory.InitVars()
        #image_file = os.path.join(base_dir,'data/jpg_images/sd21531814samm msg.jpeg')
        image = ImageUtil.PreprocessImage(self.image_path,(100,100))
        image_data = ImageUtil.ReadImageToArray(image)

        if self.token_list == None:
            raw_text =OCRUtil.ExtractText(self.image_path)
            self.token_list = TextVectorUtil.GetTokenList(raw_text)
        #image_data = array
        return self.model_subcategory.Predict(image_data,self.token_list)
    def PredictProductType(self):
        if self.self. == None:
            self.model_producttype = TrainModel.TrainModel()
            self.model_producttype.model_save_dir = os.path.join(self.base_dir,'trained_models/producttype')
            self.model_producttype.model_save_file_name='producttype_model'
            self.model_producttype.text_tokens_csv = os.path.join(self.base_dir,'data/all_trainning_tokens.csv')
            self.model_producttype.name_id_mapping_file = os.path.join(self.base_dir,'producttype_name_id_map.csv')
            self.model_producttype.InitVars()
        #image_file = os.path.join(base_dir,'data/jpg_images/sd21531814samm msg.jpeg')
        image = ImageUtil.PreprocessImage(self.image_path,(100,100))
        image_data = ImageUtil.ReadImageToArray(image)

        if self.token_list == None:
            raw_text =OCRUtil.ExtractText(self.image_path)
            self.token_list = TextVectorUtil.GetTokenList(raw_text)
        #image_data = array
        return self.model_producttype(image_data,self.token_list)
    
    def PredictProduct(self):
        if self.self. == None:
            self.model_product = TrainModel.TrainModel()
            self.model_product.model_save_dir = os.path.join(self.base_dir,'trained_models/product')
            self.model_product.model_save_file_name='product_model'
            self.model_product.text_tokens_csv = os.path.join(self.base_dir,'data/all_trainning_tokens.csv')
            self.model_product.name_id_mapping_file = os.path.join(self.base_dir,'product_name_id_map.csv')
            self.model_product.InitVars()
        #image_file = os.path.join(base_dir,'data/jpg_images/sd21531814samm msg.jpeg')
        image = ImageUtil.PreprocessImage(self.image_path,(100,100))
        image_data = ImageUtil.ReadImageToArray(image)

        if self.token_list == None:
            raw_text =OCRUtil.ExtractText(self.image_path)
            self.token_list = TextVectorUtil.GetTokenList(raw_text)
        #image_data = array
        return self.model_product.Predict(image_data,self.token_list)