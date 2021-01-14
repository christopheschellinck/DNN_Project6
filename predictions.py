############" the following is only for one image loading
#### to preprocess an image using mobile_net preprocessing
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.models import load_model

#preprocess an image
def preprocessing_image(img_path):
    img=image.load_img(img_path, target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array, axis=0)
    preprocessed_img=mobilenet.preprocess_input(img_array)
    return (preprocessed_img)
# get the prediction of preprocess an image
def get_prediction(image_path):
    predictions={}
    model = load_model("mobile_net_test.h5")
    img_processed = preprocessing_image(image_path)
    pred = model.predict(x=img_processed)
    predictions['dangerous']=np.round(pred[0][0],3)
    predictions['not-dangerous'] = np.round(pred[0][1],3)
    return (predictions)

