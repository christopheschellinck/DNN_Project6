
import flask
import os #ignore future warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to ignore warning instead of using tf

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
import pdb


############" the following is only for one image loading
#### to preprocess an image using mobile_net preprocessing
from keras.preprocessing import image
from keras.applications import mobilenet, imagenet_utils
#preprocess an image
def preprocessing_image(img_path):
    img=image.load_img(img_path, target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array, axis=0)
    preprocessed_img=mobilenet.preprocess_input(img_array)
    return (preprocessed_img)
# get the prediction of preprocess an image
def get_prediction(img_processed):
    predictions={}
    model = load_model("mobile_net_test.h5")
    img_processed = preprocessing_image(image_path)
    pred = model.predict(x=img_processed)
    predictions['dangerous']=np.round(pred[0][0],3)
    predictions['not-dangerous'] = np.round(pred[0][1],3)
    return (predictions)

image_path='/home/saba/PycharmProjects/testing/DeepLearning/DNN_Project6/skin cancer/test/Dangerous/D25.BMP'
img_processed=preprocessing_image(image_path)
output=get_prediction(img_processed)
print(output)



# ### load the model and make prediction
# model = load_model("mobile_net_test.h5")
# predictions = model.predict(x=img_processed)
#
#
# #call your data
# test_path="/home/saba/PycharmProjects/testing/DeepLearning/DNN_Project6/skin cancer/test/Dangerous/D21.BMP"
#
# #do not shuffle the test, so that we can draw the confusion matrix, and compute f1, accuracy,......
# test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
#     directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)
#
#
# assert test_batches.n == 126
# #set the number of classes
# assert test_batches.num_classes ==2
#
# ### load the model and make prediction
# model = load_model("mobile_net_test.h5")
#
# predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
# #predictions =np.round(predictions)
#print(predictions)
