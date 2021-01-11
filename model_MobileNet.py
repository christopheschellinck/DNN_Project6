

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
import cv2

# to display 10 image in
image1 = Image.open('/home/saba/Downloads/skin cancer/SET_D/D1.BMP').convert('RGB') #(387, 632, 3) this is the shape
opimage1=image1.resize((224, 224), Image.ANTIALIAS)
# arr_img = np.asarray(opimage)
# print(arr_img.shape) #to be sure that the size is changed
image2 = Image.open('/home/saba/Downloads/skin cancer/SET_D/D2.BMP').convert('RGB') #(387, 632, 3) this is the shape
opimage2=image2.resize((224, 224), Image.ANTIALIAS)
image3 = Image.open('/home/saba/Downloads/skin cancer/SET_D/D3.BMP').convert('RGB') #(387, 632, 3) this is the shape
opimage3=image3.resize((224, 224), Image.ANTIALIAS)

# to display 10 image in
image4= Image.open('/home/saba/Downloads/skin cancer/SET_E/E1.BMP').convert('RGB') #(387, 632, 3) this is the shape
opimage4=image4.resize((224, 224), Image.ANTIALIAS)
# arr_img = np.asarray(opimage)
# print(arr_img.shape) #to be sure that the size is changed
image5= Image.open('/home/saba/Downloads/skin cancer/SET_E/E2.BMP').convert('RGB') #(387, 632, 3) this is the shape
opimage5=image5.resize((224, 224), Image.ANTIALIAS)
image6 = Image.open('/home/saba/Downloads/skin cancer/SET_E/E3.BMP').convert('RGB') #(387, 632, 3) this is the shape
opimage6=image6.resize((224, 224), Image.ANTIALIAS)


# to display 10 image in
image7= Image.open('/home/saba/Downloads/skin cancer/SET_F/F1.BMP').convert('RGB') #(387, 632, 3) this is the shape
opimage7=image7.resize((224, 224), Image.ANTIALIAS)
# arr_img = np.asarray(opimage)
# print(arr_img.shape) #to be sure that the size is changed
image8 = Image.open('/home/saba/Downloads/skin cancer/SET_F/F2.BMP').convert('RGB') #(387, 632, 3) this is the shape
opimage8=image8.resize((224, 224), Image.ANTIALIAS)
image9 = Image.open('/home/saba/Downloads/skin cancer/SET_F/F3.BMP').convert('RGB') #(387, 632, 3) this is the shape
opimage9=image9.resize((224, 224), Image.ANTIALIAS)


fig = plt.figure(figsize=(8,3))
ax = fig.add_subplot(3, 3, 1, xticks=[], yticks=[])
ax.set_title('\n Sub D images:')
plt.imshow(opimage1)
ax = fig.add_subplot(3, 3, 2, xticks=[], yticks=[])
plt.imshow(opimage2)
ax = fig.add_subplot(3, 3, 3, xticks=[], yticks=[])
plt.imshow(opimage3)

ax1 = fig.add_subplot(3, 3, 4, xticks=[], yticks=[])
ax1.set_title('\n Sub E images:')
plt.imshow(opimage4)
ax1 = fig.add_subplot(3, 3, 5, xticks=[], yticks=[])
plt.imshow(opimage5)
ax1 = fig.add_subplot(3, 3, 6, xticks=[], yticks=[])
plt.imshow(opimage6)

ax2 = fig.add_subplot(3, 3, 7, xticks=[], yticks=[])
ax2.set_title('\n Sub F images:')
plt.imshow(opimage7)
ax2 = fig.add_subplot(3, 3, 8, xticks=[], yticks=[])
plt.imshow(opimage8)
ax2 = fig.add_subplot(3, 3, 9, xticks=[], yticks=[])
plt.imshow(opimage9)



# plt.show()


###############################################

def prepare_image(file):
    img_path = '/home/saba/Downloads/skin cancer/SET_D'

    #img = image.load_img(img_path + file, target_size=(224, 224))
    image = Image.open(img_path+'/'+file)
    #mg = image.convert('RGB')
    img = image.resize((224, 224), Image.ANTIALIAS)
    #img_array = image.img_to_array(img)
    img_array  = np.asarray(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

mobile = tf.keras.applications.mobilenet.MobileNet()

preprocessed_image = prepare_image('D1.BMP')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)
#from IPython.display import Image
#from PIL import Image



