"""
Visualization only, not for Mobilenet preprocessing

"""
import os #ignore future warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #to ignore warning instead of using tf

# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR) # this line is for hiding the futerwarning of tensorflow

from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import optimizers
import numpy as np
import random
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf
import itertools
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

###### process the data
#0 is a dog, 1  is a cat
train_path="/home/saba/PycharmProjects/testing/DeepLearning/DNN_Project6/skin cancer/train"
valid_path="/home/saba/PycharmProjects/testing/DeepLearning/DNN_Project6/skin cancer/valid"
test_path="/home/saba/PycharmProjects/testing/DeepLearning/DNN_Project6/skin cancer/test"




train_datagen = ImageDataGenerator(rescale=1./255) # re-scale pixel values between [0,1]
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_batches= train_datagen.flow_from_directory(train_path,target_size = (224,224),batch_size=10)
valid_batches= valid_datagen.flow_from_directory(valid_path,target_size = (224,224),batch_size=10)
test_batches= test_datagen.flow_from_directory(test_path,target_size = (224,224),batch_size=10,shuffle=False)

assert train_batches.n == 880
assert valid_batches.n == 252
assert test_batches.n == 126
assert train_batches.num_classes == valid_batches.num_classes ==test_batches.num_classes ==2

print(train_batches.class_indices) #to check the class for images, the cat has class 0, while dog has class 1
print(valid_batches.class_indices)
print(test_batches.class_indices)

# print(test_batches.classes) #print the targets without to_categorized
# num_classes=2
# epochs=5

imgs, labels=next(train_batches)


def plotImages(images_arr,labels):
    i=0
    fig, axes=plt.subplots(1, 10, figsize=(20,20))
    axes=axes.flatten()
    for img, ax in zip(images_arr, axes):

        ax.imshow(img)
        ax.set_title(labels[i])
        ax.axis('off')
        i +=1
    plt.tight_layout()
    plt.suptitle("Dangerous': 0, 'not_Dangerous': 1")
    plt.show()

plotImages(imgs, labels)

print(imgs.shape)
# print(imgs.shape[1:])
# print(labels)
