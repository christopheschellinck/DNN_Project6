
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
import cv2
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras import optimizers
import random
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import itertools
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
import pdb




#call your data
train_path="/home/saba/PycharmProjects/testing/DeepLearning/DNN_Project6/skin cancer/train"
valid_path="/home/saba/PycharmProjects/testing/DeepLearning/DNN_Project6/skin cancer/valid"
test_path="/home/saba/PycharmProjects/testing/DeepLearning/DNN_Project6/skin cancer/test"


#get the train, test, valid and use the preprocessing of mobilenet  usig mobilenet.preprocess_input
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
#do not shuffle the test, so that we can draw the confusion matrix, and compute f1, accuracy,......
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

#set the train, valid, test size
assert train_batches.n == 880
assert valid_batches.n == 252
assert test_batches.n == 126
#set the number of classes
assert train_batches.num_classes == valid_batches.num_classes ==test_batches.num_classes ==2

#be sure danerous, no-dangerous to which classes they belong
print(train_batches.class_indices) #to check the class for images, the cat has class 0, while dog has class 1
print(valid_batches.class_indices)
print(test_batches.class_indices)

#set the mobilenet DNN model, call the model to use it
mobile = tf.keras.applications.mobilenet.MobileNet()
#print the summary
print(mobile.summary())

#Use all avaimable layers except the last 6
x = mobile.layers[-6].output

#create the output layer, units=2 because we have 2 classes
#note: the mobilenet is a functional model API, x will pass all the previos layers to the output layer
output = Dense(units=2, activation='softmax')(x)

#construct new model, set the output and input
model = Model(inputs=mobile.input, outputs=output)

print(model.summary())

# # freeze the last 23 layers make them not learned, and training them during fitting
#NOTE:::::::::::::::::::::::::::::::: TO DO, play with the number of negligible layers, this effect overfitting
##### Increse the number of Freeze layers, I thinkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
for layer in model.layers[:-23]:
    layer.trainable = False

#so the number of not trained parameter is increased because we freeze the layers, we do not consider there learnign
print(model.summary())

#compile your model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#TO DO, increase epochs, this effect overfitting, when increase number of epochs then we decrease overfitting
model.fit(x=train_batches,
            #steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            #validation_steps=len(valid_batches),
            epochs=10,
            verbose=2
)


##### save the model
model.save("mobile_net_test.h5")

# cnn4 = load_model("cnn4.h5")

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
predictions =np.round(predictions)


def make_confusion_matrix(cf,group_names=None,categories='auto',count=True,
           percent=True,cbar=True,xyticks=True,xyplotlabels=True,
           sum_stats=True,figsize=(10,10),cmap='Blues',title=None):

    blanks = ['' for i in range(cf.size)]
    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks
    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks
    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])
    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))
        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}, Precision={:0.3f},Recall={:0.3f}, F1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""
    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False
    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)
    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

from sklearn.metrics import confusion_matrix#Fit the model
cf_matrix =confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Zero(dangerous)', 'One(not-dangerous)']
make_confusion_matrix(cf_matrix,group_names=labels,categories=categories,cmap='binary')

plt.show()


# """
# Testing Only
# """
# # to display 10 image in
# image1 = Image.open('/home/saba/Downloads/skin cancer/SET_D/D1.BMP').convert('RGB') #(387, 632, 3) this is the shape
# opimage1=image1.resize((224, 224), Image.ANTIALIAS)
# # arr_img = np.asarray(opimage)
# # print(arr_img.shape) #to be sure that the size is changed
# image2 = Image.open('/home/saba/Downloads/skin cancer/SET_D/D2.BMP').convert('RGB') #(387, 632, 3) this is the shape
# opimage2=image2.resize((224, 224), Image.ANTIALIAS)
# image3 = Image.open('/home/saba/Downloads/skin cancer/SET_D/D3.BMP').convert('RGB') #(387, 632, 3) this is the shape
# opimage3=image3.resize((224, 224), Image.ANTIALIAS)
#
# ###############################################
#
# def prepare_image(file):
#     img_path = '/home/saba/Downloads/skin cancer/SET_D'
#
#     #img = image.load_img(img_path + file, target_size=(224, 224))
#     image = Image.open(img_path+'/'+file)
#     #mg = image.convert('RGB')
#     img = image.resize((224, 224), Image.ANTIALIAS)
#     #img_array = image.img_to_array(img)
#     img_array  = np.asarray(img)
#     img_array_expanded_dims = np.expand_dims(img_array, axis=0)
#     return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
#
# mobile = tf.keras.applications.mobilenet.MobileNet()
#
# preprocessed_image = prepare_image('D1.BMP')
# predictions = mobile.predict(preprocessed_image)
# results = imagenet_utils.decode_predictions(predictions)
# print(results)




