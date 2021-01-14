
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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score


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
def get_prediction(image_path):
    predictions={}
    model = load_model("mobile_net_test.h5")
    img_processed = preprocessing_image(image_path)
    pred = model.predict(x=img_processed)
    predictions['dangerous']=np.round(pred[0][0],3)
    predictions['not-dangerous'] = np.round(pred[0][1],3)
    return (predictions)

# predcit only one image
image_path='/home/saba/PycharmProjects/testing/DeepLearning/DNN_Project6/skin cancer/test/Dangerous/D25.BMP'
#img_processed=preprocessing_image(image_path)
output=get_prediction(image_path)
print(output)


pdb.set_trace()

### load the model and make prediction
model = load_model("mobile_net_test.h5")

#call your data
test_path="/home/saba/PycharmProjects/testing/DeepLearning/DNN_Project6/skin cancer/test"

#do not shuffle the test, so that we can draw the confusion matrix, and compute f1, accuracy,......
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)


assert test_batches.n == 126
assert test_batches.num_classes ==2


### load the model and make prediction
model = load_model("mobile_net_test.h5")

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
predictions_prob =np.round(predictions, 3)
y_prediction=np.argmax(predictions_prob, axis=-1)
predictions_prob_y=np.max(predictions_prob, axis=-1)
print(predictions_prob_y)
print(y_prediction)

y_true=test_batches.classes


print(f"the accuracy: {accuracy_score(y_true, y_prediction) :0.3f} ")
print(f"f1 score: {f1_score(y_true, y_prediction) :0.3f}")
print(f"Precision score: {precision_score(y_true, y_prediction) :0.3f}")
print(f"Recall score: {recall_score(y_true, y_prediction) :0.3f}")
print(f'The ROC AUC  Score: {roc_auc_score(y_true, y_prediction) :0.3f}')


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(y_true, y_prediction)
#sns.heatmap(cm, annot=True) #to draw confusion-matrix
cm_plot_labels = ['dangerous','not_dangerous']
#plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import metrics

# y_true = # ground truth labels
# y_probas = # predicted probabilities generated by sklearn classifier
#skplt.metrics.plot_roc_curve(np.array(y_true),np.array(predictions_prob_y))
skplt.metrics.plot_roc(y_true, predictions_prob, plot_micro=False, plot_macro=False, classes_to_plot=1)
plt.show()