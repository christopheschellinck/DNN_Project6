from flask import Flask, render_template, flash, request, redirect, url_for, jsonify
import pickle
import json
import numpy as np
import os
from keras.preprocessing import image
from keras.applications import mobilenet
from keras.models import load_model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask("__name__")
port = int(os.environ.get("PORT", 5000))

App_ROOT=os.path.dirname(os.path.abspath(__file__))


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

#the first page
@app.route('/')
def index():
    return render_template('upload.html') #, error_message=error_message -> error_message_comming_from.preprocessing.preprocess())

@app.route('/upload', methods=["POST"])
def upload():
    filename=" "; prediction={};not_dan_pre=0;dan_pre=0
    #this is where I store the image that I get it from API, in the folder asset
    target=os.path.join(App_ROOT, 'asset/')
    print(target)

    #if there is no directory asset, create it
    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist('file'):
        print(file)
        filename=file.filename
        destination='/'.join([target, filename]) #take the image to the asset destination
        file.save(destination) #save the file

    #compute the prediction using cnn model
    image_path = destination #get the last uploaded image
    prediction= get_prediction(image_path)
    print(prediction)
    dan_pre=prediction['dangerous']
    not_dan_pre=prediction['not-dangerous']
    print(dan_pre)
    print(not_dan_pre)
    return (render_template('complete.html', dan_pre=dan_pre, not_dan_pre=not_dan_pre)) #, error_message=error_message -> error_message_comming_from.preprocessing.preprocess())



if __name__=="__main__":
    #app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=port)
