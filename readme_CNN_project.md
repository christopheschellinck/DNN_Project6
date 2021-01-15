# Mole detection of skin cancer

## Introduction

* Repository: `SabaYahyaa/DNN_Project6`
* Type of Challenge: `Consolidation of AI (Convolutional Neural Network) and Deployment on the web`
* description of the challenge: `Provided with skin mark images, we made a CNN model that predicts if those marks are likely to be skin cancer (or almost) or not. On top of that we made a deployment solution. In other words, a webpage that receives an image of a skin mark which is sent to our model. And from the model it sends back to the webpage a prediction. At the end we have to present our solution to a team of people that is not technical. Neither on an IT or an AI level. The purpose is to be able to present a solution to people that have other skills.`
* Duration: `5 days`
* Deadline: `15/11/2020 12:30` ** Until that date/time the project is finished.**
* Presentation: `15/11/2020 14:00`
* Team challenge: `Ankita, Saba, Didier, Christophe from Becode/bouman`

## webpage making use of the model and prediction service

`https://skincancer-prediction.herokuapp.com/upload`

## Mission objectives 
* Be able to apply a CNN in a real context
* Be able to preprocess data for computer vision

![AI care!](./assets/ai-care.jpg)

## Technical Summary of approach and files
### preprocess the data: 
-> data_preparation.py
-> data_splitting.py: splitting the data in the folder structure hereunder
-> data_vizualization.py: gives diagrams like confusion matrix to check between dangerous or not and the model's guesses to be True or False

### Split the data in different folders: 
    train
        |_ dangerous
        |_ not_dangerous
    test
        |_ dangerous
        |_ not_dangerous
    valid
        |_ dangerous
        |_ not_dangerous

### the making of a model that is pretrained: MobileNet from Keras
* model_mobilenet.py
* Those pretrained models enables to make use of it's complete potential and use all layers not-fully connected CNN layers or some of them. This happens by freezing some we don't want to be activated.
* the results of the training can be found in 20200112training_results.odt    

### apply data augmentation 
* in order to improve the model and reduce overfitting, data augmentation was applied: due to a discrepancy between the model's 'accuracy' and 'validation_accuracy':
* by modifying the brightness of the images, we got the most optimized results. We used several options and combinations but that option was the best.

### Predictions
* The predictions can be found in the separate file predictions.py and integrated in the app_CNN.py one

### performance evaluation of the model
* performance_measure.py

### Deployment technologies Flask, Docker and Heroku
* app_CNN.py includes the preprocessing of the single image provided and the prediction and the flask code
* file making use of flask: app_CNN.py 
* Dockerfile
* Procfile for Heroku 

### Data
* the repo also contains the data under the folder 'skin cancer', in *.BMP format* that is used for this project

