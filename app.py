import os
from flask import Flask, request, render_template, redirect, url_for
from predictions import get_prediction

app = Flask(__name__)
UPLOAD_FOLDER = 'static'

@app.route('/')
def index():
    return 'Mole prediction'

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file :
            image_location  = os.path.join(
                UPLOAD_FOLDER,image_file.filename
            )
            image_file.save(image_location)
            pred = get_prediction(image_location)

            return render_template('upload.html', prediction=pred)
    
        
    return render_template('upload.html', prediction=0)  #upload


if __name__ == '__main__':
    #global model
    #model = load_model('')
    app.run(host='0.0.0.0', port=7000, debug=True)

    