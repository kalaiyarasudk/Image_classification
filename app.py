from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Path to the saved model
model_path = 'cnn_model_with_bnn.h5'

# Load the trained model
model = load_model(model_path)

# Set the upload folder
app.config['UPLOAD_FOLDER'] =  'static/uploads/'

# Allowed extensions for uploaded images
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_image():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    # If no file is selected, redirect back to the index page
    if file.filename == '':
        return redirect(request.url)

    # If the file has an allowed extension, proceed with prediction
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform image classification
        img = image.load_img(filepath, target_size=(32, 32))
        x = image.img_to_array(img)
        x = x.reshape((1, ) + x.shape)
        x /= 255.0
        prediction = model.predict(x)

        # Map class index to class name (you can modify this based on your classes)
        class_names = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }

        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]

        return redirect(url_for('result', filename=filename, prediction=predicted_class))

    # If the file has an invalid extension, redirect back to the index page
    return redirect(request.url)

@app.route('/result')
def result():
    filename = request.args.get('filename')
    prediction = request.args.get('prediction')
    return render_template('result.html', filename=filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
