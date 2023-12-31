﻿# Image Classification Project

This project implements an image classification web application using a deep learning model. The model is trained on the CIFAR-10 dataset to classify images into 10 different categories.

## Prerequisites

- Python 3.6 or higher
- Flask
- TensorFlow
- Keras

## Installation

1. Clone the repository:


git clone https://github.com/kalaiyarasudk/Image-classification-Project.git
cd image-classification-project

Install the required dependencies  - pip install -r requirements.txt

## Usage

    Place the cnn_model_with_bn.h5 file in the root directory. This file contains the trained deep learning model.

    Run the Flask web application:
        python app.py
        Open your web browser and go to http://localhost:5000/ to access the web application.

        Upload an image using the provided form and click the "Classify" button to see the predicted class.

## Directory Structure
image-classification-project/
|-- app.py
|-- cnn_model_with_bn.h5
|-- static/
|   |-- css/
|   |   |-- style.css
|   |-- uploads/
|-- templates/
|   |-- index.html
|   |-- result.html
|-- README.md
|-- requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
    The CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
    Flask: https://flask.palletsprojects.com/
    TensorFlow: https://www.tensorflow.org/
    Keras: https://keras.io/
