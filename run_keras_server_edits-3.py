# import the necessary packages
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import keras

import flask
from flask import Flask, render_template
from flask import request, redirect
import io

from PIL import Image
import PIL

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import model as modellib, utils


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
from matplotlib.patches import Rectangle

from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename


# Define custom config
class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 5000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95

# Define custom config (used for inferences)
class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list, save_name):
     # load the image
     data = pyplot.imread(filename)
     # plot the image
     pyplot.imshow(data)
     # get the context for drawing boxes
     ax = pyplot.gca()
     # plot each box
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
          ax.add_patch(rect)
     # show the plot
     pyplot.savefig('static/uploads/'+save_name)



# initialize our Flask application and the Keras model

def load_model():
    # load the pre-trained Keras model
    global model
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir='./Mask_RCNN')
    model.load_weights('./mask_rcnn_object_0020.h5', by_name=True)

    global graph
    graph = tf.get_default_graph()

# Call load model function (necessary if running through flask app)
load_model()

# Extensions defined for our images
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# check if file is allowed
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to main page
@app.route('/')
def upload_form():
	return render_template('index.html')

# Function to display model with damages
@app.route('/', methods=['POST'])
def upload_image():

    # Ensure file is selected
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    # Save file to static folder
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


        # Load image as array
        uploaded_image = load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        uploaded_image = img_to_array(uploaded_image)

        # Make predictions
        with graph.as_default():
            preds = model.detect([uploaded_image], verbose=0)

        # Save detected image
        image_name = "detected_"+filename
        draw_image_with_boxes(os.path.join(app.config['UPLOAD_FOLDER'], filename), preds[0]['rois'], image_name)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=image_name)

    else:

        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

# Redirect to page with uploaded image
@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


# Uncomment to run as python file as opposed to through Flask
'''
if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...")
    load_model()
    app.run()
'''
