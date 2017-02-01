import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import cv2

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


state = -1
perturb_mark = 0.
perurb_steer = 0.


def perturb(perturb_steering = 0.3, perturb_dur_s = 0.5, perturb_delay_s = 10.0):
    '''
    Generate a steerig perturbation of size perturb_steering and
    duration perturb_dur_s. Wait perturb_delay_s between successive
    perturbations
    '''

    global state
    global perturb_mark
    global perturb_steer

    if state == -1: # not initialized
        perturb_mark = time.time()
        state = 0

    elif state == 0: # perturbation 
        if (time.time() - perturb_mark) > perturb_dur_s:
            perturb_mark = time.time()
            perturb_steer = 0
            state = 1
            
    else:# dwelling
        if (time.time() - perturb_mark) > perturb_delay_s:
            perturb_mark = time.time()
            perturb_steer = perturb_steering * np.random.choice([-1., 1.])
            state = 0

    if abs(perturb_steer) > 0:
        if perturb_steer > 0:
            print('>>>>>>>')
        else:
            print('<<<<<<<')
    else:
        print('')

    return perturb_steer


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    # crop
    image_array = image_array[34:-20,:,:]

    # resize & normalize
    image_array = (cv2.resize(image_array, (66,66), interpolation=cv2.INTER_LINEAR).astype(np.float64))/255. - 0.5
    
    transformed_image_array = image_array[None, :, :, :]
    
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.3
    #p=perturb()
    #print(steering_angle, throttle)
    print('{:.1f}'.format(steering_angle*25.))

    send_control(steering_angle+p, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        #model = model_from_json(jfile.read())
        model = model_from_json(json.loads(jfile.read()))


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)