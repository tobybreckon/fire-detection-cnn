################################################################################

# Example : perform validation of FireNet models in TFLearn, PB and TFLite formats

# Copyright (c) 2019 - Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

################################################################################

import cv2
import os
import sys
import math

################################################################################

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

################################################################################

VALIDATE_TO_PRECISION_N = 5

################################################################################

sys.path.append('..')
from firenet import construct_firenet

################################################################################

# tflearn - load model

print("Load tflearn model from: ../models/FireNet ...", end = '')
model_tflearn = construct_firenet (224, 224, training=False)
model_tflearn.load(os.path.join("../models/FireNet", "firenet"),weights_only=True)
print("OK")

################################################################################

# tf protocol buffer - load model (into opencv)

print("Load protocolbuf (pb) model from: firenet.pb ...", end = '')
tensorflow_pb_model = cv2.dnn.readNetFromTensorflow('firenet.pb')
print("OK")

################################################################################

# tflite - load model

print("Load tflite model from: firenet.tflite ...", end = '')
tflife_model = tf.lite.Interpreter(model_path="firenet.tflite")
tflife_model.allocate_tensors()
print("OK")

# Get input and output tensors.
tflife_input_details = tflife_model.get_input_details()
tflife_output_details = tflife_model.get_output_details()

################################################################################

# load video file

video = cv2.VideoCapture("../models/test.mp4")
print("Load test video from ../models/test.mp4 ...")

# get video properties

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_counter = 0
fail_counter = 0

while (True):

    # get video frame from file, handle end of file

    ret, frame = video.read()
    if not ret:
        print("... end of video file reached")
        break

    print("frame: " + str(frame_counter),  end = '')
    frame_counter = frame_counter + 1

    # re-size image to network input size and perform prediction

    # input to networks is: 224x224x3 colour image with channel ordering as {B,G,R}
    # as is the opencv norm, not {R,G,B} and pixel value range 0->255 for each channel

    small_frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)

    ############################################################################

    np.set_printoptions(precision=6)

    # perform predictiion with tflearn model

    output_tflearn = model_tflearn.predict([small_frame])
    print("\t: TFLearn (original): ", end = '')
    print(output_tflearn, end = '')

    # perform prediction with protocolbuf model via opencv

    tensorflow_pb_model.setInput(cv2.dnn.blobFromImage(small_frame, size=(224, 224), swapRB=False, crop=False))
    output_tensorflow_pb = tensorflow_pb_model.forward()

    print("\t: Tensorflow .pb (via opencv): ", end = '')
    print(output_tensorflow_pb, end = '')

    # perform prediction with tflite model via TensorFlow

    tflife_input_data = np.reshape(np.float32(small_frame), (1, 224, 224, 3))
    tflife_model.set_tensor(tflife_input_details[0]['index'], tflife_input_data)

    tflife_model.invoke()

    output_tflite = tflife_model.get_tensor(tflife_output_details[0]['index'])
    print("\t: TFLite (via tensorflow): ", end = '')
    print(output_tflite, end = '')

    try:
        np.testing.assert_almost_equal(output_tflearn, output_tensorflow_pb, VALIDATE_TO_PRECISION_N)
        np.testing.assert_almost_equal(output_tflearn, output_tflite, 3)
        print(": all equal test - PASS")
    except AssertionError:
        print(" all equal test - FAIL")
        fail_counter = fail_counter +1

################################################################################
print("*** FINAL cross-model validation FAILS (for precision of " + str(VALIDATE_TO_PRECISION_N) + ") = " + str(fail_counter))
################################################################################
