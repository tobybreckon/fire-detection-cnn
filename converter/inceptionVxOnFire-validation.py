################################################################################

# Example : perform validation of InceptionVx-OnFire models in TFLearn, PB and TFLite formats

# Copyright (c) 2019 - Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

################################################################################

import cv2
import os
import sys
import math
import argparse

################################################################################

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

################################################################################

VALIDATE_TO_PRECISION_N = 3

################################################################################

sys.path.append('..')
from inceptionVxOnFire import construct_inceptionv1onfire, construct_inceptionv3onfire, construct_inceptionv4onfire

################################################################################

parser = argparse.ArgumentParser(description='Perform InceptionV1/V3/V4 model validation')
parser.add_argument("-m", "--model_to_use", type=int, help="specify model to use", default=1, choices={1, 3, 4})
parser.add_argument("-sp", "--superpixel_model", action='store_true', help="use superpixel version  of model")
parser.add_argument("-i", "--input_video", type=str, help="specify test video to use", default="../models/test.mp4")
parser.add_argument("-d", "--display", action='store_true', help="use superpixel version  of model")
args = parser.parse_args()

################################################################################

# perform conversion of the specified binary or superpixel detection model

if (args.superpixel_model):
    pre_string = "SP-"
else:
    pre_string = ""

if (args.model_to_use == 1):

    # use InceptionV1-OnFire CNN model - [Dunning/Breckon, 2018]

    model_tflearn = construct_inceptionv1onfire (224, 224, training=False)
    print("[INFO] Constructed " + pre_string + "InceptionV1-OnFire ...")
    path = "../models/" + pre_string + "InceptionV1-OnFire/" + pre_string.lower() + "inceptiononv1onfire"; # path to tflearn checkpoint including filestem

elif (args.model_to_use == 3):

    # use InceptionV3-OnFire CNN model -  [Samarth/Bhowmik/Breckon, 2019]
    # N.B. here we enable batch norm as this is for the TFLearn model only, which needs it activated to work

    model_tflearn = construct_inceptionv3onfire (224, 224, training=False, enable_batch_norm=True)
    print("[INFO] Constructed " + pre_string + "InceptionV3-OnFire ...")
    path = "../models/" + pre_string + "InceptionV3-OnFire/" + pre_string.lower() + "inceptionv3onfire"; # path to tflearn checkpoint including filestem

elif (args.model_to_use == 4):

    # use InceptionV4-OnFire CNN model - [Samarth/Bhowmik/Breckon, 2019]
    # N.B. here we enable batch norm as this is for the TFLearn model only, which needs it activated to work

    model_tflearn = construct_inceptionv4onfire (224, 224, training=False, enable_batch_norm=True)
    print("[INFO] Constructed " + pre_string + "InceptionV4-OnFire ...")
    path = "../models/" + pre_string + "InceptionV4-OnFire/" + pre_string.lower() + "inceptionv4onfire"; # path to tflearn checkpoint including filestem

################################################################################

# tflearn - load model

print("Load tflearn model from: " + path + " ...", end = '')

# only use wieghts_only for V1 model, due to use of batch norm
model_tflearn.load(path,weights_only=(args.model_to_use == 1))

print("OK")

################################################################################

# tf protocol buffer - load model (into opencv)

try:
    print("Load protocolbuf (pb) model from: " + pre_string.lower() + "inceptiononv" + str(args.model_to_use) + "onfire.pb ...", end = '')
    tensorflow_pb_model = cv2.dnn.readNetFromTensorflow(pre_string.lower() + "inceptionv" + str(args.model_to_use) + "onfire.pb")
    print("OK")
except:
    print("FAIL")
    print("ERROR: file " +  (pre_string.lower() + "inceptionv" + str(args.model_to_use) + "onfire.pb") + " missing, ensure you run the correct convertor to generate it first!")
    exit(1)

################################################################################

# tflite - load model

print("Load tflite model from: " + pre_string.lower() + "inceptiononv" + str(args.model_to_use) + "onfire.tflite ...", end = '')
tflife_model = tf.lite.Interpreter(model_path=pre_string.lower() + "inceptionv" + str(args.model_to_use) + "onfire.tflite")
tflife_model.allocate_tensors()
print("OK")

# Get input and output tensors.
tflife_input_details = tflife_model.get_input_details()
tflife_output_details = tflife_model.get_output_details()

################################################################################

# load video file

video = cv2.VideoCapture(args.input_video)
print("Load test video from " + args.input_video + " ...")

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

    if (args.display):
        cv2.imshow("Validation Image", frame)
        cv2.waitKey(40)

    print("frame: " + str(frame_counter),  end = '')
    frame_counter = frame_counter + 1

    # re-size image to network input size and perform prediction

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
        np.testing.assert_almost_equal(output_tflearn, output_tflite, VALIDATE_TO_PRECISION_N)
        print(": all equal test - PASS")
    except AssertionError:
        print(" all equal test - FAIL")
        fail_counter = fail_counter +1

################################################################################
print("*** FINAL cross-model validation FAILS (for precision of " + str(VALIDATE_TO_PRECISION_N) + ") = " + str(fail_counter))
################################################################################
