################################################################################

# Example : perform conversion of inceptionVxOnFire tflearn models to TensorFlow protocol
# buffer (.pb) binary format and tflife format files (for import into other tools, example OpenCV)

# Copyright (c) 2019 Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

# Acknowledgements: some portions - tensorflow tutorial examples and URL below

################################################################################

import glob,os
import tensorflow as tf
import sys
import argparse
sys.path.append('..')

################################################################################

from inceptionVxOnFire import construct_inceptionv1onfire, construct_inceptionv3onfire, construct_inceptionv4onfire
from converter import convert_to_pb
from converter import convert_to_tflite

################################################################################

parser = argparse.ArgumentParser(description='Perform InceptionV1/V3/V4 model conversion')
parser.add_argument("-m", "--model_to_use", type=int, help="specify model to use", default=1, choices={1, 3, 4})
args = parser.parse_args()

################################################################################

# perform conversion of the specified binary detection model

if (args.model_to_use == 1):

    # use InceptionV1-OnFire CNN model - [Dunning/Breckon, 2018]

    model = construct_inceptionv1onfire (224, 224, False)
    print("[INFO] Constructed InceptionV1-OnFire (binary, full-frame)...")

    path = "../models/InceptionV1-OnFire/inceptiononv1onfire"; # path to tflearn checkpoint including filestem
    input_layer_name = 'InputData/X'            # input layer of network
    output_layer_name= 'FullyConnected/Softmax' # output layer of network
    filename = "inceptionv1onfire.pb"           # output pb format filename

elif (args.model_to_use == 3):

    # use InceptionV3-OnFire CNN model -  [Samarth/Bhowmik/Breckon, 2019]
    # N.B. for conversion to .pb + .tflite format we disable batch norm, which is a hack but seems to work
    # (if we don't then this issue occurs - https://github.com/tensorflow/tensorflow/issues/3628)

    model = construct_inceptionv3onfire (224, 224, training=False, enable_batch_norm=False)
    print("[INFO] Constructed InceptionV3-OnFire (binary, full-frame)...")

    path = "../models/InceptionV3-OnFire/inceptionv3onfire"; # path to tflearn checkpoint including filestem
    input_layer_name = 'InputData/X'            # input layer of network
    output_layer_name= 'FullyConnected/Softmax' # output layer of network
    filename = "inceptionv3onfire.pb"           # output pb format filename

elif (args.model_to_use == 4):

    # use InceptionV4-OnFire CNN model - [Samarth/Bhowmik/Breckon, 2019]
    # N.B. for conversion to .pb + .tflite format we disable batch norm, which is a hack but seems to work
    # (if we don't then this issue occurs - https://github.com/tensorflow/tensorflow/issues/3628)

    model = construct_inceptionv4onfire (224, 224, training=False, enable_batch_norm=False)
    print("[INFO] Constructed InceptionV4-OnFire (binary, full-frame)...")

    path = "../models/InceptionV4-OnFire/inceptionv4onfire"; # path to tflearn checkpoint including filestem
    input_layer_name = 'InputData/X'            # input layer of network
    output_layer_name= 'FullyConnected/Softmax' # output layer of network
    filename = "inceptionv4onfire.pb"           # output pb format filename

convert_to_pb(model, path, input_layer_name,  output_layer_name, filename)
convert_to_tflite(filename, input_layer_name,  output_layer_name)

################################################################################

# reset TensorFlow before next conversion

tf.reset_default_graph()

################################################################################

# perform conversion of the specified superpixel based detection model

if (args.model_to_use == 1):

    # use InceptionV1-OnFire CNN model - [Dunning/Breckon, 2018]

    model_sp = construct_inceptionv1onfire (224, 224, False)
    print("[INFO] Constructed InceptionV1-OnFire (superpixel)...")

    input_layer_name = 'InputData/X'            # input layer of network
    output_layer_name= 'FullyConnected/Softmax' # output layer of network
    path_sp = "../models/SP-InceptionV1-OnFire/sp-inceptiononv1onfire"; # path to tflearn checkpoint including filestem
    filename_sp = "sp-inceptionv1onfire.pb"         # output filename

elif (args.model_to_use == 3):

    # use InceptionV3-OnFire CNN model - [Samarth/Bhowmik/Breckon, 2019]
    # N.B. for conversion to .pb + .tflite format we disable batch norm, which is a hack but seems to work
    # (if we don't then this issue occurs - https://github.com/tensorflow/tensorflow/issues/3628)

    model_sp = construct_inceptionv3onfire (224, 224, training=False, enable_batch_norm=False)
    print("[INFO] Constructed InceptionV3-OnFire (superpixel)...")

    input_layer_name = 'InputData/X'            # input layer of network
    output_layer_name= 'FullyConnected/Softmax' # output layer of network
    path_sp = "../models/SP-InceptionV3-OnFire/sp-inceptionv3onfire"; # path to tflearn checkpoint including filestem
    filename_sp = "sp-inceptionv3onfire.pb"         # output filename

elif (args.model_to_use == 4):

    # use InceptionV4-OnFire CNN model - [Samarth/Bhowmik/Breckon, 2019]
    # N.B. for conversion to .pb + .tflite format we disable batch norm, which is a hack but seems to work
    # (if we don't then this issue occurs - https://github.com/tensorflow/tensorflow/issues/3628)

    model_sp = construct_inceptionv4onfire (224, 224, training=False, enable_batch_norm=False)
    print("[INFO] Constructed InceptionV4-OnFire (superpixel)...")

    input_layer_name = 'InputData/X'            # input layer of network
    output_layer_name= 'FullyConnected/Softmax' # output layer of network
    path_sp = "../models/SP-InceptionV4-OnFire/sp-inceptionv4onfire"; # path to tflearn checkpoint including filestem
    filename_sp = "sp-inceptionv4onfire.pb"         # output filename

convert_to_pb(model_sp, path_sp, input_layer_name,  output_layer_name, filename_sp)
convert_to_tflite(filename_sp, input_layer_name,  output_layer_name)

################################################################################
