################################################################################

# Example : perform conversion of inceptionV1OnFire tflearn model to TensorFlow protocol
# buffer (.pb) binary format and tflife format files (for import into other tools, example OpenCV)

# Copyright (c) 2019 Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

# Acknowledgements: some portions - tensorflow tutorial examples and URL below

################################################################################

import glob,os
import tensorflow as tf
import sys
sys.path.append('..')

################################################################################

from inceptionV1OnFire import construct_inceptionv1onfire
from converter import convert_to_pb
from converter import convert_to_tflite

################################################################################

if __name__ == '__main__':

    # construct and re-export model (so that is excludes the training layers)

    model = construct_inceptionv1onfire (224, 224, False)
    print("[INFO] Constructed InceptionV1-OnFire (binary, full-frame)...")

    path = "../models/InceptionV1-OnFire/inceptiononv1onfire"; # path to tflearn checkpoint including filestem
    input_layer_name = 'InputData/X'            # input layer of network
    output_layer_name= 'FullyConnected/Softmax' # output layer of network
    filename = "inceptionv1onfire.pb"           # output pb format filename

    convert_to_pb(model, path, input_layer_name,  output_layer_name, filename)
    convert_to_tflite(filename, input_layer_name,  output_layer_name)

    tf.reset_default_graph()

    print("[INFO] Constructed InceptionV1-OnFire (superpixel)...")

    model_sp = construct_inceptionv1onfire (224, 224, False)
    path_sp = "../models/SP-InceptionV1-OnFire/sp-inceptiononv1onfire"; # path to tflearn checkpoint including filestem
    filename_sp = "sp-inceptionv1onfire.pb"         # output filename

    convert_to_pb(model_sp, path_sp, input_layer_name,  output_layer_name, filename_sp)
    convert_to_tflite(filename_sp, input_layer_name,  output_layer_name)

################################################################################
