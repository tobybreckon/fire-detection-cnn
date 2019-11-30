################################################################################

# Example : perform conversion of FireNet tflearn model to TensorFlow protocol
# buffer (.pb) binary format and tflife format files (for import into other tools, example OpenCV)

# Copyright (c) 2019 Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

# Acknowledgements: some portions - tensorflow tutorial examples and URL below

################################################################################

import glob,os
import sys
sys.path.append('..')

################################################################################

from firenet import construct_firenet
from converter import convert_to_pb
from converter import convert_to_tflite

################################################################################

if __name__ == '__main__':

    # construct and re-export model (so that is excludes the training layers)

    model = construct_firenet (224, 224, False)
    print("[INFO] Constructed FireNet ...")

    path = "../models/FireNet/firenet"; # path to tflearn checkpoint including filestem
    input_layer_name = 'InputData/X'                  # input layer of network
    output_layer_name= 'FullyConnected_2/Softmax'     # output layer of network
    filename = "firenet.pb"                           # output pb format filename

    convert_to_pb(model, path, input_layer_name,  output_layer_name, filename)
    convert_to_tflite(filename, input_layer_name,  output_layer_name)

################################################################################
