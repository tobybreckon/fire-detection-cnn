################################################################################

# Example : perform conversion of inceptionV1OnFire tflearn model to TensorFlow protocol
# buffer (.pb) format files (for import into other tools, example OpenCV DNN)

# Copyright (c) 2019 Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

# Acknowledgements: some portions - tensorflow tutorial examples and URL below

################################################################################

import glob,os

################################################################################

# import opencv

import cv2

################################################################################

from inceptionV1OnFire import construct_inceptionv1onfire
from convertor import convert_to_pb

################################################################################

if __name__ == '__main__':

    # construct and re-export model (so that is excludes the training layers)

    model = construct_inceptionv1onfire (224, 224, True)
    print("[INFO] Constructed InceptionV1-OnFire ...")

    path = "models/InceptionV1-OnFire/inceptiononv1onfire"; # path to tflearn checkpoint including filestem
    input_layer_name = 'InputData/X'            # input layer of network
    output_layer_name= 'FullyConnected/Softmax' # output layer of network
    pbfilename = "inceptionv1onfire.pb"         # output pb format filename

    convert_to_pb(model, path, input_layer_name,  output_layer_name, pbfilename)

    ##############################

    # perform test inference using OpenCV

    print("[INFO] test inceptionV1OnFire model " + pbfilename + " with OpenCV ...")

    # Load a model imported from Tensorflow

    tensorflowNet = cv2.dnn.readNetFromTensorflow(pbfilename);

    # Input image

    img = cv2.imread('images/slic-stages.png')
    img = img[0:600,0:396]; # extract left part of example containing fire

    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(224, 224), swapRB=False, crop=False))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()
    print("[INFO] example image - Fire: " + str(networkOutput[0][0]) + " Not Fire: " + str(networkOutput[0][1]))
    print("[INFO] - press any key to exit")

    # Show the image with a rectagle surrounding the detected objects
    cv2.imshow('Example Detection', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

################################################################################
