
################################################################################

# Example : test all generated (.pb) format files with OpenCV DNN module

# Copyright (c) 2019 Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

# Acknowledgements: some portions - tensorflow tutorial examples and URL below

################################################################################

import glob,os
import cv2

################################################################################

# perform test inference using OpenCV

print("Using OpenCV: " + cv2.__version__ + " (requires version > 4.1.0-pre)");

print();

for pbfilename in glob.glob("*.pb"):

    print("[INFO] test model from: " + pbfilename + " with OpenCV ...")

    # Load a model imported from Tensorflow

    tensorflowNet = cv2.dnn.readNetFromTensorflow(pbfilename);

    # Input image

    img = cv2.imread('../images/slic-stages.png')
    img = img[0:600,0:396]; # extract left part of example containing fire

    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(224, 224), swapRB=False, crop=False))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()
    print("[INFO] example image - Fire: " + str(networkOutput[0][0]) + " Not Fire: " + str(networkOutput[0][1]))
    print();

################################################################################
