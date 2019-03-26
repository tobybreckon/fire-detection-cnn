################################################################################

# Example : perform conversion of FireNet and InceptionV1-OnFire tflearn models to
# TensorFlow protocol buffer (.pb) format files (for import into other tools, example OpenCV DNN)

# Copyright (c) 2019 Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

# Acknowledgements: some portions - tensorflow tutorial examples and derivatives

################################################################################

import os

################################################################################

# import tensorflow api

import tensorflow as tf
from tensorflow.python.framework import graph_util

################################################################################

# import tflearn api

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

################################################################################

from firenet import construct_firenet

################################################################################

if __name__ == '__main__':

    # construct and re-export model (so that is excludes the training layers)

    model = construct_firenet (224, 224, False)
    print("[INFO] Constructed FireNet ...")

    model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
    print("[INFO] Loaded CNN network weights for FireNet ...")

    print("[INFO] Re-export FireNet model ...")
    del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
    model.save("firenet-tmp.tfl")
    # os.remove("firenet-tmp.tfl.data-00000-of-00001")

    # hack 2 - from https://stackoverflow.com/questions/34343259/is-there-an-example-on-how-to-generate-protobuf-files-holding-trained-tensorflow

    input_checkpoint = "firenet-tmp.tfl"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', True)
    sess = tf.Session();
    saver.restore(sess, input_checkpoint)
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    # print out all layers to find name of output

    # op = sess.graph.get_operations()
    # [print(m.values()) for m in op][1]

    # freeze and removes nodes which are not related to feedforward prediction

    minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["FullyConnected_2/Softmax"])

    tf.train.write_graph(minimal_graph, '.', 'minimal_graph.pb', as_text=False)
    tf.train.write_graph(minimal_graph, '.', 'minimal_graph.txt', as_text=True)

    # write model to logs dir so we can visualize it as:
    # tensorboard --logdir="logs"

    writer = tf.summary.FileWriter('logs', minimal_graph)
    writer.close()

    # perform test inference using OpenCV

    import cv2

    # Load a model imported from Tensorflow
    tensorflowNet = cv2.dnn.readNetFromTensorflow('minimal_graph.pb', 'minimal_graph.txt');

    # Input image
    img = cv2.imread('/tmp/fire.jpg')

    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(224, 224), swapRB=False, crop=False))

    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()

    # Loop on the outputs
    for detection in networkOutput[0,0]:
        print(detection)

    # Show the image

    cv2.imshow('Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # clean up temp files

    # TODO

################################################################################
