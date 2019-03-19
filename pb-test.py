################################################################################

import os
import tensorflow as tf
from tensorflow.python.framework import graph_util

################################################################################

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

################################################################################

def construct_firenet (x,y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyper-parameters

    if(training):
        network = regression(network, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

    # constuct final model

    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

################################################################################

def freeze_graph(model_folder,output_graph="frozen_model.pb"):
    # We retrieve our checkpoint fullpath
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder)

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump

    output_node_names = "FullyConnected_2/Softmax"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        op = sess.graph.get_operations()
        [print(m.values()) for m in op][1]

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,                        # The session is used to retrieve the weights
            input_graph_def,             # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        print("[INFO] output_graph:",output_graph)
        print("[INFO] all done")

################################################################################

if __name__ == '__main__':

    # construct and re-export model

    model = construct_firenet (224, 224)
    print("[INFO] Constructed FireNet ...")

    model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
    print("[INFO]  Loaded CNN network weights ...")

    print("[INFO] Re-export FireNet model ...")
    model.save("/tmp/firenet-tmp.tfl")

    freeze_graph("/tmp/firenet-tmp.tfl", "firenet.pb")

################################################################################
