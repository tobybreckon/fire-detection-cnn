################################################################################

# Example : perform conversion from tflearn checkpoint format to TensorFlow
# protocol buffer (.pb) binary format and also .tflite files (for import into other tools)

# Copyright (c) 2019 Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

# Acknowledgements: some portions - tensorflow tutorial examples and URL below

################################################################################

import glob,os

################################################################################

# import tensorflow api

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.compat.v1.lite import TFLiteConverter

################################################################################

# import tflearn api

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

################################################################################
# convert a loaded model definition by loading a checkpoint from a given path
# retaining the network between the specified input and output layers
# outputs to pbfilename as a binary .pb protocol buffer format file

# e.g. for FireNet
#    model = construct_firenet (224, 224, False)
#    path = "models/FireNet/firenet"; # path to tflearn checkpoint including filestem
#    input_layer_name = 'InputData/X'                  # input layer of network
#    output_layer_name= 'FullyConnected_2/Softmax'     # output layer of network
#    filename = "firenet.pb"                              # output filename

def convert_to_pb(model, path, input_layer_name,  output_layer_name, pbfilename, verbose=False):

  model.load(path,weights_only=True)
  print("[INFO] Loaded CNN network weights from " + path + " ...")

  print("[INFO] Re-export model ...")
  del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
  model.save("model-tmp.tfl")

  # taken from: https://stackoverflow.com/questions/34343259/is-there-an-example-on-how-to-generate-protobuf-files-holding-trained-tensorflow

  print("[INFO] Re-import model ...")

  input_checkpoint = "model-tmp.tfl"
  saver = tf.train.import_meta_graph(input_checkpoint + '.meta', True)
  sess = tf.Session();
  saver.restore(sess, input_checkpoint)

  # print out all layers to find name of output

  if (verbose):
      op = sess.graph.get_operations()
      [print(m.values()) for m in op][1]

  print("[INFO] Freeze model to " +  pbfilename + " ...")

  # freeze and removes nodes which are not related to feedforward prediction

  minimal_graph = convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_layer_name])

  graph_def = optimize_for_inference_lib.optimize_for_inference(minimal_graph, [input_layer_name], [output_layer_name], tf.float32.as_datatype_enum)
  graph_def = TransformGraph(graph_def, [input_layer_name], [output_layer_name], ["sort_by_execution_order"])

  with tf.gfile.GFile(pbfilename, 'wb') as f:
      f.write(graph_def.SerializeToString())

  # write model to logs dir so we can visualize it as:
  # tensorboard --logdir="logs"

  if (verbose):
      writer = tf.summary.FileWriter('logs', graph_def)
      writer.close()

  # tidy up tmp files

  for f in glob.glob("model-tmp.tfl*"):
      os.remove(f)

  os.remove('checkpoint')

################################################################################
# convert a  binary .pb protocol buffer format model to tflite format

# e.g. for FireNet
#    pbfilename = "firenet.pb"
#    input_layer_name = 'InputData/X'                  # input layer of network
#    output_layer_name= 'FullyConnected_2/Softmax'     # output layer of network

def convert_to_tflite(pbfilename, input_layer_name,  output_layer_name,
                        input_tensor_dim_x=224, input_tensor_dim_y=224, input_tensor_channels=3):

  input_tensor={input_layer_name:[1,input_tensor_dim_x,input_tensor_dim_y,input_tensor_channels]}

  print("[INFO] tflite model to " +  pbfilename.replace(".pb",".tflite") + " ...")

  converter = tf.lite.TFLiteConverter.from_frozen_graph(pbfilename, [input_layer_name], [output_layer_name], input_tensor)
  tflite_model = converter.convert()
  open(pbfilename.replace(".pb",".tflite"), "wb").write(tflite_model)

################################################################################
