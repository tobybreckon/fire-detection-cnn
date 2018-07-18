from __future__ import division, print_function, absolute_import

import os
import tflearn
import tensorflow as tf
import h5py
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from evaluate import evaluate_in_train

tf.logging.set_verbosity(tf.logging.ERROR)

def construct_inceptionv1onfire (x,y):

    # Build network
    # 227 x 227 original size
    network = input_data(shape=[None, y, x, 3])

    conv1_7_7 = conv_2d(network, 64, 5, strides=2, activation='relu', name = 'conv1_7_7_s2')

    pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)

    conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 128,3, activation='relu', name='conv2_3_3')

    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

    inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

    #merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

    pool5_7_7 = avg_pool_2d(inception_4a_output, kernel_size=5, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')
    network = regression(loss, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    model = tflearn.DNN(network, checkpoint_path='googlenetv2',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

def train_inceptionv1onfire (data, epochs, validation_size, x, y):

    directory_path = "/home/jrnf24/data/"

    # Load hdf5 dataset
    h5f = h5py.File('{}{}.h5'.format(directory_path,data), 'r')
    X = h5f['X']
    Y = h5f['Y']

    model = construct_googlenetv2(x,y)

    model.fit(X, Y, n_epoch=epochs, validation_set=validation_size, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenetv2')

    h5f.close()

def train_with_eval_googlenetv2 (data, eval_datasets):

    directory_path = "/home/jrnf24/data/"

    # Load hdf5 dataset
    h5f = h5py.File('{}{}.h5'.format(directory_path,data), 'r')
    X = h5f['X']
    Y = h5f['Y']

    model = construct_googlenetv2(224,224)

    model.fit(X, Y, n_epoch=5, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenetv2')

    evaluate_in_train("5", eval_datasets, model)

    model.fit(X, Y, n_epoch=5, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenetv2')

    evaluate_in_train("10", eval_datasets, model)

    model.fit(X, Y, n_epoch=10, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenetv2')

    evaluate_in_train("20", eval_datasets, model)

    model.fit(X, Y, n_epoch=10, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenetv2')

    evaluate_in_train("30", eval_datasets, model)

    h5f.close()

def retrain_with_eval_inceptionv1onfire (data, eval_datasets):

    directory_path = "/home/jrnf24/data/"

    # Load hdf5 dataset
    h5f = h5py.File('{}{}.h5'.format(directory_path,data), 'r')
    X = h5f['X']
    Y = h5f['Y']

    model = construct_googlenetv2(224,224)

    for filename in os.listdir("/home/jrnf24/trained/G2/"):
        if ".meta" in filename:
            model_name = filename[:-5]

    model.load(os.path.join("/home/jrnf24/trained/G2/",model_name))

    model.fit(X, Y, n_epoch=20, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenetv2')

    evaluate_in_train("50", eval_datasets, model)

    model.fit(X, Y, n_epoch=25, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenetv2')

    evaluate_in_train("75", eval_datasets, model)

    model.fit(X, Y, n_epoch=25, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenetv2')

    evaluate_in_train("100", eval_datasets, model)

    model.fit(X, Y, n_epoch=25, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenetv2')

    evaluate_in_train("125", eval_datasets, model)

    model.fit(X, Y, n_epoch=25, validation_set=0.3, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenetv2')

    evaluate_in_train("150", eval_datasets, model)

    h5f.close()
