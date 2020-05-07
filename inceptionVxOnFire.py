################################################################################

# Example : perform live fire detection in video using InceptionV1-OnFire,
# InceptionV3-OnFire and InceptionV4-OnFire CNN models

# Copyright (c) 2017/18 - Andrew Dunnings / Toby Breckon, Durham University, UK
# Copyright (c) 2019/20 - Ganesh Samarth / Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

################################################################################

import cv2
import os
import sys
import math

################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

################################################################################

def construct_inceptionv1onfire (x,y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018]

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
    if(training):
        pool5_7_7 = dropout(pool5_7_7, 0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    # if training then add training hyperparameters

    if(training):
        network = regression(loss, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)
    else:
        network = loss;

    model = tflearn.DNN(network, checkpoint_path='inceptiononv1onfire',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

################################################################################

def construct_inceptionv3onfire(x,y, training=False, enable_batch_norm=True):

    # build network as per architecture

    network = input_data(shape=[None, y, x, 3])

    conv1_3_3 = conv_2d(network, 32, 3, strides=2, activation='relu', name = 'conv1_3_3',padding='valid')
    conv2_3_3 = conv_2d(conv1_3_3, 32, 3, strides=1, activation='relu', name = 'conv2_3_3',padding='valid')
    conv3_3_3 = conv_2d(conv2_3_3, 64, 3, strides=2, activation='relu', name = 'conv3_3_3')

    pool1_3_3 = max_pool_2d(conv3_3_3, 3,strides=2)
    if enable_batch_norm:
        pool1_3_3 = batch_normalization(pool1_3_3)
    conv1_7_7 = conv_2d(pool1_3_3, 80,3, strides=1, activation='relu', name='conv2_7_7_s2',padding='valid')
    conv2_7_7 = conv_2d(conv1_7_7, 96,3, strides=1, activation='relu', name='conv2_7_7_s2',padding='valid')
    pool2_3_3= max_pool_2d(conv2_7_7,3,strides=2)

    inception_3a_1_1 = conv_2d(pool2_3_3,64, filter_size=1, activation='relu', name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 48, filter_size=1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 64, filter_size=[5,5],  activation='relu',name='inception_3a_3_3')


    inception_3a_5_5_reduce = conv_2d(pool2_3_3, 64, filter_size=1, activation='relu', name = 'inception_3a_5_5_reduce')
    inception_3a_5_5_asym_1 = conv_2d(inception_3a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_3a_5_5_asym_1')
    inception_3a_5_5 = conv_2d(inception_3a_5_5_asym_1, 96, filter_size=[3,3],  name = 'inception_3a_5_5')


    inception_3a_pool = avg_pool_2d(pool2_3_3, kernel_size=3, strides=1,  name='inception_3a_pool')
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    # merge the inception_3a

    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3, name='inception_3a_output')


    inception_5a_1_1 = conv_2d(inception_3a_output, 96, 1, activation='relu', name='inception_5a_1_1')

    inception_5a_3_3_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3_asym_1 = conv_2d(inception_5a_3_3_reduce, 64, filter_size=[1,7],  activation='relu',name='inception_5a_3_3_asym_1')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_asym_1,96, filter_size=[7,1],  activation='relu',name='inception_5a_3_3')


    inception_5a_5_5_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation='relu', name = 'inception_5a_5_5_reduce')
    inception_5a_5_5_asym_1 = conv_2d(inception_5a_5_5_reduce, 64, filter_size=[7,1],  name = 'inception_5a_5_5_asym_1')
    inception_5a_5_5_asym_2 = conv_2d(inception_5a_5_5_asym_1, 64, filter_size=[1,7],  name = 'inception_5a_5_5_asym_2')
    inception_5a_5_5_asym_3 = conv_2d(inception_5a_5_5_asym_2, 64, filter_size=[7,1],  name = 'inception_5a_5_5_asym_3')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_asym_3, 96, filter_size=[1,7],  name = 'inception_5a_5_5')


    inception_5a_pool = avg_pool_2d(inception_3a_output, kernel_size=3, strides=1 )
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 96, filter_size=1, activation='relu', name='inception_5a_pool_1_1')

    # merge the inception_5a__
    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], mode='concat', axis=3)



    inception_7a_1_1 = conv_2d(inception_5a_output, 80, 1, activation='relu', name='inception_7a_1_1')
    inception_7a_3_3_reduce = conv_2d(inception_5a_output, 96, filter_size=1, activation='relu', name='inception_7a_3_3_reduce')
    inception_7a_3_3_asym_1 = conv_2d(inception_7a_3_3_reduce, 96, filter_size=[1,3],  activation='relu',name='inception_7a_3_3_asym_1')
    inception_7a_3_3_asym_2 = conv_2d(inception_7a_3_3_reduce, 96, filter_size=[3,1],  activation='relu',name='inception_7a_3_3_asym_2')
    inception_7a_3_3=merge([inception_7a_3_3_asym_1,inception_7a_3_3_asym_2],mode='concat',axis=3)

    inception_7a_5_5_reduce = conv_2d(inception_5a_output, 66, filter_size=1, activation='relu', name = 'inception_7a_5_5_reduce')
    inception_7a_5_5_asym_1 = conv_2d(inception_7a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_7a_5_5_asym_1')
    inception_7a_5_5_asym_2 = conv_2d(inception_7a_3_3_asym_1, 96, filter_size=[1,3],  activation='relu',name='inception_7a_5_5_asym_2')
    inception_7a_5_5_asym_3 = conv_2d(inception_7a_3_3_asym_1, 96, filter_size=[3,1],  activation='relu',name='inception_7a_5_5_asym_3')
    inception_7a_5_5=merge([inception_7a_5_5_asym_2,inception_7a_5_5_asym_3],mode='concat',axis=3)


    inception_7a_pool = avg_pool_2d(inception_5a_output, kernel_size=3, strides=1 )
    inception_7a_pool_1_1 = conv_2d(inception_7a_pool, 96, filter_size=1, activation='relu', name='inception_7a_pool_1_1')

    # merge the inception_7a__
    inception_7a_output = merge([inception_7a_1_1, inception_7a_3_3, inception_7a_5_5, inception_7a_pool_1_1], mode='concat', axis=3)



    pool5_7_7=global_avg_pool(inception_7a_output)
    if(training):
        pool5_7_7=dropout(pool5_7_7,0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    if(training):
        network = regression(loss, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    else:
        network=loss

    model = tflearn.DNN(network, checkpoint_path='inceptionv3',
                        max_checkpoints=1, tensorboard_verbose=0)

    return model

################################################################################

# InceptionV4 : definition of inception_block_a

def inception_block_a(input_a):

    inception_a_conv1_1_1 = conv_2d(input_a,96,1,activation='relu',name='inception_a_conv1_1_1')

    inception_a_conv1_3_3_reduce = conv_2d(input_a,64,1,activation='relu',name='inception_a_conv1_3_3_reduce')
    inception_a_conv1_3_3 = conv_2d(inception_a_conv1_3_3_reduce,96,3,activation='relu',name='inception_a_conv1_3_3')

    inception_a_conv2_3_3_reduce = conv_2d(input_a,64,1,activation='relu',name='inception_a_conv2_3_3_reduce')
    inception_a_conv2_3_3_sym_1 = conv_2d(inception_a_conv2_3_3_reduce,96,3,activation='relu',name='inception_a_conv2_3_3')
    inception_a_conv2_3_3 = conv_2d(inception_a_conv2_3_3_sym_1,96,3,activation='relu',name='inception_a_conv2_3_3')

    inception_a_pool = avg_pool_2d(input_a,kernel_size=3,name='inception_a_pool',strides=1)
    inception_a_pool_1_1 = conv_2d(inception_a_pool,96,1,activation='relu',name='inception_a_pool_1_1')

    # merge inception_a

    inception_a = merge([inception_a_conv1_1_1,inception_a_conv1_3_3,inception_a_conv2_3_3,inception_a_pool_1_1],mode='concat',axis=3)

    return inception_a


################################################################################

# InceptionV4 : definition of reduction_block_a

def reduction_block_a(reduction_input_a):

    reduction_a_conv1_1_1 = conv_2d(reduction_input_a,384,3,strides=2,padding='valid',activation='relu',name='reduction_a_conv1_1_1')

    reduction_a_conv2_1_1 = conv_2d(reduction_input_a,192,1,activation='relu',name='reduction_a_conv2_1_1')
    reduction_a_conv2_3_3 = conv_2d(reduction_a_conv2_1_1,224,3,activation='relu',name='reduction_a_conv2_3_3')
    reduction_a_conv2_3_3_s2 = conv_2d(reduction_a_conv2_3_3,256,3,strides=2,padding='valid',activation='relu',name='reduction_a_conv2_3_3_s2')

    reduction_a_pool = max_pool_2d(reduction_input_a,strides=2,padding='valid',kernel_size=3,name='reduction_a_pool')

    # merge reduction_a

    reduction_a = merge([reduction_a_conv1_1_1,reduction_a_conv2_3_3_s2,reduction_a_pool],mode='concat',axis=3)

    return reduction_a

################################################################################

# InceptionV4 : definition of inception_block_b

def inception_block_b(input_b):

    inception_b_1_1 = conv_2d(input_b, 384, 1, activation='relu', name='inception_b_1_1')

    inception_b_3_3_reduce = conv_2d(input_b, 192, filter_size=1, activation='relu', name='inception_b_3_3_reduce')
    inception_b_3_3_asym_1 = conv_2d(inception_b_3_3_reduce, 224, filter_size=[1,7],  activation='relu',name='inception_b_3_3_asym_1')
    inception_b_3_3 = conv_2d(inception_b_3_3_asym_1, 256, filter_size=[7,1],  activation='relu',name='inception_b_3_3')


    inception_b_5_5_reduce = conv_2d(input_b, 192, filter_size=1, activation='relu', name = 'inception_b_5_5_reduce')
    inception_b_5_5_asym_1 = conv_2d(inception_b_5_5_reduce, 192, filter_size=[7,1],  name = 'inception_b_5_5_asym_1')
    inception_b_5_5_asym_2 = conv_2d(inception_b_5_5_asym_1, 224, filter_size=[1,7],  name = 'inception_b_5_5_asym_2')
    inception_b_5_5_asym_3 = conv_2d(inception_b_5_5_asym_2, 224, filter_size=[7,1],  name = 'inception_b_5_5_asym_3')
    inception_b_5_5 = conv_2d(inception_b_5_5_asym_3, 256, filter_size=[1,7],  name = 'inception_b_5_5')


    inception_b_pool = avg_pool_2d(input_b, kernel_size=3, strides=1 )
    inception_b_pool_1_1 = conv_2d(inception_b_pool, 128, filter_size=1, activation='relu', name='inception_b_pool_1_1')

    # merge the inception_b

    inception_b_output = merge([inception_b_1_1, inception_b_3_3, inception_b_5_5, inception_b_pool_1_1], mode='concat', axis=3)

    return inception_b_output

################################################################################

# InceptionV4 : definition of reduction_block_b

def reduction_block_b(reduction_input_b):

    reduction_b_1_1 = conv_2d(reduction_input_b,192,1,activation='relu',name='reduction_b_1_1')
    reduction_b_1_3 = conv_2d(reduction_b_1_1,192,3,strides=2,padding='valid',name='reduction_b_1_3')

    reduction_b_3_3_reduce = conv_2d(reduction_input_b, 256, filter_size=1, activation='relu', name='reduction_b_3_3_reduce')
    reduction_b_3_3_asym_1 = conv_2d(reduction_b_3_3_reduce, 256, filter_size=[1,7],  activation='relu',name='reduction_b_3_3_asym_1')
    reduction_b_3_3_asym_2 = conv_2d(reduction_b_3_3_asym_1, 320, filter_size=[7,1],  activation='relu',name='reduction_b_3_3_asym_2')
    reduction_b_3_3=conv_2d(reduction_b_3_3_asym_2,320,3,strides=2,activation='relu',padding='valid',name='reduction_b_3_3')

    reduction_b_pool = max_pool_2d(reduction_input_b,kernel_size=3,strides=2,padding='valid')

    # merge the reduction_b

    reduction_b_output = merge([reduction_b_1_3,reduction_b_3_3,reduction_b_pool],mode='concat',axis=3)

    return reduction_b_output

################################################################################

# InceptionV4 : defintion of inception_block_c

def inception_block_c(input_c):
    inception_c_1_1 = conv_2d(input_c, 256, 1, activation='relu', name='inception_c_1_1')
    inception_c_3_3_reduce = conv_2d(input_c, 384, filter_size=1, activation='relu', name='inception_c_3_3_reduce')
    inception_c_3_3_asym_1 = conv_2d(inception_c_3_3_reduce, 256, filter_size=[1,3],  activation='relu',name='inception_c_3_3_asym_1')
    inception_c_3_3_asym_2 = conv_2d(inception_c_3_3_reduce, 256, filter_size=[3,1],  activation='relu',name='inception_c_3_3_asym_2')
    inception_c_3_3=merge([inception_c_3_3_asym_1,inception_c_3_3_asym_2],mode='concat',axis=3)

    inception_c_5_5_reduce = conv_2d(input_c, 384, filter_size=1, activation='relu', name = 'inception_c_5_5_reduce')
    inception_c_5_5_asym_1 = conv_2d(inception_c_5_5_reduce, 448, filter_size=[1,3],  name = 'inception_c_5_5_asym_1')
    inception_c_5_5_asym_2 = conv_2d(inception_c_5_5_asym_1, 512, filter_size=[3,1],  activation='relu',name='inception_c_5_5_asym_2')
    inception_c_5_5_asym_3 = conv_2d(inception_c_5_5_asym_2, 256, filter_size=[1,3],  activation='relu',name='inception_c_5_5_asym_3')

    inception_c_5_5_asym_4 = conv_2d(inception_c_5_5_asym_2, 256, filter_size=[3,1],  activation='relu',name='inception_c_5_5_asym_4')
    inception_c_5_5=merge([inception_c_5_5_asym_4,inception_c_5_5_asym_3],mode='concat',axis=3)


    inception_c_pool = avg_pool_2d(input_c, kernel_size=3, strides=1 )
    inception_c_pool_1_1 = conv_2d(inception_c_pool, 256, filter_size=1, activation='relu', name='inception_c_pool_1_1')

    # merge the inception_c

    inception_c_output = merge([inception_c_1_1, inception_c_3_3, inception_c_5_5, inception_c_pool_1_1], mode='concat', axis=3)

    return inception_c_output

################################################################################

def construct_inceptionv4onfire(x,y, training=True, enable_batch_norm=True):

    network = input_data(shape=[None, y, x, 3])

    #stem of inceptionV4

    conv1_3_3 = conv_2d(network,32,3,strides=2,activation='relu',name='conv1_3_3_s2',padding='valid')
    conv2_3_3 = conv_2d(conv1_3_3,32,3,activation='relu',name='conv2_3_3')
    conv3_3_3 = conv_2d(conv2_3_3,64,3,activation='relu',name='conv3_3_3')
    b_conv_1_pool = max_pool_2d(conv3_3_3,kernel_size=3,strides=2,padding='valid',name='b_conv_1_pool')
    if enable_batch_norm:
        b_conv_1_pool = batch_normalization(b_conv_1_pool)
    b_conv_1_conv = conv_2d(conv3_3_3,96,3,strides=2,padding='valid',activation='relu',name='b_conv_1_conv')
    b_conv_1 = merge([b_conv_1_conv,b_conv_1_pool],mode='concat',axis=3)

    b_conv4_1_1 = conv_2d(b_conv_1,64,1,activation='relu',name='conv4_3_3')
    b_conv4_3_3 = conv_2d(b_conv4_1_1,96,3,padding='valid',activation='relu',name='conv5_3_3')

    b_conv4_1_1_reduce = conv_2d(b_conv_1,64,1,activation='relu',name='b_conv4_1_1_reduce')
    b_conv4_1_7 = conv_2d(b_conv4_1_1_reduce,64,[1,7],activation='relu',name='b_conv4_1_7')
    b_conv4_7_1 = conv_2d(b_conv4_1_7,64,[7,1],activation='relu',name='b_conv4_7_1')
    b_conv4_3_3_v = conv_2d(b_conv4_7_1,96,3,padding='valid',name='b_conv4_3_3_v')
    b_conv_4 = merge([b_conv4_3_3_v, b_conv4_3_3],mode='concat',axis=3)

    b_conv5_3_3 = conv_2d(b_conv_4,192,3,padding='valid',activation='relu',name='b_conv5_3_3',strides=2)
    b_pool5_3_3 = max_pool_2d(b_conv_4,kernel_size=3,padding='valid',strides=2,name='b_pool5_3_3')
    if enable_batch_norm:
        b_pool5_3_3 = batch_normalization(b_pool5_3_3)
    b_conv_5 = merge([b_conv5_3_3,b_pool5_3_3],mode='concat',axis=3)
    net = b_conv_5

    # inceptionV4 modules

    net=inception_block_a(net)

    net=inception_block_b(net)

    net=inception_block_c(net)

    pool5_7_7=global_avg_pool(net)
    if(training):
        pool5_7_7=dropout(pool5_7_7,0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    if(training):
        network = regression(loss, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    else:
        network=loss

    model = tflearn.DNN(network, checkpoint_path='inceptionv4onfire',
                        max_checkpoints=1, tensorboard_verbose=0)

    return model

################################################################################

if __name__ == '__main__':

################################################################################

    # parse command line arguments

    import argparse

    parser = argparse.ArgumentParser(description='Perform InceptionV1/V3/V4 fire detection on incoming video')
    parser.add_argument("-m", "--model_to_use", type=int, help="specify model to use", default=1, choices={1, 3, 4})
    parser.add_argument('video_file', metavar='video_file', type=str, help='specify video file')
    args = parser.parse_args()

    #   construct and display model

    print("Constructing InceptionV" + str(args.model_to_use) + "-OnFire ...")

    if (args.model_to_use == 1):

        # use InceptionV1-OnFire CNN model - [Dunning/Breckon, 2018]

        model = construct_inceptionv1onfire (224, 224, training=False)
        # also work around typo in naming of original models for V1 models [Dunning/Breckon, 2018] "...iononv ..."
        model.load(os.path.join("models/InceptionV1-OnFire", "inceptiononv1onfire"),weights_only=True)

    elif (args.model_to_use == 3):

        # use InceptionV3-OnFire CNN model - [Samarth/Bhowmik/Breckon, 2019]
        # N.B. weights_only=False as we are using Batch Normalization, and need those weights loaded also

        model = construct_inceptionv3onfire (224, 224, training=False)
        model.load(os.path.join("models/InceptionV3-OnFire", "inceptionv3onfire"),weights_only=False)

    elif (args.model_to_use == 4):

        # use InceptionV4-OnFire CNN model - [Samarth/Bhowmik/Breckon, 2019]
        # N.B. weights_only=False as we are using Batch Normalization, and need those weights loaded also

        model = construct_inceptionv4onfire (224, 224, training=False)
        model.load(os.path.join("models/InceptionV4-OnFire", "inceptionv4onfire"),weights_only=False)

    print("Loaded CNN network weights ...")

################################################################################

    # network input sizes

    rows = 224
    cols = 224

    # display and loop settings

    windowName = "Live Fire Detection - InceptionV" + str(args.model_to_use) + "-OnFire";
    keepProcessing = True;

################################################################################

    # load video file from first command line argument

    video = cv2.VideoCapture(args.video_file)
    print("Loaded video ...")

    # create window

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);

    # get video properties

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_time = round(1000/fps);

    while (keepProcessing):

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount();

        # get video frame from file, handle end of file

        ret, frame = video.read()
        if not ret:
            print("... end of video file reached");
            break;

        # re-size image to network input size and perform prediction

        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

        # perform prediction on the image frame which is:
        # - an image (tensor) of dimension 224 x 224 x 3
        # - a 3 channel colour image with channel ordering BGR (not RGB)
        # - un-normalised (i.e. pixel range going into network is 0->255)

        output = model.predict([small_frame])

        # label image based on prediction

        if round(output[0][0]) == 1: # equiv. to 0.5 threshold in [Dunnings / Breckon, 2018],  [Samarth/Bhowmik/Breckon, 2019] test code
            cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
            cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
        else:
            cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
            cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);

        # stop the timer and convert to ms. (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

        # image display and key handling

        cv2.imshow(windowName, frame);

        # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

        key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
        if (key == ord('x')):
            keepProcessing = False;
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

################################################################################
