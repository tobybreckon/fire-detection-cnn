#################################################################################

#InceptionV3OnFire stacked with one bloack each of Inception A, B and C with filters reduced


#################################################################################

from __future__ import division, print_function, absolute_import

import tflearn
import h5py
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d,global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression


def construct_inceptionv3onfire(x,y,normalization_method,optimizer,dropout_value,activation,training=True):
    # Build network
    # 224 x 224 original size
    network = input_data(shape=[None, y, x, 3])
    conv1_3_3 =conv_2d(network, 32, 3, strides=2, activation=activation, name = 'conv1_3_3',padding='valid')
    conv2_3_3 =conv_2d(conv1_3_3, 32, 3, strides=1, activation=activation, name = 'conv2_3_3',padding='valid')
    conv3_3_3 =conv_2d(conv2_3_3, 64, 3, strides=2, activation=activation, name = 'conv3_3_3')

    pool1_3_3 = max_pool_2d(conv3_3_3, 3,strides=2)
    pool1_3_3 = globals()[normalization_method](pool1_3_3)
    conv1_7_7 =conv_2d(pool1_3_3, 80,3, strides=1, activation=activation, name='conv2_7_7_s2',padding='valid')
    conv2_7_7 =conv_2d(conv1_7_7, 96,3, strides=1, activation=activation, name='conv2_7_7_s2',padding='valid')
    pool2_3_3=max_pool_2d(conv2_7_7,3,strides=2)

    inception_3a_1_1 = conv_2d(pool2_3_3,64, filter_size=1, activation=activation, name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 48, filter_size=1, activation=activation, name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 64, filter_size=[5,5],  activation=activation,name='inception_3a_3_3')


    inception_3a_5_5_reduce = conv_2d(pool2_3_3, 64, filter_size=1, activation=activation, name = 'inception_3a_5_5_reduce')
    inception_3a_5_5_asym_1 = conv_2d(inception_3a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_3a_5_5_asym_1')
    inception_3a_5_5 = conv_2d(inception_3a_5_5_asym_1, 96, filter_size=[3,3],  name = 'inception_3a_5_5')


    inception_3a_pool = avg_pool_2d(pool2_3_3, kernel_size=3, strides=1,  name='inception_3a_pool')
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation=activation, name='inception_3a_pool_1_1')

    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3, name='inception_3a_output')


    
   
    inception_5a_1_1 = conv_2d(inception_3a_output, 96, 1, activation=activation, name='inception_5a_1_1')

    inception_5a_3_3_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation=activation, name='inception_5a_3_3_reduce')
    inception_5a_3_3_asym_1 = conv_2d(inception_5a_3_3_reduce, 64, filter_size=[1,7],  activation=activation,name='inception_5a_3_3_asym_1')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_asym_1,96, filter_size=[7,1],  activation=activation,name='inception_5a_3_3')


    inception_5a_5_5_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation=activation, name = 'inception_5a_5_5_reduce')
    inception_5a_5_5_asym_1 = conv_2d(inception_5a_5_5_reduce, 64, filter_size=[7,1],  name = 'inception_5a_5_5_asym_1')
    inception_5a_5_5_asym_2 = conv_2d(inception_5a_5_5_asym_1, 64, filter_size=[1,7],  name = 'inception_5a_5_5_asym_2')
    inception_5a_5_5_asym_3 = conv_2d(inception_5a_5_5_asym_2, 64, filter_size=[7,1],  name = 'inception_5a_5_5_asym_3')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_asym_3, 96, filter_size=[1,7],  name = 'inception_5a_5_5')


    inception_5a_pool = avg_pool_2d(inception_3a_output, kernel_size=3, strides=1 )
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 96, filter_size=1, activation=activation, name='inception_5a_pool_1_1')

    # merge the inception_5a__
    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], mode='concat', axis=3)

   

    inception_7a_1_1 = conv_2d(inception_5a_output, 80, 1, activation=activation, name='inception_7a_1_1')
    inception_7a_3_3_reduce = conv_2d(inception_5a_output, 96, filter_size=1, activation=activation, name='inception_7a_3_3_reduce')
    inception_7a_3_3_asym_1 = conv_2d(inception_7a_3_3_reduce, 96, filter_size=[1,3],  activation=activation,name='inception_7a_3_3_asym_1')
    inception_7a_3_3_asym_2 = conv_2d(inception_7a_3_3_reduce, 96, filter_size=[3,1],  activation=activation,name='inception_7a_3_3_asym_2')
    inception_7a_3_3=merge([inception_7a_3_3_asym_1,inception_7a_3_3_asym_2],mode='concat',axis=3)

    inception_7a_5_5_reduce = conv_2d(inception_5a_output, 66, filter_size=1, activation=activation, name = 'inception_7a_5_5_reduce')
    inception_7a_5_5_asym_1 = conv_2d(inception_7a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_7a_5_5_asym_1')
    inception_7a_5_5_asym_2 = conv_2d(inception_7a_3_3_asym_1, 96, filter_size=[1,3],  activation=activation,name='inception_7a_5_5_asym_2')
    inception_7a_5_5_asym_3 = conv_2d(inception_7a_3_3_asym_1, 96, filter_size=[3,1],  activation=activation,name='inception_7a_5_5_asym_3')
    inception_7a_5_5=merge([inception_7a_5_5_asym_2,inception_7a_5_5_asym_3],mode='concat',axis=3)


    inception_7a_pool = avg_pool_2d(inception_5a_output, kernel_size=3, strides=1 )
    inception_7a_pool_1_1 = conv_2d(inception_7a_pool, 96, filter_size=1, activation=activation, name='inception_7a_pool_1_1')

    # merge the inception_7a__
    inception_7a_output = merge([inception_7a_1_1, inception_7a_3_3, inception_7a_5_5, inception_7a_pool_1_1], mode='concat', axis=3)

  

    pool5_7_7=global_avg_pool(inception_7a_output)
    pool5_7_7=dropout(pool5_7_7,dropout_value)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    if(training):
        network = regression(loss, optimizer=optimizer,
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    else:
        network=loss

    model = tflearn.DNN(network, checkpoint_path='inceptionv3',
                        max_checkpoints=1, tensorboard_verbose=0)

    return model
