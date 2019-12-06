
#################################################################################

#InceptionV4OnFire stacked with one bloack each of Inception A, B and C without filters being reduced


#################################################################################



from __future__ import division, print_function, absolute_import

import tflearn
import h5py
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d,global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

def inception_block_a(input_a,activation):
    inception_a_conv1_1_1=conv_2d(input_a,96,1,activation=activation,name='inception_a_conv1_1_1')

    inception_a_conv1_3_3_reduce=conv_2d(input_a,64,1,activation=activation,name='inception_a_conv1_3_3_reduce')
    inception_a_conv1_3_3=conv_2d(inception_a_conv1_3_3_reduce,96,3,activation=activation,name='inception_a_conv1_3_3')

    inception_a_conv2_3_3_reduce=conv_2d(input_a,64,1,activation=activation,name='inception_a_conv2_3_3_reduce')
    inception_a_conv2_3_3_sym_1=conv_2d(inception_a_conv2_3_3_reduce,96,3,activation=activation,name='inception_a_conv2_3_3')
    inception_a_conv2_3_3=conv_2d(inception_a_conv2_3_3_sym_1,96,3,activation=activation,name='inception_a_conv2_3_3')

    inception_a_pool=avg_pool_2d(input_a,kernel_size=3,name='inception_a_pool',strides=1)
    inception_a_pool_1_1=conv_2d(inception_a_pool,96,1,activation=activation,name='inception_a_pool_1_1')
    inception_a=merge([inception_a_conv1_1_1,inception_a_conv1_3_3,inception_a_conv2_3_3,inception_a_pool_1_1],mode='concat',axis=3)

    return inception_a



def reduction_block_a(reduction_input_a,activation):
    reduction_a_conv1_1_1=conv_2d(reduction_input_a,384,3,strides=2,padding='valid',activation=activation,name='reduction_a_conv1_1_1')

    reduction_a_conv2_1_1=conv_2d(reduction_input_a,192,1,activation=activation,name='reduction_a_conv2_1_1')
    reduction_a_conv2_3_3=conv_2d(reduction_a_conv2_1_1,224,3,activation=activation,name='reduction_a_conv2_3_3')
    reduction_a_conv2_3_3_s2=conv_2d(reduction_a_conv2_3_3,256,3,strides=2,padding='valid',activation=activation,name='reduction_a_conv2_3_3_s2')

    reduction_a_pool=max_pool_2d(reduction_input_a,strides=2,padding='valid',kernel_size=3,name='reduction_a_pool')
    reduction_a=merge([reduction_a_conv1_1_1,reduction_a_conv2_3_3_s2,reduction_a_pool],mode='concat',axis=3)

    return reduction_a


def inception_block_b(input_b,activation):
    inception_b_1_1 = conv_2d(input_b, 384, 1, activation=activation, name='inception_b_1_1')

    inception_b_3_3_reduce = conv_2d(input_b, 192, filter_size=1, activation=activation, name='inception_b_3_3_reduce')
    inception_b_3_3_asym_1 = conv_2d(inception_b_3_3_reduce, 224, filter_size=[1,7],  activation=activation,name='inception_b_3_3_asym_1')
    inception_b_3_3 = conv_2d(inception_b_3_3_asym_1, 256, filter_size=[7,1],  activation=activation,name='inception_b_3_3')


    inception_b_5_5_reduce = conv_2d(input_b, 192, filter_size=1, activation=activation, name = 'inception_b_5_5_reduce')
    inception_b_5_5_asym_1 = conv_2d(inception_b_5_5_reduce, 192, filter_size=[7,1],  name = 'inception_b_5_5_asym_1')
    inception_b_5_5_asym_2 = conv_2d(inception_b_5_5_asym_1, 224, filter_size=[1,7],  name = 'inception_b_5_5_asym_2')
    inception_b_5_5_asym_3 = conv_2d(inception_b_5_5_asym_2, 224, filter_size=[7,1],  name = 'inception_b_5_5_asym_3')
    inception_b_5_5 = conv_2d(inception_b_5_5_asym_3, 256, filter_size=[1,7],  name = 'inception_b_5_5')


    inception_b_pool = avg_pool_2d(input_b, kernel_size=3, strides=1 )
    inception_b_pool_1_1 = conv_2d(inception_b_pool, 128, filter_size=1, activation=activation, name='inception_b_pool_1_1')

    # merge the inception_b__
    inception_b_output = merge([inception_b_1_1, inception_b_3_3, inception_b_5_5, inception_b_pool_1_1], mode='concat', axis=3)

    return inception_b_output

def reduction_block_b(reduction_input_b,activation):
    reduction_b_1_1=conv_2d(reduction_input_b,192,1,activation=activation,name='reduction_b_1_1')
    reduction_b_1_3=conv_2d(reduction_b_1_1,192,3,strides=2,padding='valid',name='reduction_b_1_3')

    reduction_b_3_3_reduce = conv_2d(reduction_input_b, 256, filter_size=1, activation=activation, name='reduction_b_3_3_reduce')
    reduction_b_3_3_asym_1 = conv_2d(reduction_b_3_3_reduce, 256, filter_size=[1,7],  activation=activation,name='reduction_b_3_3_asym_1')
    reduction_b_3_3_asym_2 = conv_2d(reduction_b_3_3_asym_1, 320, filter_size=[7,1],  activation=activation,name='reduction_b_3_3_asym_2')
    reduction_b_3_3=conv_2d(reduction_b_3_3_asym_2,320,3,strides=2,activation=activation,padding='valid',name='reduction_b_3_3')

    reduction_b_pool=max_pool_2d(reduction_input_b,kernel_size=3,strides=2,padding='valid')

    reduction_b_output=merge([reduction_b_1_3,reduction_b_3_3,reduction_b_pool],mode='concat',axis=3)

    return reduction_b_output


def inception_block_c(input_c,activation):
    inception_c_1_1 = conv_2d(input_c, 256, 1, activation=activation, name='inception_c_1_1')
    inception_c_3_3_reduce = conv_2d(input_c, 384, filter_size=1, activation=activation, name='inception_c_3_3_reduce')
    inception_c_3_3_asym_1 = conv_2d(inception_c_3_3_reduce, 256, filter_size=[1,3],  activation=activation,name='inception_c_3_3_asym_1')
    inception_c_3_3_asym_2 = conv_2d(inception_c_3_3_reduce, 256, filter_size=[3,1],  activation=activation,name='inception_c_3_3_asym_2')
    inception_c_3_3=merge([inception_c_3_3_asym_1,inception_c_3_3_asym_2],mode='concat',axis=3)

    inception_c_5_5_reduce = conv_2d(input_c, 384, filter_size=1, activation=activation, name = 'inception_c_5_5_reduce')
    inception_c_5_5_asym_1 = conv_2d(inception_c_5_5_reduce, 448, filter_size=[1,3],  name = 'inception_c_5_5_asym_1')
    inception_c_5_5_asym_2 = conv_2d(inception_c_5_5_asym_1, 512, filter_size=[3,1],  activation=activation,name='inception_c_5_5_asym_2')
    inception_c_5_5_asym_3 = conv_2d(inception_c_5_5_asym_2, 256, filter_size=[1,3],  activation=activation,name='inception_c_5_5_asym_3')

    inception_c_5_5_asym_4 = conv_2d(inception_c_5_5_asym_2, 256, filter_size=[3,1],  activation=activation,name='inception_c_5_5_asym_4')
    inception_c_5_5=merge([inception_c_5_5_asym_4,inception_c_5_5_asym_3],mode='concat',axis=3)


    inception_c_pool = avg_pool_2d(input_c, kernel_size=3, strides=1 )
    inception_c_pool_1_1 = conv_2d(inception_c_pool, 256, filter_size=1, activation=activation, name='inception_c_pool_1_1')

    # merge the inception_c__
    inception_c_output = merge([inception_c_1_1, inception_c_3_3, inception_c_5_5, inception_c_pool_1_1], mode='concat', axis=3)

    return inception_c_output


def construct_inceptionv4onfire(x,y,normalization,optimizer,dropout_value,activation,training=True):
    network=input_data(shape=[None, y, x, 3])
    #stem of inceptionv4
    conv1_3_3=conv_2d(network,32,3,strides=2,activation=activation,name='conv1_3_3_s2',padding='valid')
    conv2_3_3=conv_2d(conv1_3_3,32,3,activation=activation,name='conv2_3_3')
    conv3_3_3=conv_2d(conv2_3_3,64,3,activation=activation,name='conv3_3_3')
    b_conv_1_pool=max_pool_2d(conv3_3_3,kernel_size=3,strides=2,padding='valid',name='b_conv_1_pool')
    b_conv_1_pool=globals()[normalization](b_conv_1_pool)
    b_conv_1_conv=conv_2d(conv3_3_3,96,3,strides=2,padding='valid',activation=activation,name='b_conv_1_conv')
    b_conv_1=merge([b_conv_1_conv,b_conv_1_pool],mode='concat',axis=3)

    b_conv4_1_1=conv_2d(b_conv_1,64,1,activation=activation,name='conv4_3_3')
    b_conv4_3_3=conv_2d(b_conv4_1_1,96,3,padding='valid',activation=activation,name='conv5_3_3')

    b_conv4_1_1_reduce=conv_2d(b_conv_1,64,1,activation=activation,name='b_conv4_1_1_reduce')
    b_conv4_1_7=conv_2d(b_conv4_1_1_reduce,64,[1,7],activation=activation,name='b_conv4_1_7')
    b_conv4_7_1=conv_2d(b_conv4_1_7,64,[7,1],activation=activation,name='b_conv4_7_1')
    b_conv4_3_3_v=conv_2d(b_conv4_7_1,96,3,padding='valid',name='b_conv4_3_3_v')
    b_conv_4=merge([b_conv4_3_3_v, b_conv4_3_3],mode='concat',axis=3)
    
    b_conv5_3_3=conv_2d(b_conv_4,192,3,padding='valid',activation=activation,name='b_conv5_3_3',strides=2)
    b_pool5_3_3=max_pool_2d(b_conv_4,kernel_size=3,padding='valid',strides=2,name='b_pool5_3_3')
    b_pool5_3_3=globals()[normalization](b_pool5_3_3)
    b_conv_5=merge([b_conv5_3_3,b_pool5_3_3],mode='concat',axis=3)
    net=b_conv_5

    #inception modules

    for idx in range(1):
        net=inception_block_a(net,activation)

    #net=reduction_block_a(net)
    for idx in range(1):
        net=inception_block_b(net,activation)

    #net=reduction_block_b(net)

    for idx in range(1):
        net=inception_block_c(net,activation)

    pool5_7_7=global_avg_pool(net)
    pool5_7_7=dropout(pool5_7_7,dropout_value)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    if(training):
        network = regression(loss, optimizer=optimizer,
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    else:
        network=loss

    model = tflearn.DNN(network, checkpoint_path='inceptionv4',
                        max_checkpoints=1, tensorboard_verbose=0)

    return model
