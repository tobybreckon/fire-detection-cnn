#################################################################################

# Example : perform live fire detection in video using InceptionV3-OnFire CNN

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE



#################################################################################

from __future__ import division, print_function, absolute_import
import cv2
import os
import sys
import math

#################################################################################


import tflearn
import h5py
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d,global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

################################################################################

def construct_inceptionv3onfire(x,y, training=False):

    # build network as per architecture

    network = input_data(shape=[None, y, x, 3])

    conv1_3_3 = conv_2d(network, 32, 3, strides=2, activation='relu', name = 'conv1_3_3',padding='valid')
    conv2_3_3 = conv_2d(conv1_3_3, 32, 3, strides=1, activation='relu', name = 'conv2_3_3',padding='valid')
    conv3_3_3 = conv_2d(conv2_3_3, 64, 3, strides=2, activation='relu', name = 'conv3_3_3')

    pool1_3_3 = max_pool_2d(conv3_3_3, 3,strides=2)
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

if __name__ == '__main__':

################################################################################

#   construct and display model


    model = construct_inceptionv3onfire (224, 224, training=False)
    print("Constructed InceptionV3-OnFire ...")

    
    model.load(os.path.join("models/InceptionV3-OnFire", "inceptiononv3onfire"),weights_only=True)
    print("Loaded CNN network weights ...")

    ################################################################################

    # network input sizes

    rows = 224
    cols = 224

    # display and loop settings

    windowName = "Live Fire Detection - InceptionV3-OnFire";
    keepProcessing = True;

    ################################################################################

    if len(sys.argv) == 2:

        # load video file from first command line argument

        video = cv2.VideoCapture(sys.argv[1])
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
            output = model.predict([small_frame])

            # label image based on prediction

            if round(output[0][0]) == 1:
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
    else:
        print("usage: python inceptionV3-OnFire.py videofile.ext");

    ################################################################################
