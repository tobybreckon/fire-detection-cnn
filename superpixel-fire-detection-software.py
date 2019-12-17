################################################################################

# Example : perform live fire detection in video using superpixel localization


################################################################################

import cv2
import os
import sys
import math
import numpy as np

################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

################################################################################

from inceptionV4OnFire import construct_inceptionv4onfire
from inceptionV3OnFire import construct_inceptionv3onfire

################################################################################

# construct and display model
model = construct_inceptionv3onfire(224,224, training=False)

################################################################################

# load the network weights
model.load('inceptionv3_i_bn_rmsprop_d_relu.tflearn')


# network input sizes

rows = 224
cols = 224

# display and loop settings

windowName = "Live Fire Detection - Superpixels with SP-InceptionV1-OnFire";
keepProcessing = True;

################################################################################
def centering_superpixels(superpixel_image):
    imageBox = superpixel_image.getbbox()
    cropped = superpixel_image.crop(imageBox)
    np_image = np.array(cropped)
    output_image_size = resize_image(np_image, [224, 224, 3])

    return output_image_size

def resize_image(image,target_shape, pad_value = 0):
    assert isinstance(target_shape, list) or isinstance(target_shape, tuple)
    add_shape, subs_shape = [], []

    image_shape = image.shape
    shape_difference = np.asarray(target_shape, dtype=int) - np.asarray(image_shape,dtype=int)
    for diff in shape_difference:
        if diff < 0:
            subs_shape.append(np.s_[int(np.abs(np.ceil(diff/2))):int(np.floor(diff/2))])
            add_shape.append((0, 0))
        else:
            subs_shape.append(np.s_[:])
            add_shape.append((int(np.ceil(1.0*diff/2)),int(np.floor(1.0*diff/2))))
    output = np.pad(image, tuple(add_shape), 'constant', constant_values=(pad_value, pad_value))
    output = output[subs_shape]
    return output



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
    stop_t_list=list()
    count=0

    while (keepProcessing):

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount();

        # get video frame from file, handle end of file

        ret, frame = video.read()
        if not ret:
            print("... end of video file reached");
            break;

        # re-size image to network input size and perform prediction

        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA);

        # OpenCV imgproc SLIC superpixels implementation below
        #print(small_frame.shape)
        slic = cv2.ximgproc.createSuperpixelSLIC(small_frame, region_size=22)
        slic.iterate(10)

        # getLabels method returns the different superpixel segments

        segments = slic.getLabels()



        # print(len(np.unique(segments)))

        # loop over the unique segment values
        for (i, segVal) in enumerate(np.unique(segments)):

            # Construct a mask for the segment
            mask = np.zeros(small_frame.shape[:2], dtype = "uint8")
            mask[segments == segVal] = 255

            # get contours (first checking if OPENCV >= 4.x)

            if (int(cv2.__version__.split(".")[0]) >= 4):
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # create the superpixel by applying the mask

            # N.B. this creates an image of the full frame with this superpixel being the only non-zero
            # (i.e. not black) region. CNN training/testing classification is performed using these
            # full frame size images, rather than isolated small superpixel images.
            # Using the approach, we re-use the same InceptionV1-OnFire architecture as described in
            # the paper [Dunnings / Breckon, 2018] with no changes trained on full frame images each
            # containing an isolated superpixel with the rest of the image being zero/black.

            superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)
            superpixel = cv2.cvtColor(superpixel, cv2.COLOR_RGB2BGR)

            superpixel=centering_superpixels(im)

            # use loaded model to make prediction on given superpixel segments
            output = model.predict([superpixel])

            # we know the green/red label seems back-to-front here (i.e.
            # green means fire, red means no fire) but this is how we did it
            # in the paper (?!) so we'll just keep the same crazyness for
            # consistency with the paper figures

            if output[0][0]>0.96:
                # draw the contour
                # if prediction for FIRE was TRUE (round to 1), draw GREEN contour for superpixel
                cv2.drawContours(small_frame, contours, -1, (0,255,0), 1)

            else:
                # if prediction for FIRE was FALSE, draw RED contour for superpixel
                cv2.drawContours(small_frame, contours, -1, (0,0,255), 1)

        # stop the timer and convert to ms. (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;
        print(stop_t)
        stop_t_list.append(stop_t)

        # image display and key handling

        cv2.imshow(windowName, small_frame);
        count+=1

        # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

        key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
        if (key == ord('x')):
            keepProcessing = False;

        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

    print(sum(stop_t_list[1:]) / (len(stop_t_list) - 1))
    print('its here')
else:
    print("usage: python superpixel-inceptionV3-OnFire.py videofile.ext");

################################################################################
