################################################################################

# Example : perform live fire detection in video using superpixel localization
# and the superpixel trained version of the InceptionV1-OnFire CNN

# Copyright (c) 2017/18 - Andrew Dunnings / Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

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

from inceptionV1OnFire import construct_inceptionv1onfire

################################################################################

# construct and display model

model = construct_inceptionv1onfire (224, 224, training=False)
print("Constructed SP-InceptionV1-OnFire ...")

model.load(os.path.join("models/SP-InceptionV1-OnFire", "sp-inceptiononv1onfire"),weights_only=True)
print("Loaded CNN network weights ...")

################################################################################

# network input sizes

rows = 224
cols = 224

# display and loop settings

windowName = "Live Fire Detection - Superpixels with SP-InceptionV1-OnFire";
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

        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA);

        # OpenCV imgproc SLIC superpixels implementation below

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
            # cv2.imshow("superpixel", superpixel);

            # use loaded model to make prediction on given superpixel segments
            output = model.predict([superpixel])

            # we know the green/red label seems back-to-front here (i.e.
            # green means fire, red means no fire) but this is how we did it
            # in the paper (?!) so we'll just keep the same crazyness for
            # consistency with the paper figures

            if round(output[0][0]) == 1:
                # draw the contour
                # if prediction for FIRE was TRUE (round to 1), draw GREEN contour for superpixel
                cv2.drawContours(small_frame, contours, -1, (0,255,0), 1)

            else:
                # if prediction for FIRE was FALSE, draw RED contour for superpixel
                cv2.drawContours(small_frame, contours, -1, (0,0,255), 1)

        # stop the timer and convert to ms. (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

        # image display and key handling

        cv2.imshow(windowName, small_frame);

        # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

        key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
        if (key == ord('x')):
            keepProcessing = False;
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
else:
    print("usage: python superpixel-inceptionV1-OnFire.py videofile.ext");

################################################################################
