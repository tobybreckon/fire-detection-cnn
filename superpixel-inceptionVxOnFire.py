################################################################################

# Example : perform live fire detection in video using superpixel localization
# and the superpixel trained version of the InceptionV1-OnFire,
# InceptionV3-OnFire and InceptionV4-OnFire CNN models

# Copyright (c) 2017/18 - Andrew Dunnings / Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

################################################################################

import cv2
import os
import sys
import math
import numpy as np
import argparse

################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

################################################################################

from inceptionVxOnFire import construct_inceptionv1onfire, construct_inceptionv3onfire, construct_inceptionv4onfire

################################################################################

# extract non-zero region of interest (ROI) in an otherwise zero'd image

def extract_bounded_nonzero(input):

    # take the first channel only (for speed)

    gray = input[:, :, 0];

    # find bounding rectangle of a non-zero region in an numpy array
    # credit: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    rows = np.any(gray, axis=1)
    cols = np.any(gray, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # cropping the non zero image

    return input[cmin:cmax,rmin:rmax]

################################################################################

# pad a supplied image to the required [X,Y,C] size

def pad_image(image, target_shape, pad_value = 0):

    assert isinstance(target_shape, list) or isinstance(target_shape, tuple)
    add_shape, subs_shape = [], []

    # obtain the current shape of the image

    image_shape = image.shape

    # determine the difference in target shape and image shape

    shape_difference = np.asarray(target_shape, dtype=int) - np.asarray(image_shape,dtype=int)

    for diff in shape_difference:

        # determine number of pixels to pad or remove based on difference

        if diff < 0:
            subs_shape.append(np.s_[int(np.abs(np.ceil(diff/2))):int(np.floor(diff/2))])
            add_shape.append((0, 0))
        else:
            subs_shape.append(np.s_[:])
            add_shape.append((int(np.ceil(1.0*diff/2)),int(np.floor(1.0*diff/2))))

    # pad the image to fit the target shape

    output = np.pad(image, tuple(add_shape), 'constant', constant_values=(pad_value, pad_value))

    output = output[tuple(subs_shape)]
    return output

################################################################################

# parse command line arguments

parser = argparse.ArgumentParser(description='Perform superpixel based InceptionV1/V3/V4 fire detection on incoming video')
parser.add_argument("-m", "--model_to_use", type=int, help="specify model to use", default=1, choices={1, 3, 4})
parser.add_argument('video_file', metavar='video_file', type=str, help='specify video file')
args = parser.parse_args()

#   construct and display model

print("Constructing SP-InceptionV" + str(args.model_to_use) + "-OnFire ...")

if (args.model_to_use == 1):

    # use InceptionV1-OnFire CNN model - [Dunning/Breckon, 2018]

    model = construct_inceptionv1onfire (224, 224, training=False)
    # also work around typo in naming of original models for V1 models [Dunning/Breckon, 2018] "...iononv ..."
    model.load(os.path.join("models/SP-InceptionV1-OnFire", "sp-inceptiononv1onfire"),weights_only=True)

elif (args.model_to_use == 3):

    # use InceptionV3-OnFire CNN model - [Samarth/Breckon, 2019]

    model = construct_inceptionv3onfire (224, 224, training=False)
    model.load(os.path.join("models/SP-InceptionV3-OnFire", "sp-inceptionv3onfire"),weights_only=True)

elif (args.model_to_use == 4):

    # use InceptionV4-OnFire CNN model - [Samarth/Breckon, 2019]

    model = construct_inceptionv4onfire (224, 224, training=False)
    model.load(os.path.join("models/SP-InceptionV4-OnFire", "sp-inceptionv4onfire"),weights_only=True)

print("Loaded CNN network weights ...")

################################################################################

# network input sizes

rows = 224
cols = 224

# display and loop settings

windowName = "Live Fire Detection - Superpixels with SP-InceptionV" + str(args.model_to_use) + "-OnFire"
keepProcessing = True

################################################################################

# load video file from first command line argument

video = cv2.VideoCapture(args.video_file)
print("Loaded video ...")

# create window

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

# get video properties

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_time = round(1000/fps)

while (keepProcessing):

    # start a timer (to see how long processing and display takes)

    start_t = cv2.getTickCount()

    # get video frame from file, handle end of file

    ret, frame = video.read()
    if not ret:
        print("... end of video file reached")
        break

    # re-size image to network input size and perform prediction

    small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

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

        # N.B. ... but for the later work using the InceptionV3-OnFire and InceptionV4-OnFire architecture
        # as described in the paper [Samarth / Breckon, 2019] we instead centre and pad the resulting
        # image with zeros

        if ((args.model_to_use == 3) or (args.model_to_use == 4)):

            # converting the superpixel_images from BGR to RGB space

            superpixel = cv2.cvtColor(superpixel, cv2.COLOR_BGR2RGB)

            # center and re-scale the superpixels before testing

            superpixel = pad_image(extract_bounded_nonzero(superpixel), [224,224,3])

        # use loaded model to make prediction on given superpixel segment
        # which is now:
        # - an image (tensor) of dimension 224 x 224 x 3 (constructed from the superpixel as per above)
        # - for InceptionV1-OnFire: a 3 channel colour image with channel ordering BGR (not RGB)
        # - for InceptionV3-OnFire / InceptionV4-OnFire: a 3 channel colour image with channel ordering RGB
        # - un-normalised (i.e. pixel range going into network is 0->255)

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

    stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

    # image display and key handling

    cv2.imshow(windowName, small_frame)

    # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

    key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF
    if (key == ord('x')):
        keepProcessing = False
    elif (key == ord('f')):
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

################################################################################
