################################################################################

# Example : perform live fire detection in video using superpixel localization
# and the superpixel trained version of the InceptionV1-OnFire CNN


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

from inceptionVxOnFire import construct_inceptionv4onfire
from inceptionVxOnFire import construct_inceptionv3onfire

################################################################################

# construct and display model
model = construct_inceptionv3onfire(224,224, training=False)
print("Constructed SP-InceptionV3-OnFire ...")

model.load(os.path.join("models/InceptionV3-OnFire", "inceptionv3onfire"),weights_only=True)
print("Loaded CNN network weights ...")

################################################################################

# network input sizes

rows = 224
cols = 224

# display and loop settings

windowName = "Live Fire Detection - Superpixels with SP-InceptionV3-OnFire";
keepProcessing = True;


################################################################################

# centering superpixels

def centering_superpixels(superpixel_image):

    # converting the image into grayscale

    gray = cv2.cvtColor(superpixel_image,cv2.COLOR_RGB2GRAY)

    # threshold the image and determine the contours

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) ==2 else cnts[1]

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)


    for c in cnts:

        # determining the coordinates of the bounding box rectangles

        x, y, w, h = cv2.boundingRect(c)

        # cropping the superpixel image

        ROI = superpixel_image[y:y+h,x:x+w]

        # converting the image into a numpy-array

        np_image2 = np.array(ROI)

        # resizing the image into (224, 224, 3)

        ROI = resize_image(np_image2,[224,224,3])
        break

    return ROI


################################################################################

# resizing image to the required [224,224,3] size

def resize_image(image,target_shape, pad_value = 0):

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
    output = output[subs_shape]
    return output

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
            # Using the approach, we re-use the same InceptionV3-OnFire architecture as described in
            # the paper with no changes trained on full frame images each
            # containing an isolated superpixel with the rest of the image being zero/black.

            superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)

            # converting the superpixel_images from BGR to RGB space

            superpixel = cv2.cvtColor(superpixel, cv2.COLOR_BGR2RGB)

            # centering the superpixels before testing

            superpixels = centering_superpixels(superpixel)


            # use loaded model to make prediction on given superpixel segments
            output = model.predict([superpixels])

            # we know the green/red label seems back-to-front here (i.e.
            # green means fire, red means no fire) but this is how we did it
            # in the paper (?!) so we'll just keep the same crazyness for
            # consistency with the paper figures

            if output[0][0]>0.9:
                # draw the contour
                # if prediction for FIRE was TRUE (round to 1), draw GREEN contour for superpixel
                cv2.drawContours(small_frame, contours, -1, (0,255,0), 1)

            else:
                # if prediction for FIRE was FALSE, draw RED contour for superpixel
                cv2.drawContours(small_frame, contours, -1, (0,0,255), 1)

        # stop the timer and convert to ms. (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

        stop_t_list.append(stop_t)

        # image display and key handling

        cv2.imshow(windowName, small_frame);

        # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

        key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
        if (key == ord('x')):
            keepProcessing = False;

        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);


else:
    print("usage: python superpixel-inceptionV3-OnFire.py videofile.ext");

################################################################################
