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
from PIL import Image

################################################################################

from inceptionV4OnFire import construct_inceptionv4onfire
from inceptionV3OnFire import construct_inceptionv3onfire

################################################################################

# construct and display model
model = construct_inceptionv3onfire(224,224, training=False)
print("Constructed SP-InceptionV3-OnFire ...")

model.load('/home/capture/ganesh_new/weight_files/models_best_superpixels/inceptionv3_i/inceptionv3_i_bn_rmsprop_d_relu_150epoch.tflearn')
print("Loaded CNN network weights ...")

################################################################################

# network input sizes

rows = 224
cols = 224

# display and loop settings

windowName = "Live Fire Detection - Superpixels with SP-InceptionV1-OnFire";
keepProcessing = True;


################################################################################

# centering superpixels
def centering_sp2(superpixel_image):
    gray = cv2.cvtColor(superpixel_image,cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) ==2 else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ROI = superpixel_image[y:y+h,x:x+w]
        np_image2 = np.array(ROI)
        ROI = resize_image(np_image2,[224,224,3])
        break

    return ROI











def centering_superpixels(superpixel_image,superpixel_color):


    # get the coordinates of the bounding box

    x,y,w,h = cv2.boundingRect(superpixel_image)
    

    print([x,y,w,h])
    # image cropped to the size of the bounding box

    cropped = superpixel_color[x:x+w,y:y+h]
    np_image = np.array(cropped)
    print('Crop', len(superpixel_color), len(np_image))
    # output image resized to a shape of [224, 224, 3]

    output_image_size = resize_image(np_image, [224, 224, 3])

    return output_image_size

################################################################################

# resizing image to the required [224,224,3] size
def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """

    # very inefficient code ahead. Will refactor later.
    #calculate bounding box for superpixel

    coord = contours[0]
    x_min, y_min, x_max, y_max = 1000, 1000, 0, 0

    for coord_points in coord:
        coordinate = coord_points[0]
        x = coordinate[0]
        y = coordinate[1]

        if (x < x_min):
            x_min = x

        elif (x > x_max):
            x_max = x

        else:
            None

        if (y < y_min):
            y_min = y

        elif (y > y_max):
            y_max = y

        else:
            None

    #print(x_min, y_min)
    #print(x_max, y_max)

    pil_im = pil_im[y_min:y_max, x_min:x_max]

    #pil_im = cv2.cvtColor(pil_im, cv2.COLOR_BGR2RGB)
    
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    print('---->>', pil_im.shape)
    #pil_im = ImageOps.fit(pil_im, (448, 448), Image.ANTIALIAS)
    pil_im = cv2.resize(pil_im, (224, 224))
    return pil_im

    #cv2.imshow("", pil_im)

    #cv2.waitKey(0)


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
            # Using the approach, we re-use the same InceptionV1-OnFire architecture as described in
            # the paper [Dunnings / Breckon, 2018] with no changes trained on full frame images each
            # containing an isolated superpixel with the rest of the image being zero/black.

            superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)
            superpixel = cv2.cvtColor(superpixel, cv2.COLOR_BGR2RGB)
            
            #superpixels = centering_superpixels(thresh,superpixels_color)
            superpixels = centering_sp2(superpixel)
            np_image = Image.fromarray(superpixels)
            #np_image.show()
            #np_image.close()
            #superpixels = preprocess_image(superpixel,contours)

            # use loaded model to make prediction on given superpixel segments
            output = model.predict([superpixels])
            print(output[0][0])

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
        print(stop_t)
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
