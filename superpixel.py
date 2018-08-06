''' App for localization, supply video as argument
    Runs superpixel fire detection script using models'''

import cv2
import os
import sys
import time

# Original implementation of SLIC using Scikit-image, no longer used
#from skimage.segmentation import slic
#from skimage.segmentation import mark_boundaries
#from skimage.util import img_as_float

# Insert filepath of tflearn folder which contains cnn architecture files
sys.path.insert(0, 'tflearn')

from firenet import *
from inceptionv1onfire import *
from customnet import *
from evaluate import *

# Will need to change method name for relevant model
model = construct_inceptionv1onfire (224, 224)
print("Constructed FireNet")

# Insert filepath of preloaded models
# First parameter is filepath of respective model, second parameter is name of model file
model.load(os.path.join("Models/SP-InceptionV1-OnFire", "inceptiononv1onfire"),weights_only=True)
print("Loaded checkpoint")

rows = 224
cols = 224

# If video file is provided i.e. 2 arguments
if len(sys.argv) == 2:

    filename = sys.argv[1]
    video = cv2.VideoCapture(filename)

    print("Loaded video")
    cv2.namedWindow("Video")
    cv2.moveWindow("Video", 0, 0)

    width = int(video.get(3))
    height = int(video.get(4))
    scale_width = width / 224
    scale_height = height / 224

    fps = video.get(5)
    frame_time = 1/fps

    counter = 0

    while True:

        start = time.time()
        video.set(1,int(counter))
        ret, frame = video.read()
        if not ret:
            # If video has ended break while loop and exit
            break

        # Small frame because original video frame is resize to specified resolution
        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

        # SKimage SLIC implementation below from original code
        # sigma = GaussianBlur kernel, n_segments = number of superpxiels
        #segments = slic(img_as_float(small_frame), n_segments = 100, sigma = 5)

        # OpenCV imgproc SLIC implementation below

        #small_frame = cv2.GaussianBlur(small_frame, (3,3), 5)
        slic = cv2.ximgproc.createSuperpixelSLIC(small_frame, region_size=22)
        slic.iterate(10)


        # getLabels method returns the different superpixel segments
        segments = slic.getLabels()
        print(len(np.unique(segments)))
        #cv2.imshow("VideoSLIC", segments)


        # Loop over the unique segment values
        for (i, segVal) in enumerate(np.unique(segments)):

            # Construct a mask for the segment
            mask = np.zeros(small_frame.shape[:2], dtype = "uint8")
            mask[segments == segVal] = 255
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Create the superpixel by applying the mask
            superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)

            '''
            new_contours = np.copy(contours)
            for contour in new_contours:
                for i in contour:
                    i[0][0] = int(i[0][0] * scale_width)
                    i[0][1] = int(i[0][1] * scale_height)
            '''
            # Use loaded model to make prediction on given superpixel segments
            output = model.predict([superpixel])


            if round(output[0][0]) == 1:
                # Draw the contour
                # If prediction for fire was true (round to 1), draw green contour for superpixel
                cv2.drawContours(small_frame, contours, -1, (0,255,0), 1)

            else:
                # If prediction for fire was false, draw red contour for superpixel
                cv2.drawContours(small_frame, contours, -1, (0,0,255), 1)

        elapsed = time.time() - start
        new_frame = cv2.resize(small_frame, (width, height), cv2.INTER_AREA)
        cv2.imshow("Video", new_frame)
        key = cv2.waitKey(1)
        counter += elapsed / frame_time
        # SIMPLY MUTIPLY CONTOUR POINTS BY SCALING FACTOR

else:
    print("Add argument of file name")
