''' Evaluates a superpixel method '''

from __future__ import division
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import numpy as np
import os
import cv2
import tflearn
import h5py

def evaluatesp (model_directory, model):

    #h5f = h5py.File("/home/jrnf24/data/validationsp.h5", 'r')
    h5f = h5py.File("/Volumes/AJD/fire-data/superpixels/data/validationsp.h5", 'r')
    X = h5f["X"]
    Y = h5f["Y"]

    for filename in os.listdir(model_directory):
        if ".meta" in filename:
            model_name = filename[:-5]

    model.load(os.path.join(model_directory,model_name))

    fp = 0
    tp = 0
    fn = 0
    tn = 0
    loc_tp = 0
    loc_fn = 0
    sp_detections = 0
    s_tot = 0.0

    for i in range(len(X)):

        segments = slic(img_as_float(X[i]), n_segments = 100, sigma = 5)

        all_contours = []

        # Determine fire contours
        for (j, segVal) in enumerate(np.unique(segments)):

            # Construct a mask for the segment
            mask = np.zeros(X[i].shape[:2], dtype = "uint8")
            mask[segments == segVal] = 255
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Create the superpixel by applying the mask
            superpixel = cv2.bitwise_and(X[i], X[i], mask = mask)

            output = model.predict([superpixel])
            if round(output[0][0]) == 1:
                all_contours.append(contours[0])

        # Create bounding box from contours


        fire_detected = False
        if len(all_contours) > 0:

            fire_detected = True
            xmin, ymin = 224, 224
            xmax = ymax = 0
            for contour in all_contours:
                (x,y,w,h) = cv2.boundingRect(contour)
                xmin, xmax = min(x, xmin), max(x+w, xmax)
                ymin, ymax = min(y, ymin), max(y+h, ymax)

        # Find general detection results
        isfire = False
        if Y[i][0] != -1:
            isfire = True

        if isfire:
            if fire_detected:
                tp += 1
            else:
                fn += 1
        else:
            if fire_detected:
                fp += 1
            else:
                tn += 1

        # Find localtion specific detection results
        if fire_detected:
            sp_detections += 1

        if isfire:

            if fire_detected:

                true_box = np.zeros((224,224), dtype = "uint8")
                cv2.rectangle(true_box, (Y[i][0], Y[i][1]), (Y[i][2], Y[i][3]), (255), -1)
                true_area = cv2.countNonZero(true_box)

                sp_box = np.zeros((224,224), dtype = "uint8")
                cv2.rectangle(sp_box, (xmin, ymin), (xmax, ymax), (255), -1)
                sp_area = cv2.countNonZero(sp_box)

                intersect = cv2.bitwise_and(true_box, sp_box)
                intersect_area = cv2.countNonZero(intersect)

                s = intersect_area / true_area

                s_tot += s

                if s > 0.5:
                    loc_tp += 1
                else:
                    loc_fn += 1

            else:
                loc_fn += 1

    name = model_directory.split('/')[-2]

    f = open("{}.txt".format(name),'a')
    f.write("tp: {}\ntn: {}\nfp: {}\nfn: {}\nloc_tp: {}\nloc_fn: {}\nsp_detections: {}\ns_tot: {}\n".format(tp,tn,fp,fn,loc_tp,loc_fn,sp_detections,s_tot))


def evaluate_in_trainsp (model_name, model):

    #h5f = h5py.File("/home/jrnf24/data/validationsp.h5", 'r')
    h5f = h5py.File("/home/jrnf24/data/validationsp.h5", 'r')
    X = h5f["X"]
    Y = h5f["Y"]

    fp = 0
    tp = 0
    fn = 0
    tn = 0
    loc_tp = 0
    loc_fn = 0
    sp_detections = 0
    s_tot = 0.0

    for i in range(len(X)):

        segments = slic(img_as_float(X[i]), n_segments = 100, sigma = 5)

        all_contours = []

        # Determine fire contours
        for (j, segVal) in enumerate(np.unique(segments)):

            # Construct a mask for the segment
            mask = np.zeros(X[i].shape[:2], dtype = "uint8")
            mask[segments == segVal] = 255
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Create the superpixel by applying the mask
            superpixel = cv2.bitwise_and(X[i], X[i], mask = mask)

            output = model.predict([superpixel])
            if round(output[0][0]) == 1:
                all_contours.append(contours[0])

        # Create bounding box from contours
        fire_detected = False
        if len(all_contours) > 0:

            fire_detected = True
            xmin, ymin = 224, 224
            xmax = ymax = 0
            for contour in all_contours:
                (x,y,w,h) = cv2.boundingRect(contour)
                xmin, xmax = min(x, xmin), max(x+w, xmax)
                ymin, ymax = min(y, ymin), max(y+h, ymax)

        # Find general detection results
        isfire = False
        if Y[i][0] != -1:
            isfire = True

        if isfire:
            if fire_detected:
                tp += 1
            else:
                fn += 1
        else:
            if fire_detected:
                fp += 1
            else:
                tn += 1

        # Find localtion specific detection results
        if fire_detected:
            sp_detections += 1

        if isfire:

            if fire_detected:

                true_box = np.zeros((224,224), dtype = "uint8")
                cv2.rectangle(true_box, (Y[i][0], Y[i][1]), (Y[i][2], Y[i][3]), (255), -1)
                true_area = cv2.countNonZero(true_box)

                sp_box = np.zeros((224,224), dtype = "uint8")
                cv2.rectangle(sp_box, (xmin, ymin), (xmax, ymax), (255), -1)
                sp_area = cv2.countNonZero(sp_box)

                intersect = cv2.bitwise_and(true_box, sp_box)
                intersect_area = cv2.countNonZero(intersect)

                s = (intersect_area/true_area)
                s_tot += s

                if s > 0.5:
                    loc_tp += 1
                else:
                    loc_fn += 1

            else:
                loc_fn += 1

    name = model_directory.split('/')[-2]

    f = open("{}.txt".format(model_name),'a')
    f.write("tp: {}\ntn: {}\nfp: {}\nfn: {}\nloc_tp: {}\nloc_fn: {}\nsp_detections: {}\ns_tot: {}\n".format(tp,tn,fp,fn,loc_tp,loc_fn,sp_detections,s_tot))
