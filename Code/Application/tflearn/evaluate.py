''' Evaluates a classification method '''

import tflearn
import cv2
import sys
import os
import h5py
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tensorflow.python.framework import ops

def evaluate (model_directory, datasets, model):

    for filename in os.listdir(model_directory):
        if ".meta" in filename:
            model_name = filename[:-5]

    model.load(os.path.join(model_directory,model_name))

    for data in datasets:

        # Load hdf5 dataset
        h5f = h5py.File(data, 'r')
        X = h5f['X']
        Y = h5f['Y']

        def get_result(result):
            if str(result) == "[0 1]":
                return 0
            else:
                return 1

        def get_prediction(prediction):
            if round(prediction[0]) == 1:
                return 1
            else:
                return 0

        fp = 0.0
        tp = 0.0
        fn = 0.0
        tn = 0.0
        iterations = int(len(X)/100)

        for iteration in range(iterations):

            predictions = model.predict(X[:100])

            for i in range (len(predictions)):

                prediction = get_prediction(predictions[i])

                if get_result(Y[i]) == 0:
                    if prediction == 0:
                        tn += 1
                    else:
                        fn += 1
                else:
                    if prediction == 1:
                        tp += 1
                    else:
                        fp += 1

            X = X[100:]
            Y = Y[100:]

        name = model_directory.split('/')[-2]

        if tp+tn+fp+fn == 0:
           accuracy = 0.0
        else:
           accuracy = round((tp+tn)/(tp+tn+fp+fn),3)

        if tp+fp == 0:
            precision = 0.0
        else:
            precision = round(tp/(tp+fp),3)

        if tp+fn == 0:
            recall = 0.0
        else:
            recall = round(tp/(tp+fn),3)

        f = open("{}.txt".format(name),'a')
        f.write("Set: {}\ntp: {}\ntn: {}\nfp: {}\nfn: {}\naccuracy: {}\nprecision: {}\nrecall: {}\n\n".format(data,tp,tn,fp,fn,accuracy,precision,recall))

def evaluate_in_train (model_name, datasets, model):

    for data in datasets:

        # Load hdf5 dataset
        h5f = h5py.File(data, 'r')
        X_eval = h5f['X']
        Y_eval = h5f['Y']

        def get_result(result):
            if str(result) == "[0 1]":
                return 0
            else:
                return 1

        def get_prediction(prediction):
            if round(prediction[0]) == 1:
                return 1
            else:
                return 0

        fp = 0.0
        tp = 0.0
        fn = 0.0
        tn = 0.0
        iterations = int(len(X_eval)/100)

        for iteration in range(iterations):

            predictions = model.predict(X_eval[:100])

            for i in range (len(predictions)):

                prediction = get_prediction(predictions[i])

                if get_result(Y_eval[i]) == 0:
                    if prediction == 0:
                        tn += 1
                    else:
                        fn += 1
                else:
                    if prediction == 1:
                        tp += 1
                    else:
                        fp += 1

            X_eval = X_eval[100:]
            Y_eval = Y_eval[100:]

        if tp+tn+fp+fn == 0:
           accuracy = 0.0
        else:
           accuracy = round((tp+tn)/(tp+tn+fp+fn),3)

        if tp+fp == 0:
            precision = 0.0
        else:
            precision = round(tp/(tp+fp),3)

        if tp+fn == 0:
            recall = 0.0
        else:
            recall = round(tp/(tp+fn),3)

        f = open("{}.txt".format(model_name),'a')
        f.write("Set: {}\ntp: {}\ntn: {}\nfp: {}\nfn: {}\naccuracy: {}\nprecision: {}\nrecall: {}\n\n".format(data,tp,tn,fp,fn,accuracy,precision,recall))
