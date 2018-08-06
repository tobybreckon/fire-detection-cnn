''' Performs classification upon a chosen video '''

import cv2
import os

from alexnet import *
from alexnetv1 import *
from alexnetv2 import *
from alexnetv3 import *
from alexnetv4 import *
from alexnetv5 import *
from googlenet import *
from googlenetv1 import *
from googlenetv2 import *
from vgg13 import *
from customnet import *
from evaluate import *

model = construct_googlenetv2 (224, 224)

model.load(os.path.join("/Users/Andrew/Documents/models/trained/GSE/", "googlenetv2-7600"))

rows = 224
cols = 224

if len(sys.argv) == 2:
    filename = sys.argv[1]
    video = cv2.VideoCapture(filename)
    width = int(video.get(3))
    height = int(video.get(4))
    #fps = video.get(5)
    displayTime = 125
    counter = 0
    video.set (1,0)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        smallFrame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
        output = model.predict([smallFrame])
        if round(output[0][0]) == 1:
            cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 20)
        cv2.imshow("Video", frame)
        key = cv2.waitKey(displayTime)
        counter += 1
else:
    print("Add argument of file name")
