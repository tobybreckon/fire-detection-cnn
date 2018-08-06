''' App for binary classfication, supply video as argument '''

import cv2
import os
import sys
import time

sys.path.insert(0, 'tflearn')

from firenet import *
from inceptionv1onfire import *
from customnet import *
from evaluate import *

model = construct_firenet (224, 224)
print("Constructed FireNet")

model.load(os.path.join("Models/FireNet", "firenet"),weights_only=True)
print("Loaded checkpoint")

rows = 224
cols = 224

if len(sys.argv) == 2:
    filename = sys.argv[1]
    video = cv2.VideoCapture(filename)
    print("Loaded video")
    cv2.namedWindow("Video")
    cv2.moveWindow("Video", 0, 0)
    width = int(video.get(3))
    height = int(video.get(4))
    fps = video.get(5)
    frame_time = 1/fps
    counter = 0
    while True:
        start = time.time()
        video.set(1,int(counter))
        ret, frame = video.read()
        if not ret:
            break
        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
        output = model.predict([small_frame])
        if round(output[0][0]) == 1:
            cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
            cv2.putText(frame,'ALARM',(int(width/16),int(height/4)), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA)
        elapsed = time.time() - start
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)
        counter += elapsed / frame_time
else:
    print("Add argument of file name")
