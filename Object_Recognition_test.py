import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras

#camera setting
def camera():
    VideoCapture = cv2.VideoCapture(0)
    
    if VideoCapture.isOpened() == False:
        print("Could not open the Camera.")
        return;
    
    while True:
        ret, frame = VideoCapture.read()
        cv2.imshow('camera', frame)
    
    
        # ASCII 10 > Enter / 27 > ESC / 32 > Space
        # if push the space bar, frame is saved, camera is ended.
        if cv2.waitKey(1) & 0xFF == 32 :
            capture_image = frame
            cv2.destroyAllWindows()
            break
            
    # save a frame in images folder
    cv2.imwrite('./images/capture_image.jpg',frame)
