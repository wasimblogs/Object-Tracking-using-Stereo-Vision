'''Reads stereo pair images from the stereo setup'''

import cv2
import numpy as np
import time


cam1 = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(0)

# Checking cameras
print cam1.isOpened()
print cam2.isOpened()

imageSaved = 0
while cam1.isOpened() and cam2.isOpened():
    ## Ret returns if the image is read or not
    ## imageLeft or imageRight stores the images

    # Determine the time difference between frames and stereo pairs
    interFrameStart = time.time()
    retImage1, imageLeft = cam1.read()
    stereoImageStart = time.time()
    retImage2, imageRight = cam2.read()
    stereoImageEnd = time.time()
    stereoDifference = ( stereoImageEnd -stereoImageStart )*1000
    print int(stereoDifference), 'ms'

    # if proper images has been read
    if retImage1 and retImage2:

        cv2.imshow('Left', imageLeft)
        cv2.imshow('Right', imageRight)

    else:
        print 'Image Found on Left Camera : ', retImage1
        print 'Image Found on Right Camera : ', retImage2

    k = cv2.waitKey(01)
    if k == 27:
        cv2.destroyAllWindows()
        break

    interFrameEnd = time.time()
    interFrameDifference = ( interFrameEnd - interFrameStart ) *1000
    print int(interFrameDifference), 'ms'
