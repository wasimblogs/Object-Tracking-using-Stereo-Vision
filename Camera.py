'''check if the camera is working or not'''
# EXPLOITING THE VARIOUS FEATURES OF CAMERAS

# Must set both height and weight camera parameters
# Using any one of the image channel would be better than to
# convert the image in grayscale
# Logitech C310 webcam. The camera supports up to 1280x960 at 30fps


import cv2
import numpy as np


cap = cv2.VideoCapture(1)
cv2.namedWindow('Video')

if cap.isOpened():
    print 'Width : ', cap.get(3)
    print 'Height : ', cap.get(4)
    print 'FPS : ', cap.get(cv2.cv.CV_CAP_PROP_FPS)

else: print 'Camera not found'

# Setting the properties in the camera
# Setting the properties is not enough, the camera must support the resolution

#r1 = cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 800)
#r2 = cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1000)
#r3 = cap.set(cv2.cv.CV_CAP_PROP_FPS, 10)

#print 'New Height : ', cap.get(3)
#print 'New Width : ', cap.get(4)

#print r1
#print r2

camera = 1

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, str(camera), (20, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0))
        cv2.imshow('Video',frame)
        print frame.shape
    else:
        print 'No image found'
    k=cv2.waitKey(30)
    if k == 27:
        break

cv2.destroyAllWindows()
