# The camera id system doesn't work the same way as advertised. Quite arbitrary

# 800*600
# 1024*768
# 1280*960
# 2048*1536
# 1600*1200

# Integrated Webcam 1`
# Right Side : 2
# Left First (from nearest corner)
# Logitech C310 is a nice camera
# Try setting higher resolution than you intend to.

import cv2
import numpy as np
import time


cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

if cap1.isOpened():
    print 'Width : ', cap1.get(3)
    print 'Height : ', cap1.get(4)

else:
    print 'Camera 1 Not Found'

if cap2.isOpened():
    print 'Width : ', cap2.get(3)
    print 'Height : ', cap2.get(4)

else: print 'Camera 2 not found'

# Setting the properties in the camera
ret1 = cap1.set(3, 1024)
ret2 = cap1.set(4, 768)
print 'Camera 1 Parameters Set :', ret2, ret1

ret1 = cap2.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1600)
ret2 = cap2.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1200)

print 'Camera 2 Parameters Set :', ret2, ret1

print 'New width c1 : ', cap1.get(3)
print 'New Height c1: ', cap1.get(4)
print 'New width c2 : ', cap2.get(3)
print 'New Height c2: ', cap2.get(4)

while cap1.isOpened() and cap2.isOpened():

    # Read the time at which frames are captured
    t1 = time.time()
    ret1, frame1 = cap1.read()

    t2 = time.time()
    ret2, frame2 = cap2.read()

    print t2 - t1

    # if the image is found
    if ret1 and ret2:
        cv2.putText(frame1, str(t1), (20,100), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,200))
        cv2.putText(frame2, str(t2), (20,100), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,200))
        cv2.imshow('Camera 1', frame1)
        cv2.imshow('Camera 2', frame2)
        #combined = np.hstack((frame1, frame2))

        #print frame1.shape
    else:
        if not ret1:
            print 'Image 1 not found'
        if not ret2:
            print 'Image 2 not found'

    k=cv2.waitKey(5)
    if k == 27:
        break

    if k == 32:
        # Both images stacked together
        combined = np.hstack((frame1, frame2))
        cv2.imshow('Combined ', combined)

        #Save one image to file
        cv2.imwrite('Combined2.jpg', combined)
        print 'Image "Combined" saved!'

#    cv2.imshow('Combined ', combined)


cv2.destroyAllWindows()
