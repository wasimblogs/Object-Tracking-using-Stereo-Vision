import os

import cv2
import numpy as np

import Scratch7a as calibration


def go(leftFile, rightFile, SAVE_RESULTS=False):
    # CALIBRATION
    # THE ORDER OF THE CAMERA IS CRITICAL

    if SAVE_RESULTS:
        # Name formation
        dir, file = os.path.splitext(leftFile)
        rectifiedName = dir + "__Rectified" + file
        disparityName = dir + "__Disparity" + file
        resDisparity = cv2.VideoWriter(disparityName, -1, 25, (640, 480))
        resRectify = cv2.VideoWriter(rectifiedName, -1, 25, (640, 480))
        print resDisparity.isOpened()
        print resRectify.isOpened()

    left = cv2.VideoCapture(leftFile)
    right = cv2.VideoCapture(rightFile)

    print left.isOpened()
    print right.isOpened()
    i = 0
    print i
    while left.isOpened() and right.isOpened():
        # try:
        retL, frameL = left.read()
        retR, frameR = right.read()

        # frameL1 = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        # frameR1 = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        if retL and retR:
            rectifiedL, rectifiedR, disparity, depthMap = calibration.rectifyV(frameR, frameL,FIND_DEPTH_OBJECTS=True)

            inputStereo = np.hstack((frameL, frameR))
            rectified = np.hstack((rectifiedL, rectifiedR))

            cv2.line(rectified, (0,120),(640*2,120),(255,255,255))
            cv2.line(rectified, (0,240),(640*2,240),(255,255,255))
            cv2.line(rectified, (0,360),(640*2,360),(255,255,255))

            cv2.line(inputStereo, (0,120),(640*2,120),(255,255,255))
            cv2.line(inputStereo, (0,240),(640*2,240),(255,255,255))
            cv2.line(inputStereo, (0,360),(640*2,360),(255,255,255))

            disparityD = disparity / np.max(disparity)
            # cv2.imshow("IMAGE", frameL)
            # cv2.imshow('RECTIFIED', rectified)
            cv2.imshow('DISPARITY', disparityD)
            cv2.imshow("DEPTH MAP", depthMap)
            k = cv2.waitKey(100)
            if k == 27:
                break

            if k == ord("S") or k == ord("s"):
                name = "G:/Stereo/DisparityO/"
                num = np.random.randint(0,1000)
                nn = str(num)+str(i)

                filenameRect = name + "__Rectified__"+nn+".jpg"
                filenameDisp = name + "__Disparity__"+nn+".jpg"
                filenameInput = name + "__Input__"+nn+".jpg"
                filenameDepth = name + "__Depth__" + nn + ".jpg"

                cv2.imwrite(filenameRect, rectified)
                cv2.imwrite(filenameDisp, disparityD)
                cv2.imwrite(filenameInput, inputStereo)
                cv2.imwrite(filenameDepth, depthMap)

            for i in xrange(0,0):
                left.read()
                right.read()

        if SAVE_RESULTS:
            resDisparity.write(disparity)
            resRectify.write(rectifiedL)

        i += 1

        # except:
        #     print 'End of Video File Reached'
        #     break

    # when everything done, release the capture
    left.release()
    right.release()
    # resDisparity.release()
    cv2.destroyAllWindows()


# left = 'G:/Stereo/New8L.avi'
# right = 'G:/Stereo/New8R.avi'
left = 'G:/Stereo/DanceR9.avi'
right = 'G:/Stereo/DanceL9.avi'
go(left, right, SAVE_RESULTS=False)
