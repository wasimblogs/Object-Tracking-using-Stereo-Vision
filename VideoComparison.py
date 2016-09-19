import cv2
import numpy as np

disparity1 = cv2.VideoCapture("G:/Stereo/Res_Rectified.avi")
disparity2 = cv2.VideoCapture("G:/Stereo/Res_BetterMerge.avi")

while 1:
    try:
        ret1, frame1 = disparity1.read()
        ret2, frame2 = disparity2.read()

        win1 = "VIDEO 1"
        win2 = "VIDEO 2"
        cv2.imshow(win1, frame1)
        cv2.moveWindow(win1, 10,10)
        cv2.imshow(win2, frame2)
        cv2.moveWindow(win2, 700,10)

        k = cv2.waitKey(00)
        if k == 27:
            break
    except:
        cv2.destroyAllWindows()
        break
