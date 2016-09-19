"""Stores video in RAM temporarily to speed of tereo camera"""
import os

import cv2


def writeVideo(fileName, NO_OF_FRAMES=100):
    """
    This function writes video
    filename    : name of file where video is to be stored
    returns     : none
    Fails if the dir of the video doesn't exist
    """

    # Initialize camera
    camera1 = cv2.VideoCapture(1)
    camera2 = cv2.VideoCapture(0)

    # Check status of camera
    print camera1.isOpened()
    print camera2.isOpened()

    # Name formation
    dir, ext = os.path.splitext(filename)
    leftName = dir + "L" + ext
    rightName = dir + "R" + ext

    # Initialize video container
    video1 = cv2.VideoWriter(leftName, -1, 25, (640, 480))
    print video1
    video2 = cv2.VideoWriter(rightName, -1, 25, (640, 480))
    print video2

    # List to store left and right videos
    left = []
    right = []
    while 1:
        f1, img1 = camera1.read()
        f2, img2 = camera2.read()

        if f1 and f2:
            print "Cameras Ready!"
            break


    i = 0
    while True:
        f1, img1 = camera1.read()
        f2, img2 = camera2.read()

        if f1 and f2:
            left.append(img1)
            right.append(img2)

        cv2.imshow("LEFT", img1)
        cv2.imshow("RIGHT", img2)

        k = cv2.waitKey(01)
        if k == 27:
            cv2.destroyAllWindows()
            break

        i = i + 1
        print i
        if i > NO_OF_FRAMES:
            print "Limit of 400 frames reached: "
            cv2.destroyAllWindows()
            break

    camera1.release()
    camera2.release()

    left.reverse()
    right.reverse()

    for i in xrange(0, len(left)):
        print " Writing frames ", i
        video1.write(left.pop())
        video2.write(right.pop())

    del left
    del right

    video1.release()
    video2.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    filename = 'G:/Stereo/New10.avi'
    writeVideo(filename,400)
