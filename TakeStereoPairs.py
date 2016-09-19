'''
Reads stereo pair images from the stereo setup
What's new: Implemented in modular fashion
'''

import os

import cv2


def takeStereoPair(leftDir, rightDir, row=3, col=6, SAVE_COUNT=20):
    """
    Description     : saves stereo pair of images taken from stereo camera
    :param leftDir  : directory where left image of the stereo pair is to be stored
    :param rightDir : directory where right image of the stereo pair is to be stored
    :param row      : no. of rows in chessboard (Actually 1 less)
    :param col      : no. of cols in chessboard (actually 1 less)
    :return         : None
    """
    # define chessboard pattern
    # Mention one wor and one col less than that in real
    WAIT_TIME = 1000
    ROW_CHESSBOARD = row
    COL_CHESSBOARD = col

    imageNameL = "ImageLeft"
    imageNameR = "ImageRight"
    ext = ".png"

    nameL = os.path.join(leftDir, imageNameL)
    nameR = os.path.join(rightDir, imageNameR)

    # Initialize camera
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)

    # Checking cameras
    print cam1.isOpened()
    print cam2.isOpened()

    imageSaved = 0
    while 1:
        ## Ret returns if the image is read or not
        ## imageLeft or imageRight stores the images

        # Determine the time difference between frames and stereo pairs
        retImage1, imageLeft = cam1.read()
        retImage2, imageRight = cam2.read()

        # if proper images has been read
        if retImage1 and retImage2:

            cv2.imshow('Left', imageLeft)
            cv2.imshow('Right', imageRight)
            cv2.waitKey(5)

            retChess1, corners1 = cv2.findChessboardCorners(imageLeft, (ROW_CHESSBOARD, COL_CHESSBOARD))
            retChess2, corners2 = cv2.findChessboardCorners(imageRight, (ROW_CHESSBOARD, COL_CHESSBOARD))

            # if the chessboard corners has been found
            if retChess1 and retChess2:
                # Save the stereo pair images
                nameL += str(imageSaved) + ext
                nameR += str(imageSaved) + ext

                cv2.imwrite(nameL, imageLeft)
                cv2.imwrite(nameR, imageRight)

                imageSaved = imageSaved + 1
            # else:
            #     print 'Pattern found on left image', retChess1
            #     print 'Pattern found on Right image', retChess2

            if imageSaved == SAVE_COUNT: break

        else:
            print 'Image Found on Left Camera : ', retImage1
            print 'Image Found on Right Camera : ', retImage2

        k = cv2.waitKey(WAIT_TIME)
        if k == 27:
            cv2.destroyWindow("Left")
            cv2.destroyWindow("Right")
            break


if __name__ == "__main__":
    pathL = "G:/Stereo/Left6"
    pathR = "G:/Stereo/Right6"
    row, col = 3, 6
    takeStereoPair(pathL, pathR, row, col)
    # help(takeStereoPair)
