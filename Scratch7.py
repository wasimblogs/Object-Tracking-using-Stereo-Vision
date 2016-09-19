import glob
import os
import pickle

import cv2
import numpy as np

import TrackElements as te

"""
New in scratch four:
uses os.path.join to joing paths which makes it robust to manually joining paths
More modular / reusable

What's new in Scratch 5?
    Saves camera paramters to a file.
    Read camera parameters from a file

What's new in Scratch 6
Obtaining depth values of objects

What's new in Scratch7?
More precise depth determination
"""

# Parameters for disparity
WIN_SIZE = 3
MIN_DISP = 16 * 1
NUM_DISP = 16 * 8 - MIN_DISP
stereo = cv2.StereoSGBM(minDisparity=MIN_DISP,  # Max diff - 1
                        numDisparities=NUM_DISP,
                        SADWindowSize=WIN_SIZE,
                        uniquenessRatio=17,
                        speckleWindowSize=2000,
                        speckleRange=100,
                        disp12MaxDiff=10,
                        # P1 = 8*3*WIN_SIZE**2,
                        # P2 = 32*3*WIN_SIZE**2,
                        P1=20 * WIN_SIZE ** 2,
                        P2=150 * WIN_SIZE ** 2,
                        fullDP=False,
                        preFilterCap=29
                        )


def leastStdDevRoi(image, GRID_ROW=4, GRID_COL=4):
    # def calcHistogramColor(image, BINS=10):
    """
    Calculates localized histogram
    :param image:
    :return: roi with least standard deviation
    """
    r, c = image.shape[:2]
    width = c / GRID_COL
    height = r / GRID_ROW

    startW = 0
    listRoi = []
    deviations = []
    for i in xrange(0, GRID_COL):
        startH = 0
        for j in xrange(0, GRID_ROW):
            roi = image[startH:startH + height, startW: startW + width]
            listRoi.append((startW, startH, width, height))
            startH = startH + height
            mean, std_dev = cv2.meanStdDev(roi)
            deviations.append(std_dev)

        startW = startW + width

    m = np.min(deviations)
    index = deviations.index(m)
    x,y,w,h = listRoi[index]

    roi = image[y:y+h, x:x+w]
    return roi

def findDepth(disparity):
    ppc = P2[0][3] / 3
    row, col = disparity.shape[:2]
    x, y, W, z = 0, 0, 1, disparity[0][0]

    depth = np.zeros((row, col, 3), np.uint8)

    arr = np.array((x, y, z, W), np.float32)
    for i in xrange(0, row):
        for j in xrange(0, col):
            x, y, W = i, j, 1
            z = disparity[i][j]
            arr = np.array((x, y, z, W), np.float32)
            res = np.dot(Q, arr)
            res = res / res[3]
            depth[i][j] = res[:3]

    return depth


def findDepthObjects(disparity):
    """
    :param disparity: Disparity before normalizing
    :return: depth of objects in cm
    """
    print "La aba object depth nikalnu parcha"
    ppc = P2[0][3] / 6
    a = P2[0][3] * P2[0][0]
    depth = []
    depth_cm = depth

    # disparity = disparity / np.max(disparity)

    rects = te.blobs(disparity, GRAYSCALE=True)
    normalized = disparity / np.max(disparity)

    for i in xrange(0, len(rects)):
        x, y, w, h = rects[i]
        center_x = (2 * x + w) / 2
        center_y = (2 * y + h) / 2

        roi = disparity[y:y + h, x:x + w]
        roi = leastStdDevRoi(roi)

        avg_disp = np.sum(roi) / roi.size
        # d = a / disparity[center_x][center_y]
        d = a / avg_disp
        d = d / ppc
        cv2.rectangle(disparity, (x, y), (x + w, y + h), (255, 255, 255), 3)
        cv2.putText(disparity, str(int(d)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.imshow("Depth of Object", normalized)
        print "Depth of object : \t", d
    return normalized


def findDepth2(disparity):
    ppc = P2[0][3] / 6
    a = P2[0][3] * P2[0][0]
    row, col = disparity.shape[:2]
    depth = np.zeros_like(disparity)
    depth_cm = depth
    for i in xrange(0, row):
        for j in xrange(0, col):
            depth[i][j] = a / disparity[i][j]
            depth_cm[i][j] = depth[i][j] / ppc

    print "MIN DEPTH : ", np.min(depth_cm), " cm"
    print "MAX DEPTH : ", np.max(depth_cm), " cm"
    print "MEAN DEPTH : ", depth_cm.mean(), " cm"
    return depth / depth.max()


def calib(pathL, pathR, row=6, col=9, SAVE=False, DISPLAY=False):
    """
    :param pathL: directory name of images for left camera
    :param pathR: directory name of images for right camera
    :param row: no. of rows in chessboard (Actually 1 less)
    :param col: no. of cols in chessboard (1 Less)
    :return: mapx1, mapy1, mapx2, mapy2, Q, P1, P2
    :returns parameters for rectification and reprojection matrix
    : P1, P2 are projection matrices
    """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare the object points like (0,0,0), (1,0,0,), (2,0,0), . .. . .
    ROW = row
    COL = col
    objP = np.zeros((ROW * COL, 3), np.float32)
    objP[:, :2] = np.mgrid[0:COL, 0:ROW].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the image
    objpoints = []  # 3D points in real world space
    imagePointsL = []  # 2D points in image plane
    imagePointsR = []  # 2D points in image plane

    everything = "*.*"
    pathL = os.path.join(pathL, "*.*")
    pathR = os.path.join(pathR, "*.*")
    folderL = glob.glob(pathL)
    folderR = glob.glob(pathR)

    print "Number of images in FolderL", len(folderL)
    print "Number of images in FolderR", len(folderR)
    image = cv2.imread(folderL[0], 0)
    H, W = image.shape[:2]

    for i in xrange(0, len(folderL)):
        imageL = cv2.imread(folderL[i], 0)
        imageR = cv2.imread(folderR[i], 0)

        print folderL[i], folderR[i]

        retL, cornersL = cv2.findChessboardCorners(imageL, (COL, ROW))
        retR, cornersR = cv2.findChessboardCorners(imageR, (COL, ROW))

        if retL and retR:
            objpoints.append(objP)

            cornersL2 = cv2.cornerSubPix(imageL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR2 = cv2.cornerSubPix(imageR, cornersR, (11, 11), (-1, -1), criteria)

            imagePointsL.append(cornersL)
            imagePointsR.append(cornersR)

    r1, c1, d1, R1, T1 = cv2.calibrateCamera(objpoints, imagePointsL, (W, H))
    r2, c2, d2, R2, T2 = cv2.calibrateCamera(objpoints, imagePointsR, (W, H))

    k, c11, d11, c22, d22, R, T, E, F = cv2.stereoCalibrate(objpoints, imagePointsL, imagePointsR, (W, H),
                                                            c1, d1, c2, c2, flags=cv2.CALIB_FIX_INTRINSIC)

    nc1, roi1 = cv2.getOptimalNewCameraMatrix(c1, d1, (W, H), alpha=-1)
    nc2, roi2 = cv2.getOptimalNewCameraMatrix(c2, d2, (W, H), alpha=-1)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(c1, d1, c2, d1, (W, H), R, T, alpha=-1)

    # THE PARAMETER AFTER d1 is highly sensitive
    mapx1, mapy1 = cv2.initUndistortRectifyMap(c1, d1, R1, P1, (W, H), cv2.CV_32FC2)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(c2, d2, R2, P2, (W, H), cv2.CV_32FC2)

    # Displaying the results
    if DISPLAY:
        print 'Reprojection Error of Left Camera :\t', r1
        print 'Reprojection Error of Left Camera :\t', r2

        print '\nIntrinsic Matrix of Left Camera\n', c1
        print '\nIntrinsic Matrix of Right Camera\n', c2
        print "\nDistortion Matrix of Left Camera \t", d1
        print "\nDistortion Matrix of Right Camera \t", d2
        print "\nProjection Matrix of Left Camera\n", P1
        print "\nProjection Matrix of Right Camera\n", P2

        print "\nTranslation Matrix of Stereo Camera\n", T
        print "\nFundamental Matrix of Stereo Camera\n", F
        print "\nEssential Matrix of Stereo Camera\n", E
        print "\nReprojection Matrix of Stereo Camera\n", Q

        print "\nBaseline Distance of Stereo Camera: ", P2[0][3]
        print "Common Focal length of Stereo Camera:", P2[0][0]

    # SAVING THE CALIBRATION PARAMETERS
    if SAVE:
        with open("G:/stereo/stereoParams.csv", "w") as f:
            pickle.dump(mapx1, f)
            pickle.dump(mapy1, f)
            pickle.dump(mapx2, f)
            pickle.dump(mapy2, f)
            pickle.dump(Q, f)
            pickle.dump(P1, f)
            pickle.dump(P2, f)

    return mapx1, mapy1, mapx2, mapy2, Q, P1, P2


def readCalibParamters():
    """
    :return: mapx1, mapy1, mapx2, mapy2, Q, P1, P2
    """

    with open("G:/stereo/stereoParams.csv", "r") as f:
        mapx1 = pickle.load(f)
        mapy1 = pickle.load(f)
        mapx2 = pickle.load(f)
        mapy2 = pickle.load(f)
        Q = pickle.load(f)
        P1 = pickle.load(f)
        P2 = pickle.load(f)

    return mapx1, mapy1, mapx2, mapy2, Q, P1, P2


def rectify(pathL, pathR, SAVE_RES=False):
    """
    :param pathL: directory of left images of stereo pairs
    :param pathR: directory of right images of stereo pairs
    :return: rectified, disparity, imageL
    rectified : is rectified image
    disparity is disparity of stereo pairs
    imageL is left image of the stereo pairs
    """
    folderL = glob.glob(pathL + "/*.*")
    folderR = glob.glob(pathR + "/*.*")

    count = 0
    num = np.random.randint(0, 1000)
    for i in xrange(0, len(folderL)):
        imageL = cv2.imread(folderL[i])
        imageR = cv2.imread(folderR[i])

        image = np.hstack((imageL, imageR))

        print folderL[i], '\t', folderR[i]
        rectifiedL = cv2.remap(imageL, mapx1, mapy1, cv2.INTER_LINEAR)
        rectifiedR = cv2.remap(imageR, mapx2, mapy2, cv2.INTER_LINEAR)

        rectified = np.hstack((rectifiedL, rectifiedR))
        cv2.line(rectified, (0, 150), (640 * 2, 150), 2)
        cv2.line(rectified, (0, 300), (640 * 2, 300), 2)
        cv2.line(rectified, (0, 450), (640 * 2, 450), 2)

        input_image = np.hstack((imageL, imageR))
        cv2.line(input_image, (0, 150), (640 * 2, 150), 2)
        cv2.line(input_image, (0, 300), (640 * 2, 300), 2)
        cv2.line(input_image, (0, 450), (640 * 2, 450), 2)

        output = np.vstack((input_image, rectified))
        disparity = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0

        disparity = disparity / np.max(disparity)
        # cv2.imwrite("G:/Stereo/Input/Results/Disparity"+str(count)+".jpg", disparity)

        cv2.imshow("RECTIFIED L", rectified)
        cv2.imshow("DISPARITY", disparity)
        cv2.waitKey(100)

        if SAVE_RES:
            name = "G:/Stereo/Output/"
            nn = str(num) + str(i)
            inputname = name + "__input__" + nn + ".jpg"
            rectifiedname = name + "__rectified__" + nn + ".jpg"

            cv2.imwrite(inputname, output)

        count = count + 1

    return rectified, disparity, imageL


def rectifyV(imageL, imageR):
    """
    :param imageL: left image of the stereo pair
    :param imageR: right image of the stereo pair
    :return: rectified, disparity, points
    rectifiedL : is rectified imageL
    rectifiedR : is rectified imageR
    disparity is disparity of stereo pairs
    points contains depth in cm/mm
    """

    rectifiedL = cv2.remap(imageL, mapx1, mapy1, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(imageR, mapx2, mapy2, cv2.INTER_LINEAR)

    rectified = np.hstack((rectifiedL, rectifiedR))
    cv2.line(rectified, (0, 150), (640 * 2, 150), 2)
    cv2.line(rectified, (0, 300), (640 * 2, 300), 2)
    cv2.line(rectified, (0, 450), (640 * 2, 450), 2)

    disparity = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0
    depthMap = findDepthObjects(disparity)
    disparity = disparity / np.max(disparity)
    # cv2.imwrite("G:/Stereo/Input/Results/Disparity"+str(count)+".jpg", disparity)

    return rectifiedL, rectifiedR, depthMap, disparity


pathL = "G:/Stereo/Input/Left4"
pathR = "G:/Stereo/Input/Right4"

# pathL = "G:\Stereo\Input/Left_near"
# pathR = "G:\Stereo\Input/Right_near"

mapx1, mapy1, mapx2, mapy2, Q, P1, P2 = calib(pathL, pathR, SAVE=False, DISPLAY=False,row=6,col=9)
# mapx1, mapy1, mapx2, mapy2, Q, P1, P2 = calib(pathL, pathR, SAVE=False, DISPLAY=False,row=6,col=10)
print "Finished Calibration"

# mapx1, mapy1, mapx2, mapy2, Q, P1, P2 = readCalibParamters()
# print "Reading parameters"

if __name__ == "__main__":
    rectify(pathL, pathR, SAVE_RES=False)

# Dukha Paayema
# Check number of rows and columns
# Check the folder paths
