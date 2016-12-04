import cv2
import numpy as np

import TrackElements2 as te

"""
What's new in TrackingFeature2
Elegant and more portable
"""

surf = cv2.SURF()


def markObject(image, id, r1, col=(255, 255, 255), thick=2):
    """
    :param image:
    :param id:
    :param r1:
    :param col:
    :param thick:
    :return:
    Draws rectangle around objects
    """
    x, y, w, h = r1
    cv2.rectangle(image, (x, y), (x + w, y + h), col, 2)
    cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 0), 2)
    return image


def blobMovement(r1, r2):
    """
    :param r1:
    :param r2:
    :return:
    Measure distance or movement between two blobs
    """
    # image = np.zeros((640,480), np.uint8)
    #
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    # Center of first rectangle
    xc1 = (2 * x1 + w1) / 2.0
    xc2 = (2 * x2 + w2) / 2.0

    yc1 = (2 * y1 + h1) / 2.0
    yc2 = (2 * y2 + h2) / 2.0

    xd1 = abs(xc1 - xc2)
    yd1 = abs(yc2 - yc1)

    x11 = abs(x1 - x2)
    y11 = abs(y1 - y2)

    x22 = abs(x1 - x2 + w1 - w2)
    y22 = abs(y1 - y2 + h1 - h2)

    sums = x11 + x22 + y11 + y22
    sums /= 400.0
    centroidDistance = (xd1 + yd1) / 200.0

    # print "Corners Sum ", sums
    # print "Centroid : ", centroidDistance
    # print sums + centroidDistance

    # Score reflects how far are two blobs thus affecting if two blobs could be the same one
    return sums + centroidDistance


# Goodness bias should matter more than recent bias
def matchScore(des1, des2, kp, kp1):
    """
    :param des1:
    :param des2:
    :param kp:
    :param kp1:
    :return:
    Compute the feature score : match of features of two objects
    """
    # FLANN parameters for matching features
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    j = 0
    badScore, goodScore = 0, 0
    goodFeatureCount, badFeatureCount = 0, 0

    # 0.8 is default
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            mm = m.distance / n.distance
            goodScore += mm
            good.append(m)
            pts1.append(kp[m.queryIdx].pt)
            pts2.append(kp1[m.trainIdx].pt)
            j += 1
            goodFeatureCount += 1

            # if j == 100:
            #     break

        else:
            mm = m.distance / n.distance
            badScore += mm
            badFeatureCount += 1

    # print "\nMatch :", goodFeatureCount, goodScore / (goodFeatureCount + 1)
    # print "MisMatch :", badFeatureCount, (1 - badScore / (badFeatureCount + 1))

    goodBadRatio = (badFeatureCount) / (goodFeatureCount + 0.1)
    aa = badScore / (goodScore + 0.001)

    # if goodBadRatio > 2:
    #     print "Bad Match !", goodBadRatio, aa
    # else:
    #     print "Good Match !", goodBadRatio, aa

    return goodBadRatio


def calcSURF(roi):
    """
    :param roi: imag for which SURF features are to calculated
    :return: keypoint, descriptors
    """
    kp, des = surf.detectAndCompute(roi, None)
    return kp, des


def calcRecentBias(NO_OF_STATES=10):

    numbers = [x for x in xrange(1,NO_OF_STATES + 1)]
    norm = sum(numbers)
    weights = [(x*1.0)/norm for x in numbers]
    return weights


def calcGoodBias(NO_OF_STATES=20):
    pass


if __name__ == "__main__":

    dispVideo = cv2.VideoCapture("G:/Stereo/DanceR9__Disparity.avi")
    rectVideo = cv2.VideoCapture("G:/Stereo/DanceR9__Rectified.avi")

    while 1:

        list = []
        ret0, frame0 = rectVideo.read()
        retD, disparity0 = dispVideo.read()

        ret1, frame1 = rectVideo.read()
        retD, disparity1 = dispVideo.read()

        rects0 = te.blobs(disparity0)

        x, y, w, h = rects0[0]
        roi0 = frame0[y:y + h, x:x + w]
        kp, des = calcSURF(roi0)
        roi = cv2.drawKeypoints(roi0, kp)
        list.append(des)

        frame0[y:y + h, x:x + w] = roi
        drawn = te.drawBlobs(frame0, rects0)
        cv2.rectangle(frame0, (x, y), (x + w, y + h), (00, 0, 00), 2)

        rects1 = te.blobs(disparity1)

        x, y, w, h = rects1[0]
        roi1 = frame1[y:y + h, x:x + w]
        kp1, des1 = calcSURF(roi1)
        roi1 = cv2.drawKeypoints(roi1, kp1)
        list.append(des1)

        frame1[y:y + h, x:x + w] = roi1
        drawn1 = te.drawBlobs(frame1, rects1)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (00, 0, 00), 2)

        featureScore = matchScore(list[0], list[1], kp, kp1)
        overlapScore = blobMovement(rects0[0], rects1[0])
        # print "feature : ",featureScore ,"\tOverlap score :", overlapScore

        list = []
        cv2.imshow("INPUT", np.hstack((frame0, frame1)))
        k = cv2.waitKey(0)
        if k == 27: break
