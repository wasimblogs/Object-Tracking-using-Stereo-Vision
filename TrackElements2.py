import cv2
import numpy as np
"""
contains feature extraction
    Edge Histogram
    Color Histogram
Blob detection
Blob merge
"""

# BLOB DETECTION PARAMTERS
BG_AREA_CONTOUR = 0.35
NOISE_AREA_CONTOUR = 0.036

# Used during prefiltering blobs
NOISE_BLOB = 0.05
BACKGROUND_BLOB = 0.8
blobArea = []

def suddenChange(new, old):
    """
    Calculates changes between frames. Meant to detect sudden changes between frames
    :param new: new frame
    :param old: old frame
    :return: changeCount, sum
    """
    r, c = new.shape[:2]
    row, col = 4, 4
    width = c / col
    height = r / row

    startW = 0l
    changeCount = 0
    sum = 0
    for i in xrange(0, col):
        startH = 0
        for j in xrange(0, row):
            roiNew = new[startH:startH + height, startW: startW + width]
            roiOld = old[startH:startH + height, startW: startW + width]
            diff = cv2.subtract(roiNew, roiOld)
            score = np.sum(diff) / (roiNew.size * 255.0)
            sum += score
            if score > 0.10:
                changeCount += 1
            startH = startH + height
        startW = startW + width
        return changeCount, sum


def pyrEdgeF(image):
    hist = []
    pyramidNumber = 1
    col = 8
    row = 8
    for k in xrange(0,pyramidNumber):
        histL = []
        query = cv2.Canny(image, 0, 00)
        histL.append(np.sum(query)/(query.size*255.0))
        r, c = image.shape[:2]
        width = c/col
        height = r/row
        startW = 0
        for i in xrange(0,col):
            correct = False
            startH = 0
            for j in xrange(0,row):
                roi = image[startH:startH+height, startW: startW+width]
                startH = startH + height
                edge_Content = np.sum(roi)/(roi.size*255.0)
                histL.append(edge_Content)
            startW = startW + width

            if roi.size == 0:
                correct = True
                break

        if correct:
            # print 'Corrected ',i,j
            histL = []
            for i in xrange(0,col*row):
                histL.append(0.001)
            # print 'Before Normalization\t',histL
            histL = np.ravel(histL)
            histL = cv2.normalize(histL)
            # print 'After normalization \t', histL

        histL = np.ravel(histL)
        histL = cv2.normalize(histL)
        hist.append(histL)
        image = cv2.pyrDown(image)

    hist = np.float32(hist)
    hist = np.ravel(hist)
    return hist


# COLOR DISTRIBUTION ONLY
def calcHistogramColor(image, BINS=10):
    """
    Calculates histogram of given image
    :param image    : image whose histogram is to be calculated
    :param BINS     : no of bins for histogram
    :return: np.ravel(hist)
    """
    # BINS =10
    hist = []

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    Y, CB, CR = cv2.split((image))

    histCB = cv2.calcHist(CB, [0], None, [BINS], [0, 256])
    histCB = cv2.normalize(histCB)
    hist.append(histCB)

    histCR = cv2.calcHist(CR, [0], None, [BINS], [0, 256])
    histCR = cv2.normalize(histCR)
    hist.append(histCR)

    # r = cv2.calcHist([image], [0], None, [BINS], [0, 256])
    # r = cv2.normalize(r)
    # hist.append(r)
    #
    # g = cv2.calcHist([image], [1], None, [BINS], [0, 256])
    # g = cv2.normalize(g)
    # hist.append(g)
    #
    # b = cv2.calcHist([image], [2], None, [BINS], [0, 256])
    # b = cv2.normalize(b)
    # hist.append(b)

    return np.ravel(hist)


def calcLocalizedHistogram(image, BINS=10):
    # def calcHistogramColor(image, BINS=10):
    """
    Calculates localized histogram
    :param image:
    :param BINS:
    :return:
    """
    bins = 10
    hist = []

    r, c = image.shape[:2]

    GRID_ROW = 1
    GRID_COL = 1

    width = c / GRID_COL
    height = r / GRID_ROW

    startW = 0
    for i in xrange(0, GRID_COL):
        startH = 0
        for j in xrange(0, GRID_ROW):
            roi = image[startH:startH + height, startW: startW + width]
            startH = startH + height

            Y, CB, CR = cv2.split((image))
            histCB = cv2.calcHist(CB, [0], None, [BINS], [0, 256])
            histCB = cv2.normalize(histCB)
            hist.append(histCB)

            histCR = cv2.calcHist(CR, [0], None, [BINS], [0, 256])
            histCR = cv2.normalize(histCR)
            hist.append(histCR)

            # r = cv2.calcHist([roi], [0], None, [bins],[0,256])
            # r =cv2.normalize(r)
            # hist.append(r)
            # g = cv2.calcHist([roi], [1], None, [bins],[0,256])
            # g = cv2.normalize(g)
            # hist.append(g)
            # b = cv2.calcHist([roi], [2], None, [bins],[0,256])
            # b = cv2.normalize(b)
            # hist.append(b)

        startW = startW + width
    return np.ravel(hist)


def overlapArea(r1, r2):
    """
    calculates overalp between two rectangles
    :param r1   : reectangle-1
    :param r2   : rectangle-2
    :return     : overlap area
    """

    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    # Centers of objects
    xc1 = (2 * x1 + w1) / 2
    xc1 = (2 * x2 + w2) / 2
    yc1 = (2 * y1 + y1) / 2
    yc2 = (2 * y2 + y2) / 2

    yl = (y1 >= y2) and (y1 <= (y2 + h2))
    yu = ((y1 + h1) >= y2) and ((y1 + h1) <= (y2 + h2))
    xl = (x1 >= x2) and (x1 <= (x2 + w2))
    xu = ((x1 + w1) >= x2) and ((x1 + w1) <= (x2 + w2))

    x, y, w, h = 0, 0, 0, 0
    if yl and yu:
        y, h = y1, h1

    elif yl and not yu:
        y, h = y1, h2 + y2 - y1

    # else (not y1) and yu:
    elif not y1 and yu:
        y, h = y2, (y1 + h1 - y2)

    else:
        # print "DEFAULT.."
        pass

    if xl and xu:
        x, w = x1, w1

    elif xl and not xu:
        x, w = x1, (x2 + w2 - x1)

    elif not xl and xu:
        x, w = x2, (x1 + w1 - x2)

    else:
        # print "DEFAULT"
        pass

    return min(h * w * 1.0 / (h1 * w1), h * w * 1.0 / (h2 * w2))


def mergeRect(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    x = min(x1,x2)
    y = min(y1,y2)
    h = max(h1,h2)
    w = max(w1,w2)

    return [x,y,h,w]


def blobs(image, OVERLAP_THRESH=0.6, GRAYSCALE=False, DILATE=0):
    """
    Returns rectangles of blobs and images marked with blobs
    :param image    : image where blob/objects are to be detected
    :return rects : list of rectangles bounding detected blobs
    """

    # if not GRAYSCALE:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     # cv2.imshow("Blob for disp", image)
    #     # cv2.waitKey(0)

    # Change the image into grayscale if it is RGB image
    if image.size == image.shape[0]*image.shape[1]:
        pass
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # DEFINE THRESHOLDING CRITERIA FOR BLOB DETECTION
    startThresh = 30
    endThresh = 240
    THRESH_STEP = 30

    # STORES RECTANGLES of blobs
    rects = []

    # THRESHOLDING FOR BLOB DETECTION
    j = -10  # for consecutive blob detection


    for i in xrange(0, int((1.0 * endThresh) / THRESH_STEP)):

        filter = cv2.inRange(image, startThresh, startThresh + THRESH_STEP)

        # # Debugging
        # cv2.imshow("Filter", filter)
        # cv2.waitKey(100)

        # dilated = cv2.erode(filter, None, iterations=3)
        if DILATE:
            filter = cv2.dilate(filter, None, iterations=DILATE)

        # filter = dilated
        blobScore = np.sum(filter) / (image.size * 255.0)
        if blobScore > NOISE_BLOB and blobScore < BACKGROUND_BLOB:
            contours, _ = cv2.findContours(filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c) / (filter.size)

                # IDENTIFYING INDIVIDUAL BLOBS
                # THE DETECTED BLOB'S BEEN ACCEPTED
                if area > NOISE_AREA_CONTOUR and area < BG_AREA_CONTOUR:

                    # print "BLOB AREA:\t\t\t\t\t\t\t\t\t",area
                    blobArea.append(area)
                    (x, y, w, h) = cv2.boundingRect(c)
                    if w < 0.6*640 and h < 0.85*480:
                        r = cv2.boundingRect(c)

                        # print "BLOB : ", i, "N : ", detectedN,"\tO : ",detectedO
                        diff = i - j
                        j = i
                        if len(rects) > 0:
                            oldR = rects.pop()

                            # LOOKING FOR A DUPLICATE BLOB
                            if oldR == r or overlapArea(r, oldR) > OVERLAP_THRESH or overlapArea(oldR, r) > OVERLAP_THRESH:
                                rects.append(oldR)
                                # mergedRect= mergeRect(oldR, r)
                                # print "Merged!"
                                # rects.append(mergedRect)
                            # ORIGINAL BLOB DETECTED
                            else:

                                # IF CONSECUTIVE BLOBS HAVE BEEN DETECTED, MERGE THEM
                                if diff == 1:
                                    # THIS REQUIRES TUNING
                                    merge = cv2.inRange(image, startThresh, startThresh + THRESH_STEP + 10)
                                    mergeContours, _ = cv2.findContours(filter, cv2.RETR_EXTERNAL,
                                                                        cv2.CHAIN_APPROX_SIMPLE)
                                    for c in mergeContours:
                                        area = cv2.contourArea(c) / (filter.size)
                                        if area > NOISE_AREA_CONTOUR and area < BG_AREA_CONTOUR:
                                            r = cv2.boundingRect(c)
                                            rects.append(r)
                                else:
                                    rects.append(oldR)
                                    rects.append(r)

                        else:
                            rects.append(r)
                    else:
                        # print "rejected", w, 0.6*640,h,0.85*h
                        pass

                else:
                    detectedN = False
                    pass

        else:
            pass

        startThresh += THRESH_STEP
    return rects

def drawBlobs(image, rects):

    for r in rects:
        x,y,w,h = r
        # print x,y,w,h
        cv2.rectangle(image, (x,y), (x+w,y+h),(255,255,255),2)

    return image


if __name__ == "__main__":
        video = cv2.VideoCapture("G:/Stereo/Disparity9.avi")

        no_of_blobs = 0
        while 1:
            ret, frame = video.read()
            if ret:
                rects = blobs(frame, 0.6)
                l = len(rects)
                no_of_blobs += l
                # print l, no_of_blobs
                res = drawBlobs(frame, rects)
                cv2.imshow("BLOBS", res)
                cv2.waitKey(0)
