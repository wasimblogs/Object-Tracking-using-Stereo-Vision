"""
TRY TO TRACK AS MANY OBJECTS POSSIBLE

Merging rectangles reduced the count of objects by 80 in 400 frames. 606-527
Reduced the number of blobs by filtering it by rectangle width and height

TRACKING 6 Works

What's new in NewTracking
Tracks by using SURF feature descriptors

Solve
    Could assign two objects in a frame to same object model

What's new in Tracking Scratch 2?
    Similar objects overlap. This version exploits the fact. Done

What's new in Tracking Scratch 3?
    An object is most similar to the most recent version of the same object.

What's new in Tracking Scratch 4?
    What happens when a permanent object is not detected in a frame.
    We search for it in the places it's been.

What's new in TrackingScratch5?
    Well I don't need to store rectangles and their ids to search lost objects.
    I can just retrive them from the object models.
    Efficient !

What's new in TrackingScratch6?
    Two different objects are not completely overlapping.

What's new in TrackingScratch7?
    Bit elegant about uniquess of objects

# Uniquess can be used to reduce computational complexity
# Can be used in applications which do not require high accuracy
"""

import cv2
import numpy as np

import TrackElements2 as te
import TrackingFeature2 as tf
import Scratch7a as calibration
import os
# COUNTERS
# Deciding when the object is no longer in the video
lastSeenPermObj = [-1000, -1000, -1000, -1000]
lastSeenTempObj = [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]

# HOW MANY TEMP OBJECTS SHOULD BE SEEN BEFORE THEY ARE MADE PERMANENT
tempSeenCount = [0, 0, 0, 0, 0, 0, 0, 0]
identicalObjects = 0

# Keypoints and descriptors storage for temp and perm objects
tempObjectsDescriptor = [[], [], [], [], [], [], [], []]
permObjectsDescriptor = [[], [], [], [],[],[]]

PERM_OBJ_COLOR = [(0, 0, 0), (100, 100, 100), (200, 200, 200), (255, 255, 255)]
TEMP_OBJ_COLOR = [(0, 0, 150), (0, 0, 200), (200, 0, 200), (255, 0, 255), (200, 0, 200),
                  (255, 0, 255), (200, 0, 200), (255, 0, 255)]


def isUniqueToPermObject(r, UNIQUE_THRESH=0.2):
    """
    :param r: rectangle of the temporary object or permanent object
    :return: True if unique object False if identical object
    """
    for objects in permObjectsDescriptor:
        for kp0, des0, r0 in objects[-5:]:
            a = tf.blobMovement(r0, r)
            if a < UNIQUE_THRESH :
                return False
    return True


# History state 15-22 frame
def objectModels(image, frame,rectangles,HISTORY=10, SIMILARITY_THRESH_PERM=1.5,
                 SIMILARITY_THRESH_TEMP=1.5, quitSimilarityThresh=3,
                 VANISH_THRESH_PERM=30, VANISH_THRESH_TEMP=15,
                 TEMP_UPGRADE_TO_PERM_THRESH=6, PERM_OBJECTS_COUNT=4,
                 TEMP_OBJECTS_COUNT=8, WAIT_TIME=01, FEATURE_WEIGHT=0.3):
    """
    Finds histogram of the blobs/objects and groups similar objects
    Keeps track of objects during an entire video/feed

    :param image    : left image of the stereo pair
    :param HISTORY  : No of latest histograms in an object model
    :return         : None

    frame = disparity
    rectangles are rects of objects
    """

    # Constants
    # Display status
    print "No. of objects :", len(rectangles)
    # print lastSeenPermObj
    # print lastSeenTempObj
    # UPDATING THE LAST SEEN RECORDS
    # INCREASES AFTER EVERY FRAME
    for i in xrange(0, len(lastSeenPermObj)):
        lastSeenPermObj[i] += 1

    for i in xrange(0, len(lastSeenTempObj)):
        lastSeenTempObj[i] += 2


    max_scores_list_perm = []  # MATCHES WITH PERMANENT OBJECTS
    max_scores_list_temp = []  # MATCHED WITH TEMP OBJECTS

    # Assign the newly detected object to one of three kinds of objects
    # Existing Temp object
    # Existing Perm Object
    # New temp obj
    for r in rectangles:
        x, y, w, h = r
        roi = image[y:y + h, x:x + w]
        kp, des = tf.calcSURF(roi)

        # Compare the newly detected object with each object models
        permScoreWeighted = [10, 10, 10, 10]

        for i in xrange(0, len(permObjectsDescriptor)):

            #  if the permanent object list is not empty
            if len(permObjectsDescriptor[i]) != 0:

                tempBiased = 0
                overlap = 0
                recentBias = tf.calcRecentBias(len(permObjectsDescriptor[i]))

                j = 0
                print

                for kp1, des1, r1 in permObjectsDescriptor[i]:
                    feature_score = tf.matchScore(des, des1, kp, kp1)
                    overlap_score = tf.blobMovement(r, r1)
                    total_score = FEATURE_WEIGHT * feature_score + (1 - FEATURE_WEIGHT) * overlap_score
                    tempBiased += recentBias[j] * total_score
                    # print "Overlap ", overlap_score, i, "\tFeature ", feature_score, i
                    j += 1

                    if overlap_score >= 0.81:
                        tempBiased = 100000

                avg = (tempBiased * 1.0) / (len(permObjectsDescriptor[i]))
                permScoreWeighted[i] = avg

        # Let's assign the object to a perm object model
        m = np.min(permScoreWeighted)
        print "Perm match score Unbiased ",m, "\tThreshold : ", SIMILARITY_THRESH_PERM
        if m < SIMILARITY_THRESH_PERM:
            index = permScoreWeighted.index(m)
            permObjectsDescriptor[index].append((kp, des, r))
            lastSeenPermObj[index] = 0
            print("Direct Match To Permanent Model {} \tStrength : {}".format(index, len(permObjectsDescriptor[index])))
            image = tf.markObject(image, index, r, PERM_OBJ_COLOR[index], 2)
            frame = tf.markObject(frame, index, r, PERM_OBJ_COLOR[index], 2)

        # If the detected object was not similar to perm object model

        else:
            unique = isUniqueToPermObject(r,0.3)
            if unique:
                # Compare the newly detected object with each temp object models
                tempScoreWeighted = [10, 10, 10, 10, 10, 10, 10, 10]
                for i in xrange(0, len(tempObjectsDescriptor)):

                    #  if the permanent object list is not empty
                    if len(tempObjectsDescriptor[i]) != 0:

                        tempBiased = 0
                        overlap = 0
                        recentBias = tf.calcRecentBias(len(tempObjectsDescriptor[i]))
                        j = 0

                        for kp1, des1, r1 in tempObjectsDescriptor[i]:
                            feature_score = tf.matchScore(des, des1, kp, kp1)
                            overlap_score = tf.blobMovement(r, r1)
                            total_score = FEATURE_WEIGHT * feature_score + (1 - FEATURE_WEIGHT) * overlap_score

                            tempBiased += recentBias[j] * total_score
                            j += 1

                            if overlap_score >= 0.81:
                                tempBiased = 100000
                                break

                            # print "Overlap ", overlap_score, i, "\tFeature ", feature_score, i
                            overlap += overlap_score
                        l = len(tempObjectsDescriptor[i])
                        avg = (tempBiased * 1.0) / l
                        tempScoreWeighted[i] = avg
                m = np.min(tempScoreWeighted)

                # If detected object matches to temp model match assign it to the object model
                # print "\nmax temp score weighted ", m, "/tThreshold :\t", SIMILARITY_THRESH_TEMP, tempScoreWeighted.index(m)
                if m < SIMILARITY_THRESH_TEMP and unique:
                    print "Temp Object Unique : ",unique

                    index = tempScoreWeighted.index(m)
                    tempObjectsDescriptor[index].append((kp, des, r))
                    tempSeenCount[index] += 1

                    lastSeenTempObj[index] -= 2

                    print(
                        "Object assigned to Temporary Model {} . Strength : {} LastSeen : {}".format(index,
                                                                                      len(tempObjectsDescriptor[index]), lastSeenTempObj[index]))
                    # image = tf.markObject(image, index, r, TEMP_OBJ_COLOR[index], 2)
                    frame = tf.markObject(frame, index, r, TEMP_OBJ_COLOR[index], 8)

                    # Just because an object in this frame doesn't mean it has been seen
                    # consecutively seen in the preceeding frames

                # If the object didn't even match to temp object, it must be a new temp object
                else:
                    # Looking for empty temp object models
                    for i in xrange(0, len(permObjectsDescriptor)):

                        #  if the permanent object list is not empty
                        if len(tempObjectsDescriptor[i]) == 0:
                            tempObjectsDescriptor[i].append((kp, des, r))
                            lastSeenTempObj[i] = 0

                            print("New temporary object Assigned to {} Strength : {}".format(i, len(
                                    tempObjectsDescriptor[i])))
                            # image = tf.markObject(image, i, r, TEMP_OBJ_COLOR[i], 2)
                            frame = tf.markObject(frame, i, r, TEMP_OBJ_COLOR[i], 8)
                            break
            else:
                global identicalObjects
                # print "Rejected. Because of proximity to a permanent object.....", identicalObjects
                identicalObjects += 1


    # Looking for temp objects which weren't detected in the frame
    id = 0
    for lostCount in lastSeenPermObj:

        # LostCount will be 0 if the object has been seen in this frame
        if lostCount >= 1 and len(permObjectsDescriptor[id]) >= TEMP_UPGRADE_TO_PERM_THRESH:
            # print "LOST OBJECT DETECTOR MODULE : ", id, lostCount
            weights = tf.calcRecentBias(len(permObjectsDescriptor[id]))

            # Initalize to almost infinte value because we look for min value when comparing
            lostScores = [10, 10, 10]
            for i in xrange(0, 3):
                # Looking for rectangles from latest to earliest

                # print "Object :",id, -3+i, len(permObjectsDescriptor[id])
                kp0, des0, r = permObjectsDescriptor[id][-3+i]
                x, y, w, h = r
                roi = image[y:y + h, x:x + w]

                kp_lost, des_lost = tf.calcSURF(roi)
                # image = tf.markObject(image, id, r, (0, 0, 255), 5)

                total = 0
                kk = 0
                for kp1, des1, r1 in permObjectsDescriptor[id][-5:]:
                    feature_score = tf.matchScore(des_lost, des1, kp_lost, kp1)
                    # total += weights[kk] * feature_score
                    total += 0.2 * feature_score

                    # print "Feature score :lost object :", feature_score, id
                lostScores[i] = total / (len(permObjectsDescriptor[id]))

            m = np.min(lostScores)
            index = lostScores.index(m)

            # print "Minimum Score for lost object: ", m
            # print lostScores

            if m < 0.5:
                kp00, des00, r = permObjectsDescriptor[id][(-3+index)]
                image = tf.markObject(image, id, r, PERM_OBJ_COLOR[id], 5)
                frame = tf.markObject(frame, id, r, PERM_OBJ_COLOR[id], 8)
                print "Harako Object mark gare la : ", id
                print "Harayeko object vetiyeko suchcna! :\t", id

                global kp_lost
                global des_lost
                permObjectsDescriptor[id].append((kp_lost, des_lost, r))
                lastSeenPermObj[id] = 0

        id += 1


    # TRANSFER OF TEMPORARY OBJECTS TO PERMANANET
    tempMax = np.max(tempSeenCount)

    if tempMax >= TEMP_UPGRADE_TO_PERM_THRESH:
        index = tempSeenCount.index(tempMax)

        assigned = False
        for i in xrange(0, PERM_OBJECTS_COUNT):
            # ASSIGN TEMP OBJECT TO A BLANK PERMANENT OBJECT
            if len(permObjectsDescriptor[i]) == 0 and not assigned:
                permObjectsDescriptor[i] = tempObjectsDescriptor[index]

                kp, des, r = tempObjectsDescriptor[index].pop()

                tempObjectsDescriptor[index] = []

                tempSeenCount[index] = 0
                lastSeenTempObj[index] = -1000
                lastSeenPermObj[i] = 0

                print index, " UPGRADED TO PERMANENT\t", i, "Length : ", len(permObjectsDescriptor[i])
                assigned = True

                image = tf.markObject(image, i, r,thick=5)
                frame = tf.markObject(frame, i, r,thick=5)
                print "Upgrade gareko pani mark garde la ", i

        if not assigned:
            print "TEMP OBJ COULD NOT BE MADE PERMANENT. STACK FULL"


    # REMOVING OBJECTS WHICH WERE SEEN LONG TIME BEFORE
    vanishMaxPerm = np.max(lastSeenPermObj)
    index = lastSeenPermObj.index(vanishMaxPerm)
    if np.max(lastSeenPermObj) >= VANISH_THRESH_PERM and len(permObjectsDescriptor[index]) != 0:
        # Compare with the list of temp objects to see if there is a temp object for it.

        index = lastSeenPermObj.index(vanishMaxPerm)
        index = lastSeenPermObj.index(np.max(lastSeenPermObj))
        permObjectsDescriptor[index] = []
        lastSeenPermObj[index] = 0
        print("Deleting perm object {}".format(index))


    # CLEANING UNLIKELY TEMP OBJECTS FROM STACK
    vanishMaxTemp = np.max(lastSeenTempObj)
    if vanishMaxTemp >= VANISH_THRESH_TEMP:
        # Compare with the list of permanent objects to see there is a permanent object like it.
        index = lastSeenTempObj.index(vanishMaxTemp)
        tempObjectsDescriptor[index] = []
        lastSeenTempObj[index] = 0
        tempSeenCount[index] = 0
        print("Deleting Temp Object {}".format(index))


    # Removing the old keypoints and descriptors from permanent object model.
    histories = [len(x) for x in permObjectsDescriptor]
    max = np.max(histories)
    if max > HISTORY:
        index = histories.index(max)
        for i in xrange(0, max - HISTORY):
            del permObjectsDescriptor[index][0]
            # print "Removing old histograms from object models"

    # If the image is grayscale
    if frame.size == frame.shape[0]*frame.shape[1]:
        # frame = cv2.merge((frame,frame,frame))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pass

    win = "DISPARITY"

    # output = np.hstack((frame, image))
    # cv2.imshow(win, output)
    # cv2.imshow(win+"..", frame)
    # cv2.imshow(win+"....", image)
    # k = cv2.waitKey(WAIT_TIME)

    output = image
    return output


# ----------------------------------------------------------------------------------------------------------------------
def go(leftFile, rightFile, SAVE_RESULTS=False, NO_OF_FRAMES=10000, TRACKING=True):
    # CALIBRATION
    # THE ORDER OF THE CAMERA IS CRITICAL

    if SAVE_RESULTS:
        # Name formation
        dir, file = os.path.splitext(leftFile)
        rectifiedName = dir + "__Rectified" + file
        disparityName = dir + "__Disparity" + file
        trackVideoName = dir + "__Tracking" + file
        # resDisparity = cv2.VideoWriter(disparityName, -1, 25, (640, 480))
        # resRectify = cv2.VideoWriter(rectifiedName, -1, 25, (640, 480))
        trackVideo = cv2.VideoWriter(trackVideoName, -1, 10, (640*2, 480))
        # print resDisparity.isOpened()
        # print resRectify.isOpened()

    left = cv2.VideoCapture(leftFile)
    right = cv2.VideoCapture(rightFile)


    # COUNTERS
    frame_count = 0
    noObjectFrameCount = 0
    rejectedObjects = 0

    # Skip Frames
    for i in xrange(0, 230):
        retL, frameL = left.read()
        retR, frameR = right.read()



    while left.isOpened() and right.isOpened():
        # try:
        retL, frameL = left.read()
        retR, frameR = right.read()

        frame_count += 1

        # frameL1 = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        # frameR1 = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        if retL and retR:
            rectifiedL, rectifiedR, disparity, depthMap= calibration.rectifyV(frameR, frameL, FIND_DEPTH_OBJECTS=False)

            inputStereo = np.hstack((frameL, frameR))
            rectified = np.hstack((rectifiedL, rectifiedR))

            # Optional
            disparityD = disparity/np.max(disparity)

##            cv2.imshow('DISPARITY D', disparityD)
##            cv2.imshow("DEPTH MAP", depthMap)
##            k = cv2.waitKey(000)
##            if k == 27:
##                break
##
##            if k == ord("S") or k == ord("s"):
            name = "G:/Stereo/New/"
            num = np.random.randint(0,1000)
            frameNo = str(frame_count)
##
##                cv2.line(rectified, (0,120),(640*2,120),(255,255,255),3)
##                cv2.line(rectified, (0,120*2),(640*2,120*2),(255,255,255),3)
##                cv2.line(rectified, (0,120*3),(640*2,120*3),(255,255,255),3)
##
##                cv2.line(inputStereo, (0,120),(640*2,120),(255,255,255),3)
##                cv2.line(inputStereo, (0,120*2),(640*2,120*2),(255,255,255),3)
##                cv2.line(inputStereo, (0,120*3),(640*2,120*3),(255,255,255),3)
##
##
            filenameRect = name + "__Rectified__"+frameNo+".jpg"
            # filenameDisp = name + "__Disparity__"+frameNo+".jpg"
            filenameInput = name + "__Input__"+frameNo+".jpg"
            # filenameDepth = name + "__Depth__" + frameNo + ".jpg"
##
            cv2.imwrite(filenameRect, rectified)
            # cv2.imwrite(filenameDisp, disparity)
            cv2.imwrite(filenameInput, inputStereo)
                # cv2.imwrite(filenameDepth, depthMap)


            # TRACKING
            # cv2.imwrite("G:/Filters/Disparity.jpg", disparity)
            rectangles = te.blobs(disparity, DILATE=3, FRAME_NO=frame_count)
            l = len(rectangles)

            # Keeping track of no of frames in which no objects are detected.
            if l == 0: noObjectFrameCount += 1

            if TRACKING:
                output = objectModels(rectifiedL,disparity, rectangles)
                temp = cv2.imwrite("G:Filters/temp.jpg", disparity)
                disparity = cv2.imread("G:/Filters/temp.jpg",1)

                output = np.hstack((disparity, output))
                cv2.imshow("Track", output)
                cv2.imshow("disp", disparityD)
                cv2.waitKey(10)

                if SAVE_RESULTS:
                    trackFileName = name + "__Tracked__"+frameNo+".jpg"
                    cv2.imwrite(trackFileName, output)
                    # resDisparity.write(disparity)
                    # resRectify.write(rectifiedL)
                    trackVideo.write(output)

        # except:
        #     print 'End of Video File Reached'
        #     break

        if frame_count > NO_OF_FRAMES:
            break


    # when everything done, release the capture
    left.release()
    right.release()
    # resDisparity.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    left = 'G:/Stereo/DanceR9.avi'
    right = 'G:/Stereo/DanceL9.avi'

    # Order of the video is critical
    left = 'G:/Stereo/New10L.avi'
    right = 'G:/Stereo/New10R.avi'
    go(left, right, SAVE_RESULTS=True, TRACKING=True)
