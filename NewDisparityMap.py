# ALGORITHM
# Prefilter to normalize image brightness and enhance texture
# Correspondence search along horizontal epipolar lines using a SAD window
# Post filtering to eliminate bad correspodence matches

#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

import numpy as np
import cv2
import glob
import PCD_WRITE as pcd

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


def findDepth(disparity):
    h, w = disparity.shape[:2]
    for i in xrange(0,h):
        for j in xrange(0, w):
            x,y,w = i, j, 1
            z = disparity[i][j]
            arr = np.array((x,y,z,w), np.float32)
            res = Q * arr
            print res



    return disparity

# LATEST

# folderL = glob.glob("G:\Stereo\Input\Nleft\*.*")
# folderR = glob.glob("G:\Stereo\Input\Nright\*.*")

folderR = glob.glob("G:\Stereo\input/Rectifiedl1\*.*")
folderL = glob.glob("G:\Stereo\input/Rectifiedr1\*.*")

if len(folderL) ==0: print 'No File to find 3D\nShould not continue'

folderL.sort()
folderR.sort()

if __name__ == '__main__':
    print 'loading images...'
    # imgL = cv2.pyrDown( cv2.imread('../gpu/aloeL.jpg') )  # downscale images for faster processing
    # imgR = cv2.pyrDown( cv2.imread('../gpu/aloeR.jpg') )

    # Parameters for disparity
    window_size = 3
    min_disp = 16*1
    num_disp = 16*8-min_disp
    stereo = cv2.StereoSGBM(minDisparity = min_disp,        #Max diff - 1
            numDisparities = num_disp,
            SADWindowSize = window_size,
            uniquenessRatio = 11,
            speckleWindowSize = 100,
            speckleRange = 1,
            disp12MaxDiff = 10,
            # P1 = 8*3*window_size**2,
            # P2 = 32*3*window_size**2,
            P1 = 20*window_size**2,
            P2 = 150*window_size**2,
            fullDP = False,
            preFilterCap=29)

    print 'computing disparity...'

    print 'Number of images :', len(folderL)
    for i in xrange(1, len(folderL)):

        # READ IMAGE IN COLORS
        print folderL[i]
        imageL = cv2.imread(folderL[i])
        imageR = cv2.imread(folderR[i])

        # imageL = cv2.imread('G:\Stereo\Disparity\RectifiedL.jpg')
        # imageR = cv2.imread('G:\Stereo\Disparity\RectifiedR.jpg')

        print imageL.shape == imageR.shape
        print len(folderL[i]), len(folderR[i])
        disp = stereo.compute(imageL, imageR).astype(np.float32) / 16.0

        print 'generating 3d point cloud...',
        h, w = imageL.shape[:2]
        f = 0.8*w                          # guess for focal length
        Q = np.float32([[1, 0, 0, -0.5*w],
                        [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                        [0, 0, 0,     -f], # so that y-axis looks up
                        [0, 0, 1,      0]])


        # LATEST INPUT
        Q = np.float32([[1, 0,   0, -837.65],
                        [0, 1,  0,  -265.07],
                        [0, 0,  1,  987.765],
                        [0, 0,  0.10335,    0]])

        # # LATEST INPUT LEFT4
        Q = np.float32([[1, 0,   0, -938.275],
                        [0, 1,  0,  -310.741],
                        [0, 0,  1,  784.007],
                        [0, 0,  -0.2519,    0]])

        # # STEREO/INPUT/LEFT1
        Q = np.float32([[1, 0,   0, -633.11],
                        [0, 1,  0,  -206.26],
                        [0, 0,  1,  777.67],
                        [0, 0,  -0.31865,    0]])

        # MANITA DISPARITY
        # Q = np.float32([[1, 0,   0, -291.91],
        #                 [0, 1,  0,  -231.39],
        #                 [0, 0,  1,  863.92],
        #                 [0, 0,  -0.2559,    0]])

        points = findDepth(disp)
        # points = cv2.reprojectImageTo3D(disp, Q)
        # colors = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)
        # mask = disp > disp.min()
        # out_points = points[mask]
        # out_colors = colors[mask]

        # pcd.pcd_write_color(out_points, out_colors, len(out_points), "G:/Stereo/B1.pcd")
        # write_ply(folderL[i]+'.ply', points, colors)
        # print '%s saved' % 'out.ply'

        cv2.destroyAllWindows()
        cv2.imshow('left', imageL)
        cv2.imshow('Right', imageR)
        cv2.imshow('disparity', (disp-min_disp)/num_disp)
        cv2.imshow("POINTS", points)
        # cv2.imshow('Points', out_points)
        # cv2.imshow("Colors", out_colors)
        k = cv2.waitKey(0)
        if k == 27: break

        # print "\n\nPrinting Points . . ."
        # print points[0][0]
        # print points[479][0]
        # print points[0][639]
        # print points[479][639]
        # #
        # print out_points.shape
        # for i in xrange(0, 640):
        #     for j in xrange(0,480):
        #         print out_points[j][i]
        # print np.average(points)
