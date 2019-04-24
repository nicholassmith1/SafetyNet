import cv2
import numpy as np
import matplotlib.pyplot as plt
from helpers import (rotationMatrixToEulerAngles)


def get_matching_points(frame_prev, frame_cur):
#sift = cv2.xfeatures2d.SIFT_create()
    #surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1500)

    # print('frame dim={}'.format(frame_cur.shape))

    kp_cur, desc_cur = orb.detectAndCompute(frame_cur, None)
    kp_prev, desc_prev = orb.detectAndCompute(frame_prev, None)

    # TODO - remove any key features in bounding boxes with
    # people in them

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors. Hopefully improve RANSAC performance
    matches = bf.match(desc_prev, desc_cur)

    # # FLANN parameters
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary

    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(desc_prev,desc_cur,k=2)

    # # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    kpc = np.empty((len(matches), 2))
    kpp = np.empty((len(matches), 2))

    matches = matches[:10]

    for i in range(len(matches)):
        # print('-----')
        # print(matches[i].imgIdx)
        # print(matches[i].queryIdx)
        # print(matches[i].trainIdx)

        # FIXME - backwards?
        # print('{} {} {}'.format(matches[i].queryIdx, len(kp_cur), len(kp_prev)))
        # print(kp_cur[matches[i].queryIdx])
        # print(kp_cur[matches[i].queryIdx].pt)
        # kpc[i, 0] = kp_cur[matches[i].queryIdx].pt[0] # this was backwards
        # kpc[i, 1] = kp_cur[matches[i].queryIdx].pt[1]
        # kpp[i, 0] = kp_prev[matches[i].trainIdx].pt[0]
        # kpp[i, 1] = kp_prev[matches[i].trainIdx].pt[1]
        kpc[i, 0] = kp_cur[matches[i].trainIdx].pt[0]
        kpc[i, 1] = kp_cur[matches[i].trainIdx].pt[1]
        kpp[i, 0] = kp_prev[matches[i].queryIdx].pt[0]
        kpp[i, 1] = kp_prev[matches[i].queryIdx].pt[1]

    # Debug
    if False:
        # print(matches[-1:])
        # kpc[0, 0] = kp_cur[matches[-1:][0].trainIdx].pt[0]
        # kpc[0, 1] = kp_cur[matches[-1:][0].trainIdx].pt[1]
        # kpp[0, 0] = kp_prev[matches[-1:][0].queryIdx].pt[0]
        # kpp[0, 1] = kp_prev[matches[-1:][0].queryIdx].pt[1]
        # frame_prev = cv2.resize(frame_prev, None, fx=0.5, fy=0.5)
        # frame_cur = cv2.resize(frame_cur, None, fx=0.5, fy=0.5)

        print(matches[0])
        print('{} {}'.format(kpp[0], kpc[0]))
        img = cv2.drawMatches(frame_prev, kp_prev, frame_cur, kp_cur, matches, None)
        plt.imshow(img),plt.show()

    return kpp, kpc

def calc_drone_pos_delta(intr, frame_prev, frame_cur, bounding_boxes):
    """
    @return (R, t)      the translation and rotation matrices describing
                        the change in position from frame_prev to frame_cur.
    """

    # Get key features
    kpp, kpc = get_matching_points(frame_prev, frame_cur)

    

    # Calculate the essential matrix. Pose *should* be p2 to p1, in
    # p2 coordinates
    focal = (intr[0, 0] + intr[1, 1]) / 2.0
    oc = (intr[0, 2], intr[1, 2])
    # oc = (intr[1, 2], intr[0, 2]) # (row, col), rather than (x, y)? seems to minimize rotations
    E, mask = cv2.findEssentialMat(kpc, kpp, focal=focal, pp=oc, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    retval, R, t, mask = cv2.recoverPose(E, kpc, kpp, intr, mask);

    print('valid: {} / {}'.format(retval, len(kpc)))
    # print(R)
    print(rotationMatrixToEulerAngles(R))
    print(t)

    # # debug
    # if (retval / len(kpc)) < 0.5:
    #     # when this happens, the matches are noticably bad. Should consider rebuilding
    #     # with SIFT support?
    #     img = cv2.drawMatches(frame_prev, kp_prev, frame_cur, kp_cur, matches, None)
    #     plt.imshow(img),plt.show()


    return (retval / len(kpc)), R, t


def calc_disparity_map(intr, frame_prev, frame_cur):
    pass

def calc_pedestrian_position(intr, pedestrians, disparity):
    pass

def calc_pedestrian_velocity(intr, pedestrians, disparity):
    pass

def calc_abs_vel(intr, pedestrian_pos, pedestrian_vel, drone_vel):
    pass