import cv2
import numpy as np
import matplotlib.pyplot as plt
from helpers import (rotationMatrixToEulerAngles)

def calc_drone_vel(intr, frame_prev, frame_cur, bounding_boxes):
    # Get key features
    #sift = cv2.xfeatures2d.SIFT_create()
    #surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1500)

    kp_cur, desc_cur = orb.detectAndCompute(frame_cur, None)
    kp_prev, desc_prev = orb.detectAndCompute(frame_prev, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors. Hopefully improve RANSAC performance
    matches = bf.match(desc_prev,desc_cur)

    # # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # Debug
    img = cv2.drawMatches(frame_prev, kp_prev, frame_cur, kp_cur, matches, None)
    plt.imshow(img),plt.show()

    kpc = np.empty((len(matches), 2))
    kpp = np.empty((len(matches), 2))
    for i in range(len(matches)):
        # print('-----')
        # print(matches[i].imgIdx)
        # print(matches[i].queryIdx)
        # print(matches[i].trainIdx)

        # FIXME - backwards?
        kpc[i, 0] = kp_cur[matches[i].queryIdx].pt[0]
        kpc[i, 1] = kp_cur[matches[i].queryIdx].pt[1]
        kpp[i, 0] = kp_prev[matches[i].trainIdx].pt[0]
        kpp[i, 1] = kp_prev[matches[i].trainIdx].pt[1]

    # Calculate the essential matrix
    focal = (intr[0, 0] + intr[1, 1]) / 2.0
    oc = (intr[0, 2], intr[1, 2])
    E, mask = cv2.findEssentialMat(kpc, kpp, focal=focal, pp=oc, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    retval, R, t, mask = cv2.recoverPose(E, kpc, kpp, intr, mask);

    # TODO - convert to velocities using frame rate
    # TODO - validate output

    # print(R)
    # print(rotationMatrixToEulerAngles(R))
    # print(t)


    pass


def calc_disparity_map(intr, frame_prev, frame_cur):
    pass

def calc_pedestrian_position(intr, pedestrians, disparity):
    pass

def calc_pedestrian_velocity(intr, pedestrians, disparity):
    pass

def calc_abs_vel(intr, pedestrian_pos, pedestrian_vel, drone_vel):
    pass