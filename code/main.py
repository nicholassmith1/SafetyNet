

import numpy as np
import argparse
import constants
import logging
import signal
import sys
import cv2
import matplotlib.pyplot as plt
import pickle
from helpers import (rotationMatrixToEulerAngles)

from photogrammetry import (calc_drone_pos_delta,
        calc_disparity_map,
        calc_pedestrian_position,
        calc_pedestrian_velocity,
        calc_abs_vel)
from visualization import visualize
from segmentation import find_pedestrians
from helpers import (collate_pedestrians,
        classify_pedestrian,
        estimate_epicenter)
 
log = logging.getLogger(__name__)

# debug stuff
time = [0.0]
confidences = [0.0]
pos_x = [0.0]
pos_y = [0.0]
pos_z = [0.0]
# pos_R = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
rot_x = [0.0]
rot_y = [0.0]
rot_z = [0.0]

# last = 0 # debugging
# t = 0

def moving_average(data, N):
    return np.convolve(data, np.ones((N,))/N, mode='same')

def process_frame(intr, dt, frame_cur, frame_prev, bounding_boxes=None, out=None):
    pedestrians_prev = None  # FIXME
    epicenter_prev = None

    # global t
    # global last

    # t = t + dt

    # DEBUG skip ahead
    # if time[len(time) - 1] < 30:
    #     return



    # Get an array of pedestrian bounding boxes.
    if bounding_boxes is None:
        bounding_boxes = find_pedestrians(frame_cur)

    # Draw the bounding boxes
    if out is not None:
        frame = frame_cur
        for bb in bounding_boxes:
            frame = cv2.rectangle(frame,(bb[0],bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0,255,0), 3)
            # print((bb[0],bb[1]), (bb[0] + bb[2], bb[1] + bb[3]))
        out.write(frame)

    # Calculate the drone instanenous velocity, drone FOR
    confidence, R, t = calc_drone_pos_delta(intr, frame_prev, frame_cur, bounding_boxes)
    # debug stuff
    time.append(time[len(time) - 1] + dt)
    confidences.append(confidence)
    pos_x.append(t[0])
    pos_y.append(t[1])
    pos_z.append(t[2])
    rx, ry, rz = rotationMatrixToEulerAngles(R)
    rot_x.append(rx)
    rot_y.append(ry)
    rot_z.append(rz)
    # pos_R.append(R)
    # if (confidence > 0.5):
    #     pos_x.append(t[0])
    #     pos_y.append(t[1])
    #     pos_z.append(t[2])
    #     pos_R.append(R)
    # else:
    #     pos_x.append(pos_x[len(pos_x) - 1])
    #     pos_y.append(pos_y[len(pos_y) - 1])
    #     pos_z.append(pos_z[len(pos_z) - 1])
    #     pos_R.append(pos_R[len(pos_R) - 1])

    drone_vel = 0 # FIXME

    # Generate a depth map
    disparity = calc_disparity_map(intr, frame_prev, frame_cur)

    # Calculate the location of the ground, relative to the drone FOR,
    #ground_norm = estimate_ground(disparity)

    # Label bounding boxes as pedestrians. We must know both the current
    # and the previous position of the pedestrian to estimate an instantenous
    # velocity. 
    pedestrians = collate_pedestrians(pedestrians_prev, bounding_boxes)

    # FIXME
    # Calculate the position and velocity of each pedestrian in the drone FOR
    pedestrian_pos = calc_pedestrian_position(intr, pedestrians, disparity)
    pedestrian_vel = calc_pedestrian_velocity(intr, pedestrians, disparity)

    # Translate pedestrian velocities into real-world velocities
    pedestrian_vel_abs = calc_abs_vel(intr, pedestrian_pos, pedestrian_vel, drone_vel)

    # Classify each pedestrian as walking or jogging/running
    pedestrian_class = classify_pedestrian(pedestrian_vel_abs)

    # Predict an epicenter based on positions and velocities of pedestrians
    epicenter = estimate_epicenter(epicenter_prev, pedestrian_class, pedestrian_pos, pedestrian_vel_abs)

    visualize(intr, frame_cur, pedestrian_pos, pedestrian_vel, pedestrian_class, drone_vel, epicenter)

def parse_annotations(annotation_file):
    """
    Interperts the specified annotation_file as SingleActionTrackingLabel
    as specified by http://okutama-action.org/

    @returns    numpy Nx9 array of integers, per the specification
    """
    # np.loadtxt might work, but there's mixed int and strings...

    rtn = np.empty((0, 9), dtype=int)
    with open(annotation_file) as f:
        lines = f.readlines()
        rtn = np.empty((len(lines), 9), dtype=int)
        for i in range(len(lines)):
            s = lines[i].split(' ')
            rtn[i] = np.array([int(x) for x in s[:9]], dtype=int)
    return rtn

def signal_handler(sig, frame):
        log.info('Exitting')
        sys.exit(0)

def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='SafetyNet')
    parser.add_argument('video', metavar='VIDEOFILE', type=str,
                    help='The video to process')
    parser.add_argument('-a', dest='annotations', type=str, default=None,
                    required=False, help='Overrides the neural network calculation of pedestrian \
                    bounding boxes and uses annotations for bounding-box generation')
    args = parser.parse_args()

    # Install signal handlers for early termination
    signal.signal(signal.SIGINT, signal_handler)

    # Setup the logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.DEBUG)

    # FIXME ?
    # DJI Phantom-4 standard intrinsic matrix, per
    # https://stackoverflow.com/questions/47896876/camera-intrinsic-matrix-for-dji-phantom-4
    # intr = np.array([[1865.0, 0.0   , 2000.0],
    #                  [0.0   , 1865.0, 1500.0],
    #                  [0.0   , 0.0   , 1.0   ]])
    # Video input is (2160, 3840, 3). This looks like a phantom 4 (regular)
    # running in 4K mode, per https://www.dji.com/phantom-4/info#specs  
    intr = np.array([[2246.742, 0.0     , 1920.0],
                     [0.0     , 2246.742, 1080.0],
                     [0.0     , 0.0     , 1.0   ]])

    # # https://gist.github.com/dbaldwin/21d55b134d71fd66e06dce8b91651a03
    # intr = np.array([[2365.85284, 0         , 1970.24146],
    #           [0         , 2364.17864, 1497.37745],
    #           [0         , 0         , 1]])
    frame_rate = 30 # either  24 / 25 / 30p


    log.info('Processing video file: {}, annotations file: {}'.format(args.video, args.annotations))

    # Load ground-truth meta-data, if it exists
    annotations = None
    if args.annotations != None:
        annotations = parse_annotations(args.annotations)

    # Load the video stream and any ground-truth meta-data available.
    # Separate input into individual frames and process one at a
    # time

    frame_prev = None
    frame_cur = None

    vidcap = cv2.VideoCapture(args.video)
    success, frame_prev = vidcap.read()
    frame_num = 0
    skip = 60

    # Create output video for debugging
    # out = cv2.VideoWriter('{}_out.avi'.format(args.video),
    #         cv2.VideoWriter_fourcc('M','J','P','G'), 10,
    #         (frame_prev.shape[0],frame_prev.shape[1]))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, frame_rate / skip, (frame_prev.shape[1],frame_prev.shape[0]))

    while success:
        # Grab frame
        for i in range(skip):
            frame_num = frame_num + 1
            success,frame_cur = vidcap.read()
        if not success or frame_cur is None:
            break

        log.debug('FRAME: {}'.format(frame_num))

        # Grab all annotations for this frame number (column #5)
        bounding_boxes = None
        if not annotations is None:
            bounding_boxes = annotations[annotations[:, 5] == frame_num]

        process_frame(intr, 1.0 * skip / frame_rate, frame_cur, frame_prev, bounding_boxes, out)
        frame_prev = frame_cur

        # success = frame_num < 30
    vidcap.release()
    out.release()

    # DEBUG
    import matplotlib.pyplot as plt
    import pandas as pd

    ma_pos_x = moving_average(np.array(pos_x), 15)
    ma_pos_y = moving_average(np.array(pos_y), 15)
    ma_pos_z = moving_average(np.array(pos_z), 15)

    df=pd.DataFrame({'t': time, 'x': ma_pos_x, 'y': ma_pos_y, 'z': ma_pos_z })
    plt.plot('t', 'x', data=df, marker='')
    plt.plot('t', 'y', data=df, marker='')
    plt.plot('t', 'z', data=df, marker='')
    plt.legend()
    plt.show()

    pickle.dump([time, confidences, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z], open( "motion.p", "wb" ))

    log.info('Processing video complete, {} frames processed'.format(frame_num))



if __name__ == '__main__':
    main()
