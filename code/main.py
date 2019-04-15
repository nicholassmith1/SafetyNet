

import numpy as np
import argparse
import constants
import logging
import signal
import sys
import cv2

from photogrammetry import (calc_drone_vel,
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
 

def process_frame(intr, dt, frame_cur, frame_prev, bounding_boxes=None):
    pedestrians_prev = None  # FIXME
    epicenter_prev = None

    # Get an array of pedestrian bounding boxes.
    if bounding_boxes is None:
        bounding_boxes = find_pedestrians(frame_cur)

    # Calculate the drone instanenous velocity, drone FOR
    drone_vel = calc_drone_vel(intr, frame_prev, frame_cur, bounding_boxes)

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
    intr = np.array([[1865.0, 0.0   , 2000.0],
                     [0.0   , 1865.0, 1500.0],
                     [0.0   , 0.0   , 1.0   ]])


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
    while success:
        frame_num = frame_num + 1
        log.debug('FRAME: {}'.format(frame_num))

        # Grab frame
        success,frame_cur = vidcap.read()
        if not success:
            break

        # Grab all annotations for this frame number (column #5)
        bounding_boxes = None
        if not annotations is None:
            bounding_boxes = annotations[annotations[:, 5] == frame_num]

        process_frame(intr, 1.0 / constants.FRAME_RATE, frame_cur, frame_prev, bounding_boxes)

    log.info('Processing video complete, {} frames processed'.format(frame_num))



if __name__ == '__main__':
    main()
