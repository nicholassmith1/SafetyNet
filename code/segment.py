

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
import os

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

    basename, ext = os.path.splitext(args.video)
    count = 0
    out_dir = '../out'
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)


    while success:
        # Grab frame
        for i in range(skip):
            frame_num = frame_num + 1
            success,frame_cur = vidcap.read()
        if not success or frame_cur is None:
            break

        log.debug('FRAME: {}'.format(frame_num))

        # process_frame(intr, 1.0 * skip / frame_rate, frame_cur, frame_prev, bounding_boxes, out)
        frame_prev = frame_cur

        cv2.imwrite(os.path.join(out_dir, '{}_{}_{}.jpg'.format(basename, frame_num, count)), frame_cur)
        count = count + 1

        # success = frame_num < 30
    vidcap.release()
    # out.release()

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
