import numpy as np
import argparse
import constants
import logging
import signal
import sys
import cv2
import matplotlib.pyplot as plt
from helpers import (rotationMatrixToEulerAngles)
import os
import tempfile
import json

from segmentation import find_pedestrians
# from helpers import (get_points_from_ply,

#         collate_pedestrians,
#         classify_pedestrian,
#         estimate_epicenter)
import helpers
from openMVG_pipeline import (openMVG_pipe, openMVS_pipe)
from photogrammetry import (bounding_box_to_world_coord)
 
log = logging.getLogger('SafetyNet')

def signal_handler(sig, frame):
        log.info('Exitting')
        sys.exit(0)


def process_video_frames(vidcap, out_dir, basename, frame_skip=60):
    """
    @return     total frames processed
    """
    rame_prev = None
    frame_cur = None
    
    success, frame_cur = vidcap.read()
    frame_num = 0

    while success:
        log.debug('FRAME: {}'.format(frame_num, os.path.join(out_dir)))
        cv2.imwrite(os.path.join(out_dir, '{}_{:07d}.png'.format(basename, frame_num)), frame_cur)
        frame_prev = frame_cur

        # Grab frame
        for i in range(frame_skip):
            frame_num = frame_num + 1
            success,frame_cur = vidcap.read()
        if not success or frame_cur is None:
            break
        # cut short
        if frame_num > 60:
            break

    return frame_num


def unravel_video(video_filename, out_dir, frame_skip=60, skip=False):
    basename, ext = os.path.splitext(os.path.basename(video_filename))
    vidcap = cv2.VideoCapture(video_filename)

    img_dir = os.path.join(out_dir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir, exist_ok=True)

    if skip:
        frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        frame_num = process_video_frames(vidcap, img_dir, basename, frame_skip=frame_skip)

    vidcap.release()

    return frame_num, img_dir


def pose_estimation(img_dir, out_dir, skip=False):
    # generate workspace
    pose_dir = os.path.join(out_dir, 'pose')
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir, exist_ok=True)

    # Establish output
    sfm_data_bin = os.path.join(pose_dir, "sfm_data.bin")
    sfm_data_json = os.path.join(pose_dir, "sfm_data.json")

    if not skip:
        sfm_data = openMVG_pipe(img_dir, pose_dir,
                sfm_data_bin, sfm_data_json)

    return sfm_data_bin, sfm_data_json


def depth_estimation(sfm_data_filename, out_dir, skip=False):
    # Generate workspace
    depth_dir = os.path.join(out_dir, 'depth')
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir, exist_ok=True)

    # Establish output
    pc_filename = os.path.join(depth_dir, "scene_dense.ply")
    if not skip:
        # Run the openMVS pipeline
        pc_filename = openMVS_pipe(depth_dir, sfm_data_filename, pc_filename)

    return pc_filename


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='SafetyNet')
    parser.add_argument('video', metavar='VIDEOFILE', type=str,
                    help='The video to process')
    parser.add_argument('-a', dest='annotations', type=str, default=None,
                    required=False, help='Overrides the neural network calculation of pedestrian \
                    bounding boxes and uses annotations for bounding-box generation')
    parser.add_argument('--depth_pose_downsample', type=int, default=60,
            help='Only use 1 in N views for density estimation, which tends to be computationally intense')
    parser.add_argument('--skip_unravel', action='store_true', help='Skip the generation of frames from a video')
    parser.add_argument('--skip_pose', action='store_true', help='Skip the openMVG pipeline')
    parser.add_argument('--skip_depth', action='store_true', help='Skip the openMVS pipeline')

    args = parser.parse_args()

    # Install signal handlers for early termination
    signal.signal(signal.SIGINT, signal_handler)

    # Setup the logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.DEBUG)

    # # DJI Phantom-4 standard intrinsic matrix, per
    # # https://stackoverflow.com/questions/47896876/camera-intrinsic-matrix-for-dji-phantom-4
    # #
    # # Video input is (2160, 3840, 3). This looks like a phantom 4 (regular)
    # # running in 4K mode, per https://www.dji.com/phantom-4/info#specs  
    # # https://gist.github.com/dbaldwin/21d55b134d71fd66e06dce8b91651a03
    # intr = np.array([[2246.742, 0.0     , 1920.0],
    #                  [0.0     , 2246.742, 1080.0],
    #                  [0.0     , 0.0     , 1.0   ]])
    frame_rate = 29 # FPS


    log.info('Processing video file: {}, annotations file: {}'.format(args.video, args.annotations))

    # Load ground-truth meta-data, if it exists
    annotations = None
    if args.annotations != None:
        annotations = helpers.parse_annotations(args.annotations)
        # TODO get data either from annotation, or elsewhere
        # pedestrians_2d = np.array((1, 6)) # [p_id, frame_id, xmin, ymin, xmax, ymax]
        pedestrians_2d = np.zeros((len(annotations), 6))
        pedestrians_2d[:, 0]  = annotations[:,0]
        pedestrians_2d[:, 1]  = annotations[:,5]
        pedestrians_2d[:, 2:6]  = annotations[:, 1:5]
        # all_frame_ids = np.sort(np.unique(pedestrian_2d[:, 1]))

    # Create a space to work in the temporary filesystem
    basename, ext = os.path.splitext(os.path.basename(args.video))
    out_dir = tempfile.gettempdir()
    out_dir = os.path.join(out_dir, 'SafetyNet', basename)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Turn the video into a sequence of images
    frame_num, img_dir = unravel_video(args.video, out_dir, frame_skip=1, skip=args.skip_unravel)

    # Run the openMVG pipeline.
    sfm_bin_filename, sfm_data_filename = pose_estimation(img_dir, out_dir, skip=args.skip_pose)

    # Run the openMVS pipeline
    pc_filename = depth_estimation(sfm_bin_filename, out_dir, skip=args.skip_depth)

    # Get the point cloud
    point_cloud = helpers.get_points_from_ply(pc_filename)

    # bb = helpers.get_bounding_boxes_for_frame('../data/1.1.5.txt', 1080)

    sfm_json = None
    with open(sfm_data_filename, 'r') as file:
        sfm_json = json.load(file) 

    # Get calculated intrinsics
    intrinsics = helpers.get_camera_intrinsic_from_sfm_data(sfm_json)
    K = np.zeros((3,3))
    K[0, 0] = intrinsics['focal_length']
    K[1, 1] = intrinsics['focal_length']
    K[0, 2] = intrinsics['principal_point'][0]
    K[1, 2] = intrinsics['principal_point'][1]
    K[2, 2] = 1

    # Make sure everything works. Annotated data sometimes seems to
    # extend beyond the end of the video
    frame_num = max(frame_num, int(np.max(pedestrians_2d[:,1])) + 1) 

    # drone_data_avail, drone_pose, and drone_rot are all indexed by frame number
    drone_data_avail = np.zeros(frame_num)
    drone_pose = np.zeros((frame_num, 3))
    drone_rot = np.zeros((frame_num, 3, 3))

    fid_to_pid_map = helpers.get_frame_id_to_pose_id_map(sfm_json)
    for frame_id, pose_id in fid_to_pid_map:
        drone_data_avail[frame_id] = 1
        drone_pose[frame_id], drone_rot[frame_id] = helpers.get_camera_pose_from_sfm_data(sfm_json, pose_id)


    # Extrapolate drone position.
    # Processing every single frame is prohibitively time consuming using
    # openMVS and openMVG. We linearly extrapolate postion and orientation
    # between these frames by traversing the lists, ensuring with two valid
    # poses, then scaling the deltas between them.
    prev_idx = -1
    prev_pose = None
    prev_rot_eul = None
    cur_idx = -1
    cur_pos = None
    cur_rot_eul = None
    for i, avail in enumerate(drone_data_avail):
        if avail:
            prev_idx = cur_idx
            prev_pose = cur_pose
            prev_rot_eul = cur_rot_eul
            cur_idx = i
            cur_pose = drone_pose[i]
            cur_rot_eul = helpers.rotationMatrixToEulerAngles(drone_rot[i])
            # Check both sides
            if prev_idx >= 0 and cur_idx >= 0:
                j = i - 1
                n = cur_idx - prev_idx
                dt_pose = cur_pose - prev_pose
                dt_rot_eul = cur_rot_eul - prev_rot_eul
                while not drone_data_avail[j] and j > 0:
                    drone_pose[j] = (((j - prev_idx) / n) * dt_pose) + prev_pose
                    drone_rot[j] = helpers.eulerAnglesToRotationMatrix((((j - prev_idx) / n) * dt_rot_eul) + prev_rot_eul)
                    drone_data_avail[j] = 1
                    j = j - 1

    print('drone pose estimation complete')

    # Populate pedestrian poses
    pedestrians_pose = np.zeros((len(pedestrians_2d), 5)) # [frame_id, p_id, x, y, z]
    for i, p2d in enumerate(pedestrians_2d):
        frame_id = int(p2d[1])
        pedestrians_pose[i, 0] = p2d[1]  # copy frame id
        pedestrians_pose[i, 1] = p2d[0]  # copy pedestrian id

        # C, R = helpers.get_camera_pose_from_sfm_data('test_sfm_data.json', 1)
        # print(frame_id)
        if not drone_data_avail[frame_id]:
            continue
        
        C = drone_pose[frame_id]
        R = drone_rot[frame_id]
        try:
            pedestrians_pose[i, 2:] = bounding_box_to_world_coord(K, R, C, p2d[2:6], point_cloud)
            # print(pedestrians_pose[i, 2:])
        except:
            print('failure')
            print(R)
            print(C)
            print(p2d[2:6])

    print('pedestrian pose estimation complete')

    # Calculate the velocity by staggering the sequential positional
    # data, and subtracting the two. We ignore jumps of more than
    # MAX_FRAME_DELTA frames, which we detect by looking at the difference
    # in associated frame_ids.
    #
    # NOTE - this assumes densly populated pedestrian poses
    # pedestrians_vel = np.zeros((len(pedestrians_2d), 4)) # [frame_id, p_id, vx, vy, vz]
    MAX_FRAME_DELTA = 29
    pp = pedestrians_pose.copy()
    idx = np.lexsort((pp[:,1], pp[:,0])) # sorty by p_id, then by frame_id
    pp = pp[idx]
    pp_next = np.zeros((len(pp) + 1, 5))
    pp_next[0:] = pp[1:]
    pp_next[len(pp_next) - 1, 0] = np.inf # ensure will not be used
    pp_dt_pid = pp_next[:, 1] - pp[:, 1]
    pp_dt_fid = pp_next[:, 0] - pp[:, 0]
    pp_dt_pos = pp_next[: 2:] - pp[:, 2:]
    # pp_dt_pid == 0 => same pedestrian
    # pp_dt_fid > MAX_FRAME_DELTA => timely data
    pp[:, 2:] = pp_dt_pos / (pp_dt_fid / frame_rate) # v = dx / dt
    pp[pp[pp_dt_pid != 0], 2:] = 0
    pp[pp[pp_dt_fid< 0], 2:] = 0
    pp[pp[pp_dt_fid > MAX_FRAME_DELTA], 2:] = 0
    pedestrians_vel = pp

    # Calculate epicenters
    # Our current methedology is a weighted least-square fit estimation
    # of velocity vector-line intercetion in ||R^3||
    # http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf


    # Serialize the output
    helpers.serial_safetynet_out(out_dir, frame_num,
            drone_pose, drone_rot,
            pedestrians_pose, pedestrians_vel,
            K, frame_rate)


if __name__ == '__main__':
    main()
