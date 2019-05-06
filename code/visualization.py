import matplotlib.pyplot as plt
from skimage import io
from scipy import misc
import helpers
import cv2
import numpy as np

def moving_average(data, N):
    return np.convolve(data, np.ones((N,))/N, mode='same')

def visualize(video, combined, safetynet):
    bounding_boxes = helpers.deserial_combined_out(combined)
    # args.pos
    frame_num, drone_pose, drone_rot, K, pedestrians_pose, pedestrians_vel, frame_rate = helpers.pickle_safetynet_in(safetynet)

    vidcap = cv2.VideoCapture(video)
    success, frame = vidcap.read()

    out = cv2.VideoWriter('{}_out.avi'.format(video),
    cv2.VideoWriter_fourcc('M','J','P','G'), 10,
            (frame.shape[1],frame.shape[0]))

    # Pre-filter velocity
    pp = pedestrians_vel
    pp = pp[np.invert(np.any(np.isnan(pp), axis=1))]
    # for pid in np.unique(pp[:, 1])
    idx = np.lexsort((pp[:,0], pp[:,1])) # lexsort is weird (reversed params) sorty by p_id, then by frame_id
    pp = pp[idx]
    last_pid = -1
    last_idx = 0
    i = 0
    # Traverse a sorted (p_id prim, frame_id sec) list of velocities.
    # Everytime  the p_id changes, run a filter on all elements since
    # the last p_id change
    while i < len(pp):
        if last_pid != pp[i, 1]:
            if last_pid != -1:
                N = min(10, i - last_idx)
                # print(pp[last_idx:i, 2])
                # print( moving_average(pp[last_idx:i, 2], N))
                pp[last_idx:i, 2] = moving_average(pp[last_idx:i, 2], N)
                pp[last_idx:i, 3] = moving_average(pp[last_idx:i, 3], N)
                pp[last_idx:i, 4] = moving_average(pp[last_idx:i, 4], N)
                # Zero the last 10 to compensate for bounding box interpolation issues
                zz = max(last_idx, i - 10)
                pp[zz:i, 2:6] = 0
            last_pid = pp[i, 1]
            last_idx = i
        i = i + 1
    pedestrians_vel = pp

    frame_num = 0
    while success:
        frame_num = frame_num + 1

        # print(bounding_boxes)
        # print(bounding_boxes[: 1] == frame_num)
        fbb = bounding_boxes[bounding_boxes[:, 1] == frame_num]
        ppos = pedestrians_pose[pedestrians_pose[:, 0] == frame_num]
        pvel = pedestrians_vel[pedestrians_vel[:, 0] == frame_num]
        # TODO - make scale and framerate part of json
        out_frame = process_frame(frame, frame_num, fbb, K, drone_rot[frame_num], drone_pose[frame_num], ppos, pvel, 7.198792, 10)
        out.write(out_frame)

        # Grab next frame
        success,frame = vidcap.read()

    vidcap.release()
    out.release()



def process_frame(img, frame_id, bounding_boxes, K, R, C, ppos, pvel, scale, fps):
    # Draw bounding boxes
    for i in range(len(bounding_boxes)):
        p1 = bounding_boxes[i, 2:4]
        p2 = bounding_boxes[i, 4:6]
        # print(p1)
        # print(p2)
        cv2.rectangle(img, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), 2)

    # print(pvel)

    # Draw position and velocity
    for i in range(len(ppos)):
        if np.isnan(ppos[i]).any():
            continue

        # print(ppos[i])
        # print(K)
        # print(R)
        # print(C)
        p1 = helpers.real_world_to_pixel(K, R, C, ppos[i, 2:6])

        # print(ppos[i])
        # print('....')
        # print(p1)

        # p2 = helpers.real_world_to_pixel(K, R, C, ppos[i, 2:6] + np.array([3, 0, 0]))
        # print('{}'.format(np.linalg.norm(pvel[:, 2:6])))

        # Find the corresponding instantenous velocity
        v2 = pvel[pvel[:,1] == ppos[i, 1]]
        if v2.size == 0:
            v2 = np.zeros(5)
        else:
            v2 = v2[0]

        # Point 2 is projected head by ptime seconds
        ptime = 0.25
        p2 = helpers.real_world_to_pixel(K, R, C, ppos[i, 2:6] + (fps * ptime) * v2[2:6])

        if np.isnan(p1).any() or np.isnan(p1).any():
            print('Something strange happened')
            print(ppos[i])
            print(C)
            print(p1), print(p2)
            continue

        cv2.arrowedLine(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 4)

        # Draw the instaneous magnitude
        # text = '{}'.format(ppos[i])
        text = '({:0.3f}, {:0.3f}, {:0.3f})'.format(ppos[i, 2] * scale, ppos[i, 3] * scale, ppos[i, 4] * scale)
        cv2.putText(img, text, (int(p1[0]) + 10, int(p1[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 3, cv2.LINE_AA)
        text = '{:0.3f} m/s'.format(np.linalg.norm(v2[2:6]) * scale)
        cv2.putText(img, text, (int(p1[0]) + 10, int(p1[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 3, cv2.LINE_AA)


    # Print the drone position
    # text = 'id={} ({:0.3f}, {:0.3f}, {:0.3f})'.format(frame_id, scale * C[0],
    #         scale * C[1], scale * C[2])
    text = 'id={} ({:0.3f}, {:0.3f}, {:0.3f})'.format(frame_id, C[0], C[1], C[2])
    # text = ''
    cv2.putText(img, text, (30, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
            3.0, (255, 0, 0), 4, cv2.LINE_AA)

    return img
