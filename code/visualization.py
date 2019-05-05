import matplotlib.pyplot as plt
from skimage import io
from scipy import misc
import helpers
import cv2


def parse_safetynet(js):
    pass

def visualize(video, combined, safetynet):
    bounding_boxes = helpers.deserial_combined_out(combined)
    # args.pos
    frame_num, drone_pose, drone_rot, K, pedestrians_pose, pedestrians_vel, frame_rate = helper.sdeserial_safetynet_out(safetynet)

    vidcap = cv2.VideoCapture(args.video)
    success, frame = vidcap.read()

    out = cv2.VideoWriter('{}_out.avi'.format(video),
    cv2.VideoWriter_fourcc('M','J','P','G'), 10,
            (frame.shape[0],frame.shape[1]))


    frame_num = 0
    while success:
        frame_num = frame_num + 1

        fbb = bounding_boxes[bounding_boxes[: 1] == frame_num]
        ppos = pedestrians_pose[pedestrians_pose[:, 0] == frame_num]
        pvel = pedestrians_vel[pedestrians_vel[:, 0] == frame_num]
        out_frame = process_frame(frame, K, R, C, ppos, pvel, 1.0, 29)
        out.write(out_frame)

        cv2.wait()

        # Grab next frame
        success,frame_cur = vidcap.read()

    vidcap.release()
    out.release()



def process_frame(img, bounding_boxes, K, R, C, ppos, pvel, scale, fps):
    # Draw bounding boxes
    for i in range(len(bounding_boxes)):
        p1 = bounding_boxes[i, 2:4]
        p2 = bounding_boxes[i, 4:6]
        cv2.rectangle(img, p1, p2, (255, 0, 0))

    # Draw position and velocity
    for i in range(len(ppos)):
        p1 = helpers.real_world_to_pixel(K, R, C, ppos[i])
        p2 = helpers.real_world_to_pixel(K, R, C, ppos[i] + np.array([0.5, 0, 0]))
        # p2 = helpers.real_world_to_pixel(K, R, C, ppos[i] + pvel[i])
        cv2.line(img, p1, p2, (255, 0, 0))

        # Draw the instaneous magnitude
        text = '{:0.3f} m/s'.format(np.linalg.norm(pvel) * scale)
        cv2.putText(img, text, p1, cv2.FONT_HERSHEY_SIMPLEX,
                3.0, (255, 0, 0), 4, cv2.LINE_AA)

    # Print the drone position
    text = '({:0.3f}, {:0.3f}, {:0.3f})'.format(scale * C[0],
            scale * C[1], scale * C[2])
    cv2.putText(img, text, (30, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
            3.0, (255, 0, 0), 4, cv2.LINE_AA)

    return img
