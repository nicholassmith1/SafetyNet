import numpy as np
import math
import json
from plyfile import PlyData, PlyElement
import os
import pickle

def get_camera_intrinsic_from_sfm_data(sfm_json):
    """
    @param sfm_json     sfm_data.json file content
    @param pose_id      the camera position (extrinsic) data to retrieve
    @return P, R        numpy position and rotation matrices
    """
    return sfm_json['intrinsics'][0]['value']['ptr_wrapper']['data']


def get_frame_id_to_pose_id_map(sfm_json):
    """
    @param sfm_json     sfm_data.json file content
    @param frame_id    the video frame number to get a pose id for
    @return             a numpy Nx2 array of (frame_id, pose_id)
    """
    ret = []
    views = sfm_json['views']
    for view in range(len(views)):
        data = views[view]['value']['ptr_wrapper']['data']
        name = data['filename']
        name, ext = os.path.splitext(os.path.basename(name))
        name = name.split('_')
        try:
            name = name[len(name) - 1]
            num = int(name)
            ret.append([num, data['id_pose']])
        except:
            pass
    return np.asarray(ret)


# def get_pose_id_for_frame_id(filename, frame_id):
#     """
#     @param filename     sfm_data.json file
#     @param frame_id    the video frame number to get a pose id for
#     @return             frame number, of -1 if it doesn't exist
#     """
#     with open(filename, 'r') as file:
#         js = json.load(file)
#         # FIXME - wow. I'm super inefficient
#         views = js['views']
#         for view in range(len(views)):
#             data = js['views'][view]['value']['ptr_wrapper']['data']
#             name = data['filename']
#             name, ext = os.path.splitext(os.path.basename(name))
#             name = name.split('_')
#             try:
#                 name = name[len(name) - 1]
#                 num = int(name)
#                 if num == frame_id:
#                     return data['id_pose']
#             except:
#                 pass
#     return -1

def get_camera_pose_from_sfm_data(sfm_json, pose_id):
    """
    @param sfm_json     sfm_data.json file content
    @param pose_id      the camera position (extrinsic) data to retrieve
    @return P, R        numpy position and rotation matrices
    """
    poses = sfm_json['extrinsics']
    for p in poses:
        if p['key'] == pose_id:
            return (True, np.asarray(p['value']['center']), np.asarray(p['value']['rotation']))
    # Can fail, if view didn't achieve enough keypoints
    return False, np.zeros(3), np.zeros((3,3))

# http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf
# pg 155
def real_world_to_pixel(K, R, C, X):
    IC = np.c_[ np.identity(3), -C ]
    XX = np.append(X, 1)
    out = np.matmul(np.matmul(np.matmul(K, R), IC), XX.T)
    out = out / out[2]
    return out

# Test code - real world to pixel broken?
"""
>>> X = np.array([0.26217, -0.25947, 0.91148])
>>> C = np.array([0.1447, -0.39166, 0.24284])
>>> R = np.array([[0.99832343, 0.05357055, -0.024284906], [-0.05534797, 0.9943409, -0.09067949], [0.01693913, 0.09174074, 0.99563884]])
>>> K = np.array([[4699.8318, 0, 1885.765668], [0, 4699.83192, 1049.5404], [0, 0, 1]])
>>> x_est = helpers.real_world_to_pixel(K, R, C, X)
>>> x_est
array([2.63319077e+03, 1.49411204e+03, 1.00000000e+00])
>>> X_est = helpers.pixel_to_real_world(K, R, C, x_est[:2])
>>> X_est
array([-1.13542375, -1.27770209, -6.46283931])

"""

# # https://math.stackexchange.com/questions/2237994/back-projecting-pixel-to-3d-rays-in-world-coordinates-using-pseudoinverse-method?
# def pixel_to_real_world(K, R, C, x):
#     IC = np.c_[ np.identity(3), -C ]
#     xx = np.append(x, 1)
#     # np.matmul(np.linalg.inv(IC np.matmul(np.linalg.inv(R), np.matmul(np.linalg.inv(K), xx))
#     B = np.matmul(np.matmul(K, R), IC)
#     X, resid, rank, s = np.linalg.lstsq(B, xx.T)
#     # x,resid,rank,s = np.linalg.lstsq(B,b)
#     return X

# def pixel_to_real_world2(K, R, C, x):
#     xx = np.append(x, 1)
#     P = K.dot(np.c_[R, C])
#     X = np.dot(np.linalg.pinv(P),xx)
#     return X

# http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf
# p 161 "The projective camera"
# NOTE - points relative to camera, so absolute real world coordinates should be adjusted by C
def pixel_to_real_world(K, R, C, x):
    IC = np.c_[ np.identity(3), -C ]
    xx = np.append(x, 1)
    P = np.matmul(np.matmul(K, R), IC)
    Pt = np.matmul(P.T, np.linalg.inv(np.matmul(P, P.T)))
    X = np.matmul(Pt, xx)
    X = X / X[3]
    X = X[:3] - C
    return X


# X = np.dot(lin.pinv(P),p1)

def get_points_from_ply(filename):
    """
    Extract all vertex points from the specified PLY file
    @return             numpy array (Nx3) of detected depth points
    """
    with open(filename, 'rb') as file:
        plydata = PlyData.read(file)
        ve = plydata['vertex']
        pts = np.empty((ve.count, 3))
        pts[:, 0] = np.asarray(ve['x'])
        pts[:, 1] = np.asarray(ve['y'])
        pts[:, 2] = np.asarray(ve['z'])
        return pts
    assert False, 'Failure to import points'

# Testing code
# As a general note to myself, I'm considering all FOR as +x camera right,
# +y camera down, +z into frame.
#
# This generate s frustrum with a point at the origin, aimed into
# the image
"""
R = np.diag((1,1,1))
P = np.zeros(3)
bounding_box = np.asarray((3820/2 - 20, 2160/2 - 20, 3820/2 + 20, 2160/2 + 20)) 
frustrum_norms = helpers.camera_pose_to_frustrum_norms(P, R, 4699, np.asarray((3820/2, 2160/2)), bounding_box)
pts = np.zeros((6, 3))
pts[0] = np.asarray((0, 0, 0))      # inside
pts[1] = np.asarray((10000, 0, 0))  # outside
pts[2] = np.asarray((0, 0, -1))     # outside
pts[3] = np.asarray((0, 0, 1))      # inside
pts[4] = np.asarray((0, 0, -1))     # outside
pts[5] = np.asarray((-0.001, 0, 0)) # outside
helpers.get_points_inside_frustrum(pts, frustrum_norms)
"""

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

def get_bounding_boxes_for_frame(filename, frame_num):
    a = parse_annotations(filename)
    return a[a[:, 5] == frame_num]

# Real world test
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import helpers

P, R = helpers.get_camera_pose_from_sfm_data('test_sfm_data.json', 1)
intrinsics = helpers.get_camera_intrinsic_from_sfm_data('test_sfm_data.json')
bb = helpers.get_bounding_boxes_for_frame('../data/1.1.5.txt', 1080)

bounding_box = bb[6][1:5]
# why do the focal_length get divided by 2?
frustrum_norms = helpers.camera_pose_to_frustrum_norms(R, intrinsics['focal_length'] / 2, intrinsics['principal_point'], bounding_box)
pts = helpers.get_points_from_ply('scene_dense.ply')

in_frustrum = helpers.get_points_inside_frustrum(pts - P, frustrum_norms)
#np.where(in_pts == True)
p2 = pts[in_frustrum]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(p2[:,0], p2[:,1], p2[:,2]), plt.show()

"""

# Based on this derivation
# http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf
def least_square_line_intersect(A, B, c):
    """
    Finds the least-square-fit intersection of K lines in N-dimensional
    space with lines described as a point A and a vector from that
    point N to another point B.
    @param A        numpy KxN array
    @param B        numpy KxN array
    @param c        numpy 1xK cost
    @return         numpy 1xN array
    """
    K = A.shape[0]
    N = A.shape[1]
    # Rp = q
    # R = c * (np.eyes(K) - np.matmul(B, B.T)) # KxK
    # q = np.matmul(c * (np.eyes(K) - np.matmul(B, B.T)), A) # KxN

    R = np.zeros((N, N))
    q = np.zeros((N, 1))
    for i in range(K):
        R += c[i] * ( np.eye(N) - np.matmul( np.matrix(B[i]).T, np.matrix(B[i]) ) )
        # test = np.matmul((np.eye(N) - np.matmul(np.matrix(B[i]).T, np.matrix(B[i]))), np.matrix(A[i]).T)
        # print((np.eye(N) - np.matmul(B[i].T, B[i])).shape)
        # print(A.shape)
        # print(np.matrix(A[i]).shape)
        # print(np.matrix(A[i]).T.shape)
        # print(test.shape)
        # print(np.matmul((np.eye(N) - np.matmul(B[i].T, B[i])), A[i].T))
        # print(c[i])
        # print(c[i] * np.matmul((np.eye(N) - np.matmul(B[i].T, B[i])), A[i].T))
        # print(q)
        q += c[i] * np.matmul((np.eye(N) - np.matmul(np.matrix(B[i]).T, np.matrix(B[i]))), np.matrix(A[i]).T)

    # R = c * (np.eyes(K) - np.matmul(B, B.T)) # KxK
    # q = np.matmul(c * (np.eyes(K) - np.matmul(B, B.T)), A) # KxN

    # Moore pseudo-inverse
    Rt = np.matmul(R.T, np.linalg.inv(np.matmul(R, R.T)))
    # print(q)
    return np.matmul(Rt, q)


def estimate_epicenter(epicenter_prev, pedestrian_class, pedestrian_pos, pedestrian_vel_abs):
    pass


# From: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# from io import StringIO
# io = StringIO('["streaming API"]')
def deserial_safetynet_out(file):
    js = None
    with open(file, 'w') as f:
        js = json.load(f)

    scale = js['scale']
    timedelta = js['timedelta']
    K = js['K']
    frames = js['frames']

    frame_num = len(frames)

    drone_pose = np.zeros((frame_num, 3))
    drone_rot = np.zeros((frame_num, 3, 3))
    pedestrians_pose = []
    pedestrians_vel = []

    for i in range(frame_num):
        frame_id = frames[i]['frame_id']

        drone_pose[frame_id] = frames[i]['drone']['pose']
        drone_rot[frame_id] = frames[i]['drone']['R']
        # drone_rot[frame_id] = frames[i]['drone']['vel']

        # TODO -epicenter
        pedestrians = frames[i]['pedestrians']
        for p in pedestrians:
            pedestrians_pose.append([frame_id, p['id'], p['pose']])
            pedestrians_vel.append([frame_id, p['id'], p['vel']])







    # frames = []
    # for frame_id in range(frame_num):
    #     drone = {
    #         "pose" : drone_pose[frame_id],
    #         "R" : drone_rot[frame_id],
    #         "vel" : np.zeros(3) # TODO - placeholder
    #     }
    #     # Get pedestrian data
    #     pedestrians = []
    #     for pdata in pedestrians_pose[pedestrians_pose[:, 0] == frame_id]:
    #         p = {
    #             "id" : pdata[1],
    #             "pose" : pdata[2:5],
    #             # "vel" : pdata[5:8]
    #             "vel" : np.zeros(3)  # TODO - placeholder
    #         }
    #         pedestrians.append(p)
    #     # Assemble frame
    #     frame = {
    #         "frame_id" : frame_id,
    #         "drone" : drone,
    #         "epicenter" : np.zeros(3),
    #         "pedestrians": pedestrians
    #     }
    #     frames.append(frame)
 
    return (frame_num, drone_pose, drone_rot, pedestrians_pose, pedestrians_vel, K, frame_rate)


def deserial_combined_out(file):
    with open(file, 'r') as f:
        js = json.load(f)
    people = js['people']

    pedestrians_2d = []
    for person in people:
        id = person['id']
        for pos in person['pos']:
            p = [0] * 6
            p[0] = int(id)
            # p[1] = pos['frame_id']
            # p[1] = int(pos['frame_id'].split('_')[-1]) # hack, shouldn't have stored like this --legacy support
            p[1] = int(pos['frame_id']) 
            # p[2:6] = int(pos['box'])
            p[2] = int(pos['box'][0])
            p[3] = int(pos['box'][1])
            p[4] = int(pos['box'][2])
            p[5] = int(pos['box'][3])
            pedestrians_2d.append(p)
    return np.asarray(pedestrians_2d)


def pickle_safetynet_in(file):
    with open(file, 'rb') as f:
        frame_num = pickle.load(f)
        drone_pose = pickle.load(f)
        drone_rot = pickle.load(f)
        K = pickle.load(f)
        pedestrians_pose = pickle.load(f)
        pedestrians_vel = pickle.load(f)
        frame_rate = pickle.load(f)
        return (frame_num, drone_pose, drone_rot, K, pedestrians_pose, pedestrians_vel, frame_rate)


def pickle_safetynet_out(out_dir, frame_num, drone_pose, drone_rot, pedestrians_pose, pedestrians_vel, K, frame_rate):
    with open(os.path.join(out_dir, 'safetynet.pickle'), 'wb') as f:
        pickle.dump(frame_num, f)
        pickle.dump(drone_pose, f)
        pickle.dump(drone_rot, f)
        pickle.dump(K, f)
        pickle.dump(pedestrians_pose, f)
        pickle.dump(pedestrians_vel, f)
        pickle.dump(frame_rate, f)


def serial_safetynet_out(out_dir, frame_num, drone_pose, drone_rot, pedestrians_pose, pedestrians_vel, K, frame_rate):
    frames = []
    for frame_id in range(frame_num):
        drone = {
            "pose" : drone_pose[frame_id],
            "R" : drone_rot[frame_id],
            "vel" : np.zeros(3) # TODO - placeholder
        }
        # Get pedestrian data
        pedestrians = []
        for pdata in pedestrians_pose[pedestrians_pose[:, 0] == frame_id]:
            pv = pedestrians_vel[pedestrians_vel[:,1] == pdata[1]]
            pv = pv[pv[:,0] == frame_id]
            if pv.size == 0:
                np.asarray([1, 0, 0])
            else:
                pv = pv[0]
            p = {
                "id" : pdata[1],
                "pose" : pdata[2:5],
                "vel" : pv[2:5],
                # "vel" : pdata[5:8]
                # "vel" : np.asarray([1, 0, 0])  # TODO - placeholder
            }
            pedestrians.append(p)
        # Assemble frame
        frame = {
            "frame_id" : frame_id,
            "drone" : drone,
            "epicenter" : np.zeros(3),
            "pedestrians": pedestrians
        }
        frames.append(frame)
        
    data = {}
    data['scale'] = 1.0 # placeholder
    data['timedelta'] = 1 / frame_rate
    data['K'] = K
    data['frames'] = frames
    # data_dump = json.dumps(data, cls=NumpyEncoder, indent=4)
    # print(data_dump)
    with open(os.path.join(out_dir, 'safetynet.json'), 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)


# From https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def rotationMatrixToEulerAngles(R) :
    # assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

# From https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
 
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R