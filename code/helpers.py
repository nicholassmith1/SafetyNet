import numpy as np
import math
import json
from plyfile import PlyData, PlyElement
import os

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
            return (np.asarray(p['value']['center']), np.asarray(p['value']['rotation']))
    assert False, 'Unable to discover pose_id {}'.format(pose_id)

# http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf
# pg 155
def real_world_to_pixel(K, R, C, X):
    IC = np.c_[ np.identity(3), -C ]
    XX = np.append(X, 1)
    out = np.matmul(np.matmul(np.matmul(K, R), IC), XX.T)
    out = out / out[2]
    return out

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



def collate_pedestrians(pedestrians_prev, bounding_boxes):
    pass

def classify_pedestrian(pedestrian_vel_abs):
    pass

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
            p = {
                "id" : pdata[1],
                "pose" : pdata[2:5],
                # "vel" : pdata[5:8]
                "vel" : np.zeros(3)  # TODO - placeholder
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
    data_dump = json.dumps(data, cls=NumpyEncoder, indent=4)
    print(data_dump)
    with open(os.path.join(out_dir, 'safetynet.json'), 'w') as f:
        json.dump(data_dump, f)


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