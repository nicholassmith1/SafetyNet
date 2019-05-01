import numpy as np
import math
import json
from plyfile import PlyData, PlyElement

def get_camera_intrinsic_from_sfm_data(filename):
    """
    @param filename     sfm_data.json file
    @param pose_id      the camera position (extrinsic) data to retrieve
    @return P, R        numpy position and rotation matrices
    """
    with open(filename, 'r') as file:
        js = json.load(file)
        return js['intrinsics'][0]['value']['ptr_wrapper']['data']
    assert False, 'Unable to discover intrinsic'


def get_camera_pose_from_sfm_data(filename, pose_id):
    """
    @param filename     sfm_data.json file
    @param pose_id      the camera position (extrinsic) data to retrieve
    @return P, R        numpy position and rotation matrices
    """
    with open(filename, 'r') as file:
        js = json.load(file)
        poses = js['extrinsics']
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

def camera_pose_to_frustrum_norms(R, focal, cc, bounding_box):
    """
    @param R            numpy array (3x3) representing the camera optical center rotation
    @param focal        focal length in pixels
    @param cc           principle point (Cx, Cy) in pixels
    @param bounding_box bounding boxes, in pixel, in the form (x0, y0, x1, y1)
    @return             numpy array (4x3) of vectors representing the normal vectors
                        of the planes that make up the frustrum of the specified bounding box
                        applied at the specified camera pose.
    """
    v0 = np.asarray((bounding_box[0] - cc[0], bounding_box[1] - cc[1], focal))
    v1 = np.asarray((bounding_box[2] - cc[0], bounding_box[1] - cc[1], focal))
    v2 = np.asarray((bounding_box[2] - cc[0], bounding_box[3] - cc[1], focal))
    v3 = np.asarray((bounding_box[0] - cc[0], bounding_box[3] - cc[1], focal))

    # TODO - account for radial distortion?

    # explanation of coordinates https://github.com/openMVG/openMVG/issues/788
    # apply rotations
    # v0 = np.matmul(R, v0)
    # v1 = np.matmul(R, v1)
    # v2 = np.matmul(R, v2)
    # v3 = np.matmul(R, v3)
    # v0 = np.dot(R, v0)
    # v1 = np.dot(R, v1)
    # v2 = np.dot(R, v2)
    # v3 = np.dot(R, v3)
    # # apply translations
    # v0 = v0 + P
    # v1 = v1 + P
    # v2 = v2 + P
    # v3 = v3 + P

    print(v0 / v0[2] * 0.68)
    print(v1 / v1[2] * 0.68)
    print(v2 / v2[2] * 0.68)
    print(v3 / v3[2] * 0.68)
    # print(v0 / v0[2] * 0.668)
    # print(v1 / v1[2] * 0.668)
    # print(v2 / v2[2] * 0.668)
    # print(v3 / v3[2] * 0.668)

    # why would i need to invert the rotation of the pose?
    # Ri = np.linalg.inv(R)
    # v0 = np.matmul(Ri, v0)
    # v1 = np.matmul(Ri, v1)
    # v2 = np.matmul(Ri, v2)
    # v3 = np.matmul(Ri, v3)

    # watch the winding consistency (normal vectors should point inwards)
    n = np.empty((4, 3))
    n[0,:] = np.cross(v0, v1)
    n[1,:] = np.cross(v1, v2)
    n[2,:] = np.cross(v2, v3)
    n[3,:] = np.cross(v3, v0)

    # return n
    return (n, np.asarray([v0, v1, v2, v3]))

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

def get_points_inside_frustrum(points, frustrum_norms):
    """
    @param points       numpy array (Nx3) of points in the scene
    @param frustrum_nroms   numpy array (4x3) representing the planar normal vectors
                        of the frustrum
    @return             numpy array (Nx1) of bool, True if the point is within the
                        frustrum, False otherwise
    """
    # Dot each point onto each frustrum norm to get it's projection.
    # Results in an Nx4 matrix
    dots = np.dot(points, frustrum_norms.T)
    # For a point to be within the frustrum, the projection on each normal
    # vector must be positive
    return (dots >= 0).all(axis=1)



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