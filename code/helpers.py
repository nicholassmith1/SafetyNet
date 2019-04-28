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

def camera_pose_to_frustrum_norms(P, R, focal, cc, bounding_box):
    """
    @param P            numpy array (1x3) representing the camera pose optical center
    @param R            numpy array (3x3) representing the camera optical center rotation
    @param focal        focal length in pixels
    @param cc           principle point (Cx, Cy) in pixels
    @return             numpy array (4x3) of vectors representing the normal vectors
                        of the planes that make up the frustrum of the specified bounding box
                        applied at the specified camera pose.
    """
    v0 = np.asarray((bounding_box[0]                   - cc[0], bounding_box[1]                   - cc[1], focal))
    v1 = np.asarray((bounding_box[0] + bounding_box[2] - cc[0], bounding_box[1]                   - cc[1], focal))
    v2 = np.asarray((bounding_box[0] + bounding_box[2] - cc[0], bounding_box[1] + bounding_box[3] - cc[1], focal))
    v3 = np.asarray((bounding_box[0]                   - cc[0], bounding_box[1] + bounding_box[3] - cc[1], focal))

    # apply rotations
    v0 = np.matmul(R, v0)
    v1 = np.matmul(R, v1)
    v2 = np.matmul(R, v2)
    v3 = np.matmul(R, v3)
    # apply translations
    v0 = v0 + P
    v1 = v1 + P
    v2 = v2 + P
    v3 = v3 + P

    # watch the winding consistency (normal vectors should point inwards)
    n = np.empty((4, 3))
    n[0,:] = np.cross(v0, v1)
    n[1,:] = np.cross(v1, v2)
    n[2,:] = np.cross(v2, v3)
    n[3,:] = np.cross(v3, v0)

    return n

def get_points_from_ply(filename):
    """
    Extract all vertex points from the specified PLY file
    @return             numpy array (Nx3) of detected depth points
    """
    with open(filename 'rb') as file:
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
#P, R = helpers.get_camera_pose_from_sfm_data('test_sfm_data.json', 0)
R = np.diag((1,1,1))
P = np.zeros(3)
intrinsics = helpers.get_camera_intrinsic_from_sfm_data('test_sfm_data.json')
bounding_box = np.asarray((3820, 2160, 40, 40)) 
#frustrum_norms = helpers.camera_pose_to_frustrum_norms(P, R, intrinsics['focal_length'], intrinsics['principal_point'], bounding_box)
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