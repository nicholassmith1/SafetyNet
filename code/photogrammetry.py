import cv2
import numpy as np
import matplotlib.pyplot as plt
import helpers

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

def camera_pose_to_frustrum_norms(K, R, C, bounding_box):
    """
    @param R            numpy array (3x3) representing the camera optical center rotation
    @param focal        focal length in pixels
    @param cc           principle point (Cx, Cy) in pixels
    @param bounding_box bounding boxes, in pixel, in the form (x0, y0, x1, y1)
    @return             numpy array (4x3) of vectors representing the normal vectors
                        of the planes that make up the frustrum of the specified bounding box
                        applied at the specified camera pose.
    """
    v0 = helpers.pixel_to_real_world(K, R, C, np.asarray((bounding_box[0], bounding_box[1])))
    v1 = helpers.pixel_to_real_world(K, R, C, np.asarray((bounding_box[2], bounding_box[1])))
    v2 = helpers.pixel_to_real_world(K, R, C, np.asarray((bounding_box[2], bounding_box[3])))
    v3 = helpers.pixel_to_real_world(K, R, C, np.asarray((bounding_box[0], bounding_box[3])))
    # v0 = np.asarray((bounding_box[0] - cc[0], bounding_box[1] - cc[1], focal))
    # v1 = np.asarray((bounding_box[2] - cc[0], bounding_box[1] - cc[1], focal))
    # v2 = np.asarray((bounding_box[2] - cc[0], bounding_box[3] - cc[1], focal))
    # v3 = np.asarray((bounding_box[0] - cc[0], bounding_box[3] - cc[1], focal))

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

    # print(v0 / v0[2] * 0.68)
    # print(v1 / v1[2] * 0.68)
    # print(v2 / v2[2] * 0.68)
    # print(v3 / v3[2] * 0.68)
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

    return n
    # return (n, np.asarray([v0, v1, v2, v3]))

def bounding_box_to_world_coord(K, R, C, bounding_box, point_cloud):
    norms = camera_pose_to_frustrum_norms(K, R, C, bounding_box)
    inner_pts = get_points_inside_frustrum(point_cloud, norms)
    # FIXME? For the moment, just calculate based on average of
    # surrounding groud

    # Return to global F.O.R.
    return np.mean(point_cloud[inner_pts] + C, axis=0)
    