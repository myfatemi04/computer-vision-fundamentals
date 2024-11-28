# A simple test with the Essential Matrix.
# Given a pair of images and 8 correspondences between them, I should be able to compute the fundamental (and therefore, essential) matrix.
# Then, I should be able to perform triangulation.

# It seems like the continuity camera captures at 1920x1080, and regular video capture occurs at 3840x2160.
# Images are captured at 4032x3024.
# Do I just create a center crop in that case?
# Likely the sensor size is 4032x3024, and they use video sizes of 3840x2160 because it's easier to resize to 1920x1080...
# It would not make sense for them to have a scaling step.

import json
import os

import cv2
import numpy as np

from .load_image import image_to_1080p

base = os.path.abspath(os.path.dirname(__file__))

left = image_to_1080p(cv2.imread(base + "/IMG_9728.JPG"))
right = image_to_1080p(cv2.imread(base + "/IMG_9729.JPG"))

correspondences = np.loadtxt(base + "/selections.txt", skiprows=1, delimiter=",")
left_correspondences = correspondences[:8, 2:]
right_correspondences = correspondences[8:, 2:]

# Now, we compute correspondences in canonical camera frame.
with open("modules/calibs/iphone12promax_continuity_camera.json") as f:
    calibration_1080p = json.load(f)

matrix = np.array(calibration_1080p["matrix"])
distCoeffs = np.array(calibration_1080p["distortion"])

left_correspondences = cv2.undistortImagePoints(
    left_correspondences[:, None, :], matrix, distCoeffs
)[:, 0, :]
right_correspondences = cv2.undistortImagePoints(
    right_correspondences[:, None, :], matrix, distCoeffs
)[:, 0, :]

# Invert the intrinsic matrix.
left_correspondences_canonical = (
    left_correspondences - np.array([matrix[0, 2], matrix[1, 2]])
) / np.array([matrix[0, 0], matrix[1, 1]])
right_correspondences_canonical = (
    right_correspondences - np.array([matrix[0, 2], matrix[1, 2]])
) / np.array([matrix[0, 0], matrix[1, 1]])

# Create object points.
left_correspondences_canonical_obj = np.ones(
    (left_correspondences_canonical.shape[0], 3)
)
left_correspondences_canonical_obj[:, :2] = left_correspondences_canonical

# Measure reprojection error.
identity_tvec = np.zeros((3, 1))
identity_rvec = np.zeros((3, 1))

# The identity rvec is a 3x1 zero vector.
# identity_rvec, jacobian = cv2.Rodrigues(np.eye(3))
# print(identity_rvec)

projected, jacobian = cv2.projectPoints(
    left_correspondences_canonical_obj,
    identity_rvec,
    identity_tvec,
    matrix,
    distCoeffs,
)
projected = projected[:, 0, :]

print("Reprojection error:", np.linalg.norm(projected - left_correspondences), "pixels")
