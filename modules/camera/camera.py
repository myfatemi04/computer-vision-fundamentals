import cv2
import numpy as np


class Camera:
    def __init__(
        self, intrinsic_matrix: np.ndarray, distortion_coefficients: np.ndarray
    ):
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients

    @property
    def fx(self):
        return self.intrinsic_matrix[0, 0]

    @property
    def fy(self):
        return self.intrinsic_matrix[1, 1]

    @property
    def cx(self):
        return self.intrinsic_matrix[0, 2]

    @property
    def cy(self):
        return self.intrinsic_matrix[1, 2]

    @classmethod
    def from_calibration_file(cls, path: str):
        import json

        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            np.array(data["matrix"]),
            np.array(data["distortion"]),
        )

    def rectify_points(self, points: np.ndarray):
        return cv2.undistortImagePoints(
            points[:, None, :], self.intrinsic_matrix, self.distortion_coefficients
        )[:, 0, :]

    def rectified_to_homogeneous(self, points: np.ndarray):
        points = (points - np.array([self.cx, self.cy])) / np.array([self.fx, self.fy])

        result = np.ones((points.shape[0], 3))
        result[:, :2] = points

        return result

    def unrectified_to_homogeneous(self, points: np.ndarray):
        return self.rectified_to_homogeneous(self.rectify_points(points))

    def project_points(self, points: np.ndarray):
        identity_tvec = np.zeros((3, 1))
        identity_rvec = np.zeros((3, 1))

        # The identity rvec is a 3x1 zero vector.
        # identity_rvec, jacobian = cv2.Rodrigues(np.eye(3))
        # print(identity_rvec)

        projected, jacobian = cv2.projectPoints(
            points,
            identity_rvec,
            identity_tvec,
            self.intrinsic_matrix,
            self.distortion_coefficients,
        )
        projected = projected[:, 0, :]

        return projected
