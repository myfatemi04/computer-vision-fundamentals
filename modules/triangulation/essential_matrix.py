# A simple test with the Essential Matrix.
# Given a pair of images and 8 correspondences between them, I should be able to compute the fundamental (and therefore, essential) matrix.
# Then, I should be able to perform triangulation.

# It seems like the continuity camera captures at 1920x1080, and regular video capture occurs at 3840x2160.
# Images are captured at 4032x3024.
# Do I just create a center crop in that case?
# Likely the sensor size is 4032x3024, and they use video sizes of 3840x2160 because it's easier to resize to 1920x1080...
# It would not make sense for them to have a scaling step.

import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

from ..camera import Camera
from .load_image import image_to_1080p


# Create an intrinsic matrix...
def get_normalized_image_points(points: np.ndarray):
    K = np.eye(3)
    K[0, 0], K[1, 1] = points.std(axis=0)
    K[0, 2], K[1, 2] = points.mean(axis=0)
    K = np.linalg.inv(K)

    points_homogeneous = np.ones((points.shape[0], 3))
    points_homogeneous[:, :2] = points

    points_norm = points_homogeneous @ K.T

    return points_norm[:, :2], K


# Let's make a function that generates an epipolar line.
# Note: If we want to create an epipolar line for a point on the right camera's image plane
# instead, we can simply transpose the fundamental matrix.
def get_epipolar_line(left_point: np.ndarray, F: np.ndarray):
    """
    We have left_point.T @ F @ (right_point) = 0.
    We constrain right_point[2] = 1 (so it must lie on the camera plane).
    We can return a line in standard form.

    [u u', u v', u, v u', v v', v, u', v', 1] @ [F00, F01, ..., F21, F22].T = 0

    [u F00 + v F10 + F20, u F01 + v F11 + F21, u F02 + v F12 + F22] @ [u', v', 1] = 0
    """

    u = left_point[0]
    v = left_point[1]

    # ax + by = c
    a = F[0, 0] * u + F[1, 0] * v + F[2, 0]
    b = F[0, 1] * u + F[1, 1] * v + F[2, 1]
    c = -(F[0, 2] * u + F[1, 2] * v + F[2, 2])

    return a, b, c


def compute_fundamental_matrix(
    camera: Camera, left_correspondences, right_correspondences
):
    # Now, we can compute the essential matrix. We use the 8-point algorithm, which solves a least-squares problem.
    # min |Ax| subject to |x| = 1, where x is a vector representing the entries of the Essential Matrix.
    left_correspondences = camera.rectify_points(left_correspondences)
    right_correspondences = camera.rectify_points(right_correspondences)

    # Normalize correspondences for numerical stability.
    # These transformations are linear, so we will fuse them into the Fundamental matrix created
    # from the normalized points.
    left_corr_norm, left_normalizing_intrins = get_normalized_image_points(
        left_correspondences
    )
    right_corr_norm, right_normalizing_intrins = get_normalized_image_points(
        right_correspondences
    )

    # Create weights for homogeneous linear system.
    A = np.zeros((8, 9))
    for i in range(8):
        # (3, 1) @ (1, 3) -> (3, 3) -> 9
        A[i] = (
            np.array([*left_corr_norm[i], 1])[:, None]
            @ np.array([*right_corr_norm[i], 1])[None, :]
        ).reshape(9)

    # Lowest singular value is last.
    U, S, Vh = np.linalg.svd(A)  # .T @ A)

    # print(S)

    F_normalized = Vh[-1].reshape(3, 3)

    # Invert the normalization that was performed for numerical stability.
    F = left_normalizing_intrins.T @ F_normalized @ right_normalizing_intrins

    # Project to rank-2 matrix.
    FU, FS, FVh = np.linalg.svd(F)

    FS[-1] = 0
    F = FU @ np.diag(FS) @ FVh

    # Make F unit norm.
    F = F / np.linalg.norm(F)

    # Verify that the fundamental matrix equality constraint holds.
    # for i in range(8):
    #     print(
    #         np.array([*left_correspondences[i], 1])[None, :]
    #         @ F
    #         @ np.array([*right_correspondences[i], 1])[:, None]
    #     )

    # print(F)

    return F


# Now, we can have an interactive version.
def interactive_demo(
    left, right, camera: Camera, left_correspondences, right_correspondences
):
    F = compute_fundamental_matrix(camera, left_correspondences, right_correspondences)

    selected_point_side = "left"
    selected_point = (0, 0)

    plt.gcf().set_figwidth(12)
    plt.gcf().set_figheight(4)

    left_axes = plt.subplot(1, 2, 1)
    left_axes.imshow(left)

    right_axes = plt.subplot(1, 2, 2)
    right_axes.imshow(right)

    def click_callback(event: MouseEvent):
        nonlocal selected_point_side, selected_point

        if event.inaxes is None:
            return

        if event.inaxes == left_axes:
            selected_point_side = "left"
        else:
            selected_point_side = "right"

        selected_point = (event.xdata, event.ydata)

    plt.gcf().canvas.mpl_connect(
        "button_release_event",
        click_callback,  # type: ignore
    )

    plt.tight_layout()

    while 1:
        # Clear the list of artists.
        for artist in [*left_axes.get_children(), *right_axes.get_children()]:
            if isinstance(artist, (PathCollection, Line2D)):
                artist.remove()

        if selected_point_side == "left":
            left_axes.scatter(selected_point[0], selected_point[1], c="red", s=5)
        else:
            a, b, c = get_epipolar_line(np.array(selected_point), F.T)

            # Solve for x = 0.
            # ax + by = c
            # y = c/b

            # Solve for x = right.width
            # a(right.width) + by = c
            # y = (c - a * (right.width)) / b

            left_axes.plot(
                [0, right.shape[1] - 1],
                [c / b, (c - a * (right.shape[1])) / b],
                c="blue",
            )

        if selected_point_side == "right":
            right_axes.scatter(selected_point[0], selected_point[1], c="red", s=5)
        else:
            a, b, c = get_epipolar_line(np.array(selected_point), F)

            # Solve for x = 0.
            # ax + by = c
            # y = c/b

            # Solve for x = right.width
            # a(right.width) + by = c
            # y = (c - a * (right.width)) / b

            right_axes.plot(
                [0, right.shape[1] - 1],
                [c / b, (c - a * (right.shape[1])) / b],
                c="blue",
            )

        plt.pause(0.01)

    plt.show()


def main():
    base = os.path.abspath(os.path.dirname(__file__))

    left = image_to_1080p(cv2.imread(base + "/IMG_9728.JPG"))
    right = image_to_1080p(cv2.imread(base + "/IMG_9729.JPG"))
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    camera = Camera.from_calibration_file(
        "modules/calibs/iphone12promax_continuity_camera.json"
    )

    correspondences = np.loadtxt(base + "/selections.txt", skiprows=1, delimiter=",")
    left_correspondences = correspondences[:8, 2:]
    right_correspondences = correspondences[8:, 2:]

    F = compute_fundamental_matrix(camera, left_correspondences, right_correspondences)

    # Rectify points.
    left_correspondences_rect = camera.rectify_points(left_correspondences)

    # Create object points.
    left_correspondences_canonical_obj = camera.rectified_to_homogeneous(
        left_correspondences_rect
    )

    # Measure reprojection error.
    projected = camera.project_points(left_correspondences_canonical_obj)

    print(
        "Reprojection error:",
        np.linalg.norm(projected - left_correspondences),
        "pixels",
    )

    # Now, we can plot the correspondences.
    colors = ["red", "orange", "yellow", "green", "blue", "pink", "gold", "lightgrey"]

    plt.subplot(1, 2, 1)
    plt.imshow(left)

    for i in range(8):
        a, b, c = get_epipolar_line(right_correspondences[i], F.T)

        # Solve for x = 0.
        # ax + by = c
        # y = c/b

        # Solve for x = right.width
        # a(right.width) + by = c
        # y = (c - a * (right.width)) / b

        plt.scatter(
            [left_correspondences[i, 0]], [left_correspondences[i, 1]], c=colors[i], s=5
        )
        plt.plot(
            [0, right.shape[1] - 1],
            [c / b, (c - a * (right.shape[1])) / b],
            c=colors[i],
        )

    plt.subplot(1, 2, 2)
    plt.imshow(right)

    for i in range(8):
        a, b, c = get_epipolar_line(left_correspondences[i], F)

        # Solve for x = 0.
        # ax + by = c
        # y = c/b

        # Solve for x = right.width
        # a(right.width) + by = c
        # y = (c - a * (right.width)) / b

        plt.scatter(
            [right_correspondences[i, 0]],
            [right_correspondences[i, 1]],
            c=colors[i],
            s=5,
        )
        plt.plot(
            [0, right.shape[1] - 1],
            [c / b, (c - a * (right.shape[1])) / b],
            c=colors[i],
        )

    plt.tight_layout()
    plt.show()

    interactive_demo(left, right, camera, left_correspondences, right_correspondences)


if __name__ == "__main__":
    main()
