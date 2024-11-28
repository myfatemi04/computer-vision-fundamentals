import cv2
import numpy as np
import pickle


def check_chessboard_corners(corners: np.ndarray):
    # See if the direction between adjacent points is similar or not.
    # How do you check if something is grid-like?
    # A simple check for my case is to see if the step between consecutive corners is pretty consistent.
    corners = corners[:, 0, :].reshape((6, 7, 2))
    row_difference = corners[1:] - corners[:-1]
    col_difference = corners[:, 1:] - corners[:, :-1]

    row_diff_var = row_difference.var(axis=(0, 1))
    col_diff_var = col_difference.var(axis=(0, 1))

    print(row_diff_var, col_diff_var)


def main():
    object_points = np.load("object_points.npy")

    with open("corners.pkl", "rb") as f:
        corners = pickle.load(f)

    images: list[np.ndarray] = []

    corners2 = []

    for i in range(len(corners)):
        if corners[i].shape != (42, 1, 2):
            continue

        if i % 10 != 0:
            continue

        check_chessboard_corners(corners[i])

        images.append(cv2.imread(f"detections/frame_{i+1}.jpg"))
        corners2.append(corners[i])

    corners = np.stack(corners2)

    objpoints = [object_points.astype(np.float32)] * len(corners)

    ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(
        # mm -> meters
        objpoints / 1000,  # type: ignore
        corners,  # type: ignore
        (images[0].shape[1], images[0].shape[0]),
        None,  # type: ignore
        None,  # type: ignore
    )

    print(mtx)

    intrinsics_pred = [
        [2.25268834e03, 0.00000000e00, 9.39407581e02],
        [0.00000000e00, 2.25173110e03, 5.62981781e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]

    # Pretty close to center huh
    print(images[0].shape, (intrinsics_pred[1][2], intrinsics_pred[0][2]))


if __name__ == "__main__":
    main()
