import os
import pickle

import cv2
import numpy as np

SCREEN_WIDTH_PX = 1728
SCREEN_HEIGHT_PX = 1117
# DIAGONAL_IN = 16.2
DIAGONAL_MM = 411.48
DIAGONAL_PX = (SCREEN_WIDTH_PX**2 + SCREEN_HEIGHT_PX**2) ** 0.5
SCREEN_WIDTH_MM = DIAGONAL_MM / DIAGONAL_PX * SCREEN_WIDTH_PX
SCREEN_HEIGHT_MM = DIAGONAL_MM / DIAGONAL_PX * SCREEN_HEIGHT_PX
PIXEL_WIDTH_MM = SCREEN_WIDTH_MM / SCREEN_WIDTH_PX
PIXEL_HEIGHT_MM = SCREEN_HEIGHT_MM / SCREEN_HEIGHT_PX


def make_charuco_board(width: int, height: int, square_size_mm: float):
    square_size_px = square_size_mm / PIXEL_WIDTH_MM
    image_size = (int(square_size_px * width), int(square_size_px * height))
    margin_size = int(square_size_px / 2)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
    board = cv2.aruco.CharucoBoard(
        (width, height),
        square_size_px,
        square_size_px * 0.8,
        dictionary,
    )
    image = board.generateImage(image_size, marginSize=margin_size)
    object_points = board.getObjPoints()

    return image, object_points, board


def main():
    checkerboard_width = 8
    checkerboard_height = 7
    checkerboard_square_size_mm = 10
    checkerboard_image, checkerboard_object_points, board = make_charuco_board(
        checkerboard_width, checkerboard_height, checkerboard_square_size_mm
    )
    cv2.imshow("ChArUco Board", checkerboard_image)
    cv2.waitKey(1)

    print(checkerboard_object_points)

    detector = cv2.aruco.CharucoDetector(board)

    detections = {
        "object_points": [],
        "image_points": [],
    }

    cap = cv2.VideoCapture(1)
    DOWNSCALE = 2

    if not os.path.exists("frames"):
        os.mkdir("frames")

    while 1:
        ret, frame = cap.read()

        if not ret:
            break

        frame_original = frame
        frame = np.ascontiguousarray(frame[::DOWNSCALE, ::DOWNSCALE, :])

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Find checkerboard corners.
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            frame_gray
        )
        corners_ = charuco_corners

        if corners_ is not None:
            # Draw the result on the frame.
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

            (detected_object_points, detected_image_points) = board.matchImagePoints(
                charuco_corners,  # type: ignore
                charuco_ids,
            )

            # Store the detection.
            detections["object_points"].append(detected_object_points)
            detections["image_points"].append(detected_image_points)

            cv2.imwrite(f"frames/frame_{len(detections)}.jpg", frame_original)

        cv2.imshow("Frame", frame)
        if ord("q") == cv2.waitKey(1):
            break

    cap.release()

    # Store the detected frames to disk.
    with open("detections.pkl", "wb") as f:
        pickle.dump(detections, f)

    np.save("object_points.npy", checkerboard_object_points)


if __name__ == "__main__":
    main()
