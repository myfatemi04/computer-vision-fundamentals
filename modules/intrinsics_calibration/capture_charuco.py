import os
from pathlib import Path
import pickle

import cv2
import numpy as np
import argparse
from .calibrate_intrinsics_charuco import calibrate

SCREEN_WIDTH_PX = 1728
SCREEN_HEIGHT_PX = 1117
# DIAGONAL_IN = 16.2
DIAGONAL_MM = 411.48
DIAGONAL_PX = (SCREEN_WIDTH_PX**2 + SCREEN_HEIGHT_PX**2) ** 0.5
SCREEN_WIDTH_MM = DIAGONAL_MM / DIAGONAL_PX * SCREEN_WIDTH_PX
SCREEN_HEIGHT_MM = DIAGONAL_MM / DIAGONAL_PX * SCREEN_HEIGHT_PX
PIXEL_WIDTH_MM = SCREEN_WIDTH_MM / SCREEN_WIDTH_PX
PIXEL_HEIGHT_MM = SCREEN_HEIGHT_MM / SCREEN_HEIGHT_PX


def make_charuco_board(
    width: int, height: int, square_size_mm: float, dictionary=cv2.aruco.DICT_7X7_100
):
    square_size_px = square_size_mm / PIXEL_WIDTH_MM
    image_size = (int(square_size_px * width), int(square_size_px * height))
    margin_size = int(square_size_px / 2)

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
    board = cv2.aruco.CharucoBoard(
        (width, height),
        square_size_px,
        square_size_px * 0.8,
        dictionary,
    )
    image = board.generateImage(image_size, marginSize=margin_size)

    return image, board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Path to store calibration result in.",
    )
    parser.add_argument(
        "-W", "--width", default=8, help="ChArUco board width (default: 8)"
    )
    parser.add_argument(
        "-H", "--height", default=7, help="ChArUco board height (default: 7)"
    )
    parser.add_argument(
        "-s",
        "--source",
        default="camera:1",
        help="The source to use. You can either specify a string of the form 'camera:X', for a camera device ID, or a path (any string that does not begin with 'camera:'), or the string 'future_video' to export the ChArUco board to an image file that can be opened on your screen.",
        type=str,
    )
    ns = parser.parse_args()
    output_path = Path(ns.output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "frames").mkdir(parents=True, exist_ok=True)

    checkerboard_width = ns.width
    checkerboard_height = ns.height
    checkerboard_square_size_mm = 10
    checkerboard_image, board = make_charuco_board(
        checkerboard_width, checkerboard_height, checkerboard_square_size_mm
    )
    cv2.imshow("ChArUco Board", checkerboard_image)
    cv2.waitKey(1)

    detector = cv2.aruco.CharucoDetector(board)

    detections = {"object_points": [], "image_points": []}
    detections_counter = 0

    source = ns.source

    cv2.imwrite(str(output_path / "charuco.png"), checkerboard_image)
    if source == "future_video":
        return

    elif source.startswith("camera:"):
        source = int(source[7:])

    else:
        if not os.path.exists(source):
            raise ValueError("Path for source video does not exist: " + source)

    cap = cv2.VideoCapture(source)
    DOWNSAMPLE = 2

    while 1:
        ret, frame = cap.read()

        if not ret:
            break

        frame_original = frame
        frame = np.ascontiguousarray(frame[::DOWNSAMPLE, ::DOWNSAMPLE, :])

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Find checkerboard corners.
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            frame_gray
        )
        if charuco_corners is not None:
            # Draw the result on the frame.
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

            (detected_object_points, detected_image_points) = board.matchImagePoints(
                charuco_corners,  # type: ignore
                charuco_ids,
            )

            if detected_image_points.shape[0] >= 6:
                # (n, 1, 3) -> (n, 3)
                detected_image_points = detected_image_points[:, 0, :]
                detected_object_points = detected_object_points[:, 0, :]

                # Store the detection.
                detections["object_points"].append(detected_object_points)
                detections["image_points"].append(detected_image_points * DOWNSAMPLE)

                cv2.imwrite(
                    str(output_path / "frames" / f"frame_{detections_counter}.jpg"),
                    frame_original,
                )
                detections_counter += 1

        cv2.imshow("Frame", frame)
        if ord("q") == cv2.waitKey(1):
            break

    cap.release()

    # Store the detected frames to disk.
    with open(output_path / "detections.pkl", "wb") as f:
        pickle.dump(detections, f)

    # Now, run the calibration itself, on the detection results.
    print(f"... Running calibration on path {output_path} ...")
    calibrate(output_path, save=True)


if __name__ == "__main__":
    main()
