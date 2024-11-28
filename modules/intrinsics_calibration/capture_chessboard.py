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


def make_checkerboard(width: int, height: int, square_size_mm: float):
    square_width_px = square_size_mm / PIXEL_WIDTH_MM
    square_height_px = square_size_mm / PIXEL_HEIGHT_MM

    x_coordinates = (np.arange(width + 3) * square_width_px).astype(int)
    y_coordinates = (np.arange(height + 3) * square_height_px).astype(int)
    image = np.ones((y_coordinates[-1], x_coordinates[-1]), dtype=np.uint8) * 255

    object_points = []

    # The results from findChessboardCorners are returned in row-major order.
    # Thus, we return the object points in row-major order as well.
    for j in range(height):
        for i in range(width):
            color = (i + j) % 2
            image[
                y_coordinates[j + 1] : y_coordinates[j + 2],
                x_coordinates[i + 1] : x_coordinates[i + 2],
            ] = (
                255 * color
            )

            if i > 0 and j > 0:
                object_points.append([square_size_mm * i, square_size_mm * j, 0])

    return image, np.array(object_points)


def main():
    checkerboard_width = 8
    checkerboard_height = 7
    checkerboard_square_size_mm = 10
    checkerboard_image, checkerboard_object_points = make_checkerboard(
        checkerboard_width, checkerboard_height, checkerboard_square_size_mm
    )
    cv2.imshow("Checkerboard", checkerboard_image)
    cv2.waitKey(1)

    corners = []

    cap = cv2.VideoCapture(1)
    DOWNSCALE = 2

    if not os.path.exists("detections"):
        os.mkdir("detections")

    while 1:
        ret, frame = cap.read()

        if not ret:
            break

        frame_original = frame
        frame = np.ascontiguousarray(frame[::DOWNSCALE, ::DOWNSCALE, :])

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Find checkerboard corners.
        ret, corners_ = cv2.findChessboardCorners(
            frame_gray, (checkerboard_width - 1, checkerboard_height - 1)
        )

        if corners_ is not None:
            # Draw the result on the frame.
            cv2.drawChessboardCorners(
                frame, (checkerboard_width - 1, checkerboard_height - 1), corners_, True
            )

            # Store the detection.
            corners.append(corners_ * DOWNSCALE)

            cv2.imwrite(f"detections/frame_{len(corners)}.jpg", frame_original)

        cv2.imshow("Frame", frame)
        if ord("q") == cv2.waitKey(1):
            break

    cap.release()

    # Store the detected frames to disk.
    with open("corners.pkl", "wb") as f:
        pickle.dump(corners, f)

    np.save("object_points.npy", checkerboard_object_points)


if __name__ == "__main__":
    main()
