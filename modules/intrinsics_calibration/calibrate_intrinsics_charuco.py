import argparse
import json
import pickle
from pathlib import Path

import cv2


def calibrate(dir: Path, save=True):
    with open(dir / "detections.pkl", "rb") as f:
        detections = pickle.load(f)

    object_points = detections["object_points"]
    image_points = detections["image_points"]

    frame_path = next(
        (frame for frame in (dir / "frames").iterdir() if frame.name.endswith(".jpg")),
        None,
    )
    if frame_path is None:
        raise FileNotFoundError("No frames found in `frames` folder.")

    image = cv2.imread(str(frame_path))
    if image is None:
        raise RuntimeError("Unable to load image from path " + str(frame_path))

    ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        (image.shape[1], image.shape[0]),
        None,  # type: ignore
        None,  # type: ignore
    )

    result_dict = {
        "matrix": mtx.tolist(),
        "distortion": dst.tolist(),
        "rvecs": [rv[:, 0].tolist() for rv in rvecs],
        "tvecs": [tv[:, 0].tolist() for tv in tvecs],
    }

    if save:
        with open(dir / "calibration_result.json", "w") as f:
            json.dump(result_dict, f)

    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="Directory of ChArUco capture.")
    args = parser.parse_args()

    calibrate(Path(args.dir))


if __name__ == "__main__":
    main()
