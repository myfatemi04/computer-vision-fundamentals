import numpy as np


def image_to_1080p(image: np.ndarray):
    ORIGINAL_SIZE = (3024, 4032)
    TARGET_SIZE = (2160, 3840)
    DOWNSAMPLE = 2

    assert image.shape[:-1] == ORIGINAL_SIZE

    return image[
        ORIGINAL_SIZE[0] // 2
        - TARGET_SIZE[0] // 2 : ORIGINAL_SIZE[0] // 2
        + TARGET_SIZE[0] // 2 : DOWNSAMPLE,
        ORIGINAL_SIZE[1] // 2
        - TARGET_SIZE[1] // 2 : ORIGINAL_SIZE[1] // 2
        + TARGET_SIZE[1] // 2 : DOWNSAMPLE,
    ]
