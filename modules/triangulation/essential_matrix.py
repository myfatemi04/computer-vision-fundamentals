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
from .load_image import image_to_1080p


base = os.path.abspath(os.path.dirname(__file__))

left = image_to_1080p(cv2.imread(base + "/IMG_9728.JPG"))
right = image_to_1080p(cv2.imread(base + "/IMG_9729.JPG"))
