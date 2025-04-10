# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import skimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    return ski.exposure.match_histograms(source_img, reference_img, channel_axis=2)

if __name__ == "__main__":

    import cv2

    src = cv2.imread('source.jpg', cv2.IMREAD_COLOR_RGB)
    ref = cv2.imread('reference.jpg', cv2.IMREAD_COLOR_RGB)

    cv2.imshow('source', src)
    cv2.imshow('reference', ref)
    cv2.imshow('result', match_histograms_rgb(src, ref))

    cv2.waitKey(0)
