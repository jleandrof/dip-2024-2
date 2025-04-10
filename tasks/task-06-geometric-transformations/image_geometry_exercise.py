# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def translate(img, h=0, v=0):
    img = np.roll(img, h, axis=0) # horizontal
    img = np.roll(img, v, axis=1) # vertical

    return img

def rotate(img):
    return np.rot90(img, axes=(1, 0))

def stretch(img, scale=1.5):
    print(img.shape)
    print((img.shape[0], int(img.shape[1]*scale)))

    return np.resize(img, (img.shape[0], int(img.shape[1]*scale)))

def mirror(img):
    return np.fliplr(img)

def distort(image, k=2):
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    y, x = np.indices((h, w), dtype=np.float32)

    x_norm = (x - cx) / w
    y_norm = (y - cy) / h

    r = np.sqrt(x_norm**2 + y_norm**2)

    factor = 1 + k * r**2

    x_distorted = x_norm * factor
    y_distorted = y_norm * factor

    x_new = (x_distorted * w + cx).astype(np.float32)
    y_new = (y_distorted * h + cy).astype(np.float32)

    x_new = np.clip(x_new, 0, w - 1)
    y_new = np.clip(y_new, 0, h - 1)

    xd = np.round(x_new).astype(np.int32)
    yd = np.round(y_new).astype(np.int32)

    return image[yd, xd]

def apply_geometric_transformations(img: np.ndarray) -> dict:
    
    return {
        "translated": translate(img, 100, 100),
        "rotated": rotate(img),
        "stretched": stretch(img),
        "mirrored": mirror(img),
        "distorted": distort(img)
    }

if __name__ == "__main__":

    import cv2

    i1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)

    print(apply_geometric_transformations(i1))

    cv2.imshow('original', i1)
    # cv2.imshow('translated', translate(i1, 100, 100))
    # cv2.imshow('rotated', rotate(i1))
    # cv2.imshow('stretched', stretch(i1))
    # cv2.imshow('flipped', mirror(i1))
    # cv2.imshow('distorted', distort(i1))
    cv2.waitKey(0)