# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def mean_square_error(a, b):
    return ((a - b)**2).mean()

def peak_signal_to_noise_ratio(m, mse):
    return 20*np.log10(m)-10*np.log10(mse)

def structural_similarity_index(a, b, m=255):
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    C1 = (0.01 * m)**2
    C2 = (0.03 * m)**2

    mu1 = a.mean()
    mu2 = b.mean()

    sigma1_sq = ((a - mu1) ** 2).mean()
    sigma2_sq = ((b - mu2) ** 2).mean()
    sigma12 = ((a - mu1) * (b - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    if denominator == 0:
        return 0

    return numerator / denominator

def normalized_pearson_correlation_coefficient(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)

    numerator = np.sum(a_mean * b_mean)
    denominator = np.sqrt(np.sum(a_mean ** 2) * np.sum(b_mean ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

    mse = mean_square_error(i1, i2)
    psnr = peak_signal_to_noise_ratio(255, mse)
    ssim = structural_similarity_index(i1, i2)
    npcc = normalized_pearson_correlation_coefficient(i1, i2)

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "npcc": npcc
    }

# Example Usage
if __name__ == "__main__":

    import cv2

    i1 = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)
    i2 = cv2.imread('image2.jpg', cv2.IMREAD_COLOR)

    print(compare_images(i1, i2))
