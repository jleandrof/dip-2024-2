import argparse
from urllib.request import urlopen
import numpy as np
import cv2 as cv

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###

    res = urlopen(url)
    image = cv.imdecode(np.array(bytearray(res.read()), dtype=np.uint8), **kwargs)

    ### END CODE HERE ###
    
    return image

load_image_from_url()
