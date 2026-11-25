
"""Consistent preprocessing shared by training and inference."""
import cv2

IMAGE_SIZE = 256

def preprocess_array(image, size=IMAGE_SIZE):
    if image is None:
        raise ValueError("Could not read image")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    return cv2.GaussianBlur(image, (3, 3), 0)

def preprocess_image(path, size=IMAGE_SIZE):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read uploaded image: {path}")
    return preprocess_array(image, size=size)
