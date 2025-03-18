import cv2
import numpy as np
from util import show_image


def task_1(image):
    numpy_result = np.clip(image + 50, 0, 255).astype(np.uint8)
    opencv_result = cv2.add(image, np.full(image.shape, 50, dtype=np.uint8))
    show_image("Increased Brightness (NumPy)", numpy_result)
    show_image("Increased Brightness (OpenCV)", opencv_result)


def task_2(image):
    numpy_result = np.clip(image + 150, 0, 255).astype(np.uint8)
    opencv_result = cv2.add(image, np.full(image.shape, 150, dtype=np.uint8))
    show_image("Overexposed (NumPy)", numpy_result)
    show_image("Overexposed (OpenCV)", opencv_result)


def task_3(image):
    numpy_result = np.clip(image - 80, 0, 255).astype(np.uint8)
    opencv_result = cv2.subtract(image, np.full(image.shape, 80, dtype=np.uint8))
    show_image("Darkened (NumPy)", numpy_result)
    show_image("Darkened (OpenCV)", opencv_result)


def task_4(image):
    modified = image.copy()
    modified[:, :, 2] = np.clip(modified[:, :, 2] + 30, 0, 255)  # Red channel
    modified[:, :, 1] = np.clip(modified[:, :, 1] - 20, 0, 255)  # Green channel
    modified[:, :, 0] = np.clip(modified[:, :, 0] + 10, 0, 255)  # Blue channel
    show_image("Custom Instagram Filter", modified)


def task_5(image1, image2):
    if image1.shape != image2.shape:
        print("Error: Images must have the same dimensions.")
        return
    difference = cv2.absdiff(image1, image2)
    show_image("Difference Between Images", difference)
