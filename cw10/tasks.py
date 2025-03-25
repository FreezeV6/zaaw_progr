import cv2
import numpy as np
from util import show_image


def task_1(image=None):
    shape = (300, 300)
    triangle = np.zeros(shape, dtype=np.uint8)
    circle = np.zeros(shape, dtype=np.uint8)

    pts = np.array([[150, 50], [100, 250], [200, 250]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(triangle, [pts], 255)
    cv2.circle(circle, (150, 150), 100, 255, -1)

    and_result = cv2.bitwise_and(triangle, circle)
    or_result = cv2.bitwise_or(triangle, circle)
    xor_result = cv2.bitwise_xor(triangle, circle)
    not_triangle = cv2.bitwise_not(triangle)

    show_image("Triangle", triangle)
    show_image("Circle", circle)
    show_image("AND", and_result)
    show_image("OR", or_result)
    show_image("XOR", xor_result)
    show_image("NOT Triangle", not_triangle)

def task_2(image1, image2):
    h, w = image1.shape[:2]
    image2 = cv2.resize(image2, (w, h))
    if image1.shape != image2.shape:
        print("Error: Images must have the same dimensions.")
        return
    diff = cv2.bitwise_xor(image1, image2)
    show_image("XOR Difference", diff)
