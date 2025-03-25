import cv2
import numpy as np

from util import show_image

def task_1(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype="uint8")
    center = (w // 2, h // 2)
    axes = (w // 4, h // 3)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    show_image("Masked Face Region", masked)

def task_2(image):
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype="uint8") * 255
    eye_region = (w//2, h//2, w//2, h//10)
    x, y, ew, eh = eye_region
    cv2.rectangle(mask, (x - x // 2, y - y // 4), (x+ew, y+eh), 0, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    show_image("Eyes Hidden", masked)

def task_3(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 100, 100])  # Example for blue
    upper = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    show_image("Extracted Blue Color", result)