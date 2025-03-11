import cv2
import numpy as np
from util import show_image, RED, GREEN, BLUE, WHITE, BLACK

def task_1(image):
    h, w = image.shape[:2]
    cv2.line(image, (h//2, w//2), (w, h), GREEN)
    show_image("Line", image)

def task_2():
    canvas = np.zeros((400, 400, 3), dtype="uint8")
    cv2.rectangle(canvas, (0,0), (100, 50), GREEN)
    cv2.rectangle(canvas, (400,400), (300, 350), RED, 3)
    show_image("Rectangle", canvas)

def task_3():
    canvas = np.zeros((300, 300, 3), dtype="uint8")
    h, w = canvas.shape[:2]
    cv2.circle(canvas, (30, 30), 30, BLUE)
    cv2.circle(canvas, (h//2, w//2), 40, GREEN)
    show_image("Rectangle", canvas)

def task_4():
    canvas = np.zeros((700, 700, 3), dtype="uint8")
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (h//2 - 50, w//2 - 50), (h//2 + 50, w//2 + 50), RED, 3)
    cv2.circle(canvas, (h//2, w//2), 30, GREEN)
    show_image("Circle", canvas)

def task_5():
    canvas = np.zeros((800, 800, 3), dtype="uint8")
    h, w = canvas.shape[:2]
    for r in range(0, 700, 20):
        cv2.rectangle(canvas, (h//2 - r, w//2 - r), (h//2 + r, w//2 + r), WHITE)
    show_image("Loop", canvas)

def task_6(image):
    cv2.circle(image, (222, 60), 15, RED, 30)
    cv2.circle(image, (330, 72), 15, RED, 30)
    cv2.rectangle(image, (215, 182), (330, 218), GREEN, 40)
    cv2.circle(image, (273, 129), 150, BLUE)
    show_image("Rectangle", image)
