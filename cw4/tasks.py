import cv2
import imutils
import numpy as np
from util import show_image, RED, GREEN, BLUE, WHITE, BLACK

def task_1(image):
    show_image("Original", image)
    M = np.float32([[1, 0, 30], [0, 1, 40]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    show_image("Shifted", shifted)

def task_2(image):
    # to co wczesniej (nie wiem czy shifted czy po prostu orginalne co wczesniej)
    M = np.float32([[1, 0, 30], [0, 1, 40]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    shifted = imutils.translate(shifted, -20, -50)
    show_image("Shifted", shifted)

def task_3(image):
    shifted = imutils.translate(image, 360, 380)
    show_image("Shifted", shifted)

def task_4(image):
    shifted = imutils.translate(image, 100, 50)
    show_image("Shifted image utils", shifted)
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    show_image("Shifted cv2", shifted)

def task_5(image):
    h, w, _ = image.shape
    directions_y = ['down', 'up']
    directions_x = ['left', 'right']
    while True:
        try:
            direction_y, y, direction_x, x = input('Please enter shift values as integers and directions ([up/down] y [left/right] x) like: "up 40 left 30": ').split()
            x, y = int(x), int(y)
            if 0 <= x < w and 0 <= y < h and direction_x in directions_x and direction_y in directions_y:
                if direction_y == 'up':
                     y = -y
                if direction_x == 'left':
                    x = -x
                shifted = imutils.translate(image, x, y)
                show_image("Shifted", shifted)
                break
            else:
                print(f"Invalid values. Please enter values within the image dimensions: {h} x {w}, else the image will not be visible.")
        except ValueError:
            print('Invalid input. Please enter directions and two integers separated by space, like: "up 40 left 30", or "down 120 right 76"')

