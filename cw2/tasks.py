import cv2
from util import show_image
import numpy as np

def task_1(image):
    pixel = image[0, 0]
    print(f'Pixel at (0,0): {pixel}')

def task_2(image):
    modified_image = image.copy()
    h, w, _ = modified_image.shape
    modified_image[h-1, w-1] = (0, 0, 255)
    show_image("Before Modification", image)
    show_image("After Modification", modified_image)

def task_3(image):
    h, w, _ = image.shape
    center = (h//2, w//2)
    pixel = image[center]
    print(f'Pixel at center {center}: {pixel}')


def task_4(image):
    h, w, _ = image.shape
    while True:
        try:
            x, y = map(int, input('Please enter x, y: ').split())
            if 0 <= x < w and 0 <= y < h:
                image[y, x] = (0, 0, 0)
                show_image("Modified Image", image)
                break
            else:
                print(f"Invalid coordinates. Please enter values within the image dimensions: {h} x {w}")
        except ValueError:
            print('Invalid input. Please enter two integers separated by space.')

def task_5(image):
    h, w, _ = image.shape
    modified_image = image.copy()
    modified_image[:h//2, :w//2] = (255, 0, 0)
    show_image("Modified Image", modified_image)

def task_6(image):
    h, w, _ = image.shape
    center = (h//2, w//2)
    x1, x2 = center[1] - 50, center[1] + 50
    y1, y2 = center[0] - 50, center[0] + 50
    image[y1:y2, x1:x2] = (0, 0, 255)
    show_image("Modified Image", image)

def task_7(image):
    h, w, _ = image.shape
    dx, dy = w//3, h//3
    cropped = image[dy:2*dy, dx:2*dx]
    show_image("Cropped Center", cropped)

def task_8(image):
    modified_image = image.copy()
    modified_image[100, :] = (0, 255, 0)
    show_image("Before Modification", image)
    show_image("After Modification", modified_image)

def task_9(image):
    modified_image = image.copy()
    modified_image[50:100, 50:100] = (255, 255, 255)
    show_image("Before Modification", image)
    show_image("After Modification", modified_image)

def task_10(image):
    p1, p2 = image[50, 50], image[200, 200]
    diff = np.abs(p1 - p2)
    print(f'Difference: {diff}')

def task_11(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    print(f'Brightest pixel at {max_loc} with value {max_val}')