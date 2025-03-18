import cv2
from util import show_image

def task_1(image):
    flipped = cv2.flip(image, 1)
    show_image("Flipped Horizontally", flipped)

def task_2(image):
    flipped = cv2.flip(image, 0)
    show_image("Flipped Vertically", flipped)

def task_3(image):
    flipped = cv2.flip(image, -1)
    show_image("Flipped Both Axes", flipped)

def task_4(image):
    flipped_h = cv2.flip(image, 1)
    flipped_v = cv2.flip(image, 0)
    flipped_both = cv2.flip(image, -1)
    show_image("Original Image", image)
    show_image("Flipped Horizontally", flipped_h)
    show_image("Flipped Vertically", flipped_v)
    show_image("Flipped Both Axes", flipped_both)

def task_5(image):
    h, w = image.shape[:2]
    roi = image[:, w//2:].copy()
    flipped_roi = cv2.flip(roi, 1)
    image[:, w//2:] = flipped_roi
    show_image("Flipped Right Half", image)

def task_6(image):
    choice = int(input("Enter flip mode (0 - vertical, 1 - horizontal, -1 - both): "))
    if choice in [0, 1, -1]:
        flipped = cv2.flip(image, choice)
        show_image(f"Flipped mode {choice}", flipped)
    else:
        print("Invalid choice.")