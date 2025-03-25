import cv2
import numpy as np
from util import show_image

def task_1(image):
    B, G, R = cv2.split(image)
    show_image("Blue Channel", B)
    show_image("Green Channel", G)
    show_image("Red Channel", R)
    cv2.imwrite("blue_channel.jpg", B)
    cv2.imwrite("green_channel.jpg", G)
    cv2.imwrite("red_channel.jpg", R)

def task_2(image):
    B, G, R = cv2.split(image)
    show_image("Blue Channel", B)
    show_image("Green Channel", G)
    show_image("Red Channel", R)
    print("Look for features that appear strongly in one channel and weakly in others.")

def task_3(image):
    B, G, R = cv2.split(image)
    merged_rgb = cv2.merge([R, B, G])
    show_image("Swapped Channels (R, B, G)", merged_rgb)

    G[:] = 0  # Set green channel to zero
    modified = cv2.merge([B, G, R])
    show_image("Green Channel Zeroed", modified)

def task_4(image):
    B, G, R = cv2.split(image)
    R = cv2.add(R, 50)
    merged = cv2.merge([B, G, R])
    show_image("Enhanced Red Channel", merged)

def task_5(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    red_channel = image[:, :, 2]
    red_boost = cv2.add(red_channel, (mask > 0).astype(np.uint8) * 50)
    image[:, :, 2] = np.clip(red_boost, 0, 255)
    show_image("Red Enhanced with Mask", image)

def task_6():
    image = cv2.imread(cv2.samples.findFile("images/opencv.png"))
    if image is None:
        print("OpenCV logo not found.")
        return
    B, G, R = cv2.split(image)
    swapped = cv2.merge([R, G, B])
    show_image("Swapped Red and Blue", swapped)

    zero_blue = image.copy()
    zero_blue[:, :, 0] = 0
    show_image("Blue Channel Removed", zero_blue)