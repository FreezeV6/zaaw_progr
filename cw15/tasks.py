import cv2
import numpy as np
from util import show_image

def task_1(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show_image("RGB Image", rgb_image)

    r, g, b = cv2.split(rgb_image)
    show_image("Red Channel", r)
    show_image("Green Channel", g)
    show_image("Blue Channel", b)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    show_image("HSV Image", hsv_image)

    h, s, v = cv2.split(hsv_image)
    show_image("Hue Channel", h)
    show_image("Saturation Channel", s)
    show_image("Value Channel", v)

def task_2(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)

    s = cv2.add(s, 30)

    modified_hsv = cv2.merge([h, s, v])
    modified_image = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)

    show_image("Original Image", image)
    show_image("Modified Image", modified_image)

def task_3(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    result = cv2.bitwise_and(image, image, mask=blue_mask)
    show_image("Blue Objects", result)

def task_4(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hsv_image[:, :, 0] = hsv_image[:, :, 0] + 30

    modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # d. Wyświetl obraz przed i po zmianie odcienia
    show_image("Original Image", image)
    show_image("Modified Image", modified_image)

def task_5(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    result = cv2.bitwise_and(image, image, mask=green_mask)
    show_image("Green Objects", result)

def task_6(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    result = cv2.bitwise_and(image, image, mask=skin_mask)
    show_image("Skin Detection", result)

def task_7(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)

    s_low = cv2.subtract(s, 50)

    s_high = cv2.add(s, 50)

    show_image("Original Image", image)
    show_image("Low Saturation Image", cv2.cvtColor(cv2.merge([h, s_low, v]), cv2.COLOR_HSV2BGR))
    show_image("High Saturation Image", cv2.cvtColor(cv2.merge([h, s_high, v]), cv2.COLOR_HSV2BGR))

def task_8(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    combined_mask = cv2.bitwise_or(red_mask, cv2.bitwise_or(blue_mask, green_mask))

    result = cv2.bitwise_and(image, image, mask=combined_mask)
    show_image("Segmented Colors", result)
