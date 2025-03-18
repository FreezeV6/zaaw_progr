import cv2
import imutils
from util import show_image

def task_1(image):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    show_image("Rotated 45 degrees", rotated)

def task_2(image):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, -90, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    show_image("Rotated -90 degrees", rotated)

def task_3(image):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((0, 0), 30, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    show_image("Rotated 30 degrees around top-left corner", rotated)

def task_4(image):
    angle = float(input("Enter rotation angle: "))
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    show_image(f"Rotated {angle} degrees", rotated)

def task_5(image):
    rotated = imutils.rotate(image, 180)
    show_image("Rotated 180 degrees (imutils)", rotated)

def task_6(image):
    rotated = imutils.rotate_bound(image, -33)
    show_image("Rotated -33 degrees (imutils, rotate_bound)", rotated)

def task_7(image):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 60, 1.0)
    rotated_warpAffine = cv2.warpAffine(image, matrix, (w, h))
    rotated_imutils = imutils.rotate(image, 60)
    show_image("Rotated 60 degrees (warpAffine)", rotated_warpAffine)
    show_image("Rotated 60 degrees (imutils)", rotated_imutils)

def task_8(image):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
    rotated_once = cv2.warpAffine(image, matrix, (w, h))
    rotated_twice = cv2.warpAffine(rotated_once, matrix, (w, h))
    rotated_thrice = cv2.warpAffine(rotated_twice, matrix, (w, h))
    rotated_90 = cv2.warpAffine(image, cv2.getRotationMatrix2D(center, 90, 1.0), (w, h))
    show_image("Rotated 3x 30 degrees", rotated_thrice)
    show_image("Rotated 90 degrees", rotated_90)

def task_9(image):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 75, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    cv2.imwrite("rotated_output.jpg", rotated)
    show_image("Rotated 75 degrees and saved", rotated)

def task_10(image):
    for angle in range(0, 361, 15):
        rotated = imutils.rotate(image, angle)
        show_image(f"Rotated {angle} degrees", rotated)
        cv2.waitKey(500)