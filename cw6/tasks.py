import cv2
import imutils
from util import show_image

def task_1(image):
    resized = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    show_image("Resized to 50%", resized)

def task_2(image):
    resized = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    show_image("Resized 2x", resized)

def task_3(image):
    resized = cv2.resize(image, (200, 300))
    show_image("Resized to 200x300", resized)

def task_4(image):
    methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    names = ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LANCZOS4"]
    for method, name in zip(methods, names):
        resized = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3), interpolation=method)
        show_image(f"{name} Scaling 3x", resized)

def task_5(image):
    resized = imutils.resize(image, width=500)
    show_image("Resized width 500px", resized)

def task_6(image):
    h, w = image.shape[:2]
    scale = 400 / h
    resized = cv2.resize(image, (int(w * scale), 400))
    show_image("Resized height 400px", resized)

def task_7(image):
    resized = cv2.resize(image, (image.shape[1] // 5, image.shape[0] // 5), interpolation=cv2.INTER_AREA)
    show_image("Reduced 5x (INTER_AREA)", resized)

def task_8(image):
    resized_cubic = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    resized_lanczos = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4), interpolation=cv2.INTER_LANCZOS4)
    show_image("Upscaled 4x (INTER_CUBIC)", resized_cubic)
    show_image("Upscaled 4x (INTER_LANCZOS4)", resized_lanczos)

def task_9(image):
    for scale in range(100, 301, 20):
        resized = cv2.resize(image, (int(image.shape[1] * scale / 100), int(image.shape[0] * scale / 100)))
        show_image(f"Resized {scale}%", resized)
        cv2.waitKey(500)

def task_10(image):
    resized = imutils.resize(image, width=800)
    cv2.imwrite("resized_output.jpg", resized)
    show_image("Resized width 800px and saved", resized)