import cv2
from util import show_image

def task_1(image):
    roi = image[:100, :100]
    show_image("ROI (100x100 from top-left)", roi)

def task_2(image):
    h = image.shape[0]
    roi = image[h//2:, :]
    show_image("Bottom Half", roi)

def task_3(image):
    w = image.shape[1]
    roi = image[:, w//2:]
    show_image("Right Half", roi)

def task_4(image):
    startX = int(input("Enter startX: "))
    endX = int(input("Enter endX: "))
    startY = int(input("Enter startY: "))
    endY = int(input("Enter endY: "))
    roi = image[startY:endY, startX:endX]
    show_image("User Selected ROI", roi)

def task_5(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        show_image("Cropped Face", face)
        return
    print("No face detected.")

def task_6(image):
    roi = image[50:150, 50:150].copy()
    target_h, target_w = roi.shape[:2]
    h, w = image.shape[:2]
    if 200 + target_h <= h and 200 + target_w <= w:
        image[200:200+target_h, 200:200+target_w] = roi
        show_image("Copied ROI", image)
    else:
        print("Error: ROI exceeds image dimensions.")

def task_7(image):
    h, w = image.shape[:2]
    grid_h, grid_w = h // 3, w // 3
    for i in range(3):
        for j in range(3):
            roi = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            show_image(f"Grid {i},{j}", roi)

def task_8(image):
    h, w = image.shape[:2]
    step = 10
    for x in range(0, w-100, step):
        roi = image[:, x:x+100]
        show_image("Moving ROI", roi)
        cv2.waitKey(500)

def task_9(image):
    roi = image[:300, :300]
    cv2.imwrite("cropped_image.jpg", roi)
    show_image("Saved Cropped Image", roi)