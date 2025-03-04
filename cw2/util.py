import cv2

def load_image(path):
    return cv2.imread(path)

def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
