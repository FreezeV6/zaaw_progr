import cv2
from util import show_image
import matplotlib.pyplot as plt

def task_1(image):
    kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_square = cv2.erode(image, kernel_square, iterations=1)
    eroded_ellipse = cv2.erode(image, kernel_ellipse, iterations=1)
    show_image("Original", image)
    show_image("Eroded - Square", eroded_square)
    show_image("Eroded - Ellipse", eroded_ellipse)

def task_2(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')
    for i in range(1, 4):
        dilated = cv2.dilate(image, kernel, iterations=i)
        axs[i].imshow(dilated, cmap='gray')
        axs[i].set_title(f"Dilation {i}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def task_3(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    show_image("Original", image)
    show_image("Denoised - Opening", opened)

def task_4(image):
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_rect = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_rect)
    closed_ellipse = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_ellipse)
    show_image("Original", image)
    show_image("Closed - Rect", closed_rect)
    show_image("Closed - Ellipse", closed_ellipse)

def task_5(image):
    kernels = {
        "square": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        "ellipse": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        "cross": cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    }
    for name, kernel in kernels.items():
        eroded = cv2.erode(image, kernel)
        dilated = cv2.dilate(image, kernel)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        show_image(f"{name} - Erosion", eroded)
        show_image(f"{name} - Dilation", dilated)
        show_image(f"{name} - Opening", opened)
        show_image(f"{name} - Closing", closed)
        show_image(f"{name} - Gradient", gradient)

def task_6(image):
    # Example: Improve license plate readability
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    show_image("Original", image)
    show_image("Thresholded", thresh)
    show_image("After Morphological Closing", closed)