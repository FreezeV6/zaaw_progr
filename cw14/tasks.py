import cv2
import numpy as np
from util import show_image

def task_1(image):
    blur_image = cv2.blur(image, (5, 5))
    gaussian_blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    median_blur_image = cv2.medianBlur(image, 5)
    bilateral_filter_image = cv2.bilateralFilter(image, 9, 75, 75)

    show_image("Simple Blur", blur_image)
    show_image("Gaussian Blur", gaussian_blur_image)
    show_image("Median Blur", median_blur_image)
    show_image("Bilateral Filter", bilateral_filter_image)

def task_2(image):
    kernel_sizes = [(3, 3), (5, 5), (9, 9), (15, 15)]

    for kernel_size in kernel_sizes:
        blur_image = cv2.blur(image, kernel_size)
        show_image(f"Blur with kernel size {kernel_size}", blur_image)

def task_3(image):
    bilateral_image = cv2.bilateralFilter(image, 9, 75, 75)
    show_image("Bilateral Filter", bilateral_image)

def task_4(image):
    blur_image = cv2.blur(image, (5, 5))
    gaussian_blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    median_blur_image = cv2.medianBlur(image, 5)
    bilateral_filter_image = cv2.bilateralFilter(image, 9, 75, 75)

    show_image("Blurred Image", blur_image)
    show_image("Gaussian Blurred Image", gaussian_blur_image)
    show_image("Median Blurred Image", median_blur_image)
    show_image("Bilateral Filtered Image", bilateral_filter_image)

def task_5(image):
    noise_image = image.copy()
    noise_image = np.uint8(np.random.normal(0, 25, noise_image.shape) + noise_image)
    show_image("Image with Noise", noise_image)

    blur_image = cv2.blur(noise_image, (5, 5))
    gaussian_blur_image = cv2.GaussianBlur(noise_image, (5, 5), 0)
    median_blur_image = cv2.medianBlur(noise_image, 5)
    bilateral_filter_image = cv2.bilateralFilter(noise_image, 9, 75, 75)

    show_image("Blurred Image", blur_image)
    show_image("Gaussian Blurred Image", gaussian_blur_image)
    show_image("Median Blurred Image", median_blur_image)
    show_image("Bilateral Filtered Image", bilateral_filter_image)

def task_6(image):
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 3:2 * h // 3, w // 3:2 * w // 3] = 255

    background_blurred = cv2.GaussianBlur(image, (15, 15), 0)
    mask_colored = cv2.merge([mask, mask, mask])

    final_image = np.where(mask_colored == 255, image, background_blurred)
    show_image("Simulated Depth of Field", final_image)