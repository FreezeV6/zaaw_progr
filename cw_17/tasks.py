import cv2
from util import show_image


def task_1(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, threshold_simple = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    show_image("Simple Threshold", threshold_simple)

    _, threshold_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_image("Otsu Threshold", threshold_otsu)

    threshold_mean = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    show_image("Adaptive Threshold Mean", threshold_mean)

    threshold_gaussian = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                               2)
    show_image("Adaptive Threshold Gaussian", threshold_gaussian)


def task_2(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    block_sizes = [11, 21, 31, 41]
    for block_size in block_sizes:
        threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                block_size, 2)
        show_image(f"Block Size {block_size}", threshold_image)


def task_3(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_mean = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    show_image("Adaptive Threshold Mean C", threshold_mean)

    threshold_gaussian = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                               2)
    show_image("Adaptive Threshold Gaussian C", threshold_gaussian)

    c_values = [2, 5, 10, 15]
    for c in c_values:
        threshold_mean_c = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, c)
        show_image(f"Mean C={c}", threshold_mean_c)

        threshold_gaussian_c = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                     11, c)
        show_image(f"Gaussian C={c}", threshold_gaussian_c)



def task_4(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_text = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    show_image("Text Segmentation", threshold_text)


def task_5(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_roi = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    result = cv2.bitwise_and(image, image, mask=threshold_roi)
    show_image("ROI Mask", result)


def task_6(image):
    def nothing(x):
        pass

    cv2.namedWindow("Adaptive Threshold", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("Block Size", "Adaptive Threshold", 11, 51, nothing)  # Ustawienie maksymalnego na 51
    cv2.createTrackbar("C Value", "Adaptive Threshold", 2, 20, nothing)

    while True:
        block_size = cv2.getTrackbarPos("Block Size", "Adaptive Threshold")
        c_value = cv2.getTrackbarPos("C Value", "Adaptive Threshold")

        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_value)

        cv2.imshow("Adaptive Threshold", threshold_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



