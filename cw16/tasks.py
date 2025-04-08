import cv2
import numpy as np
from util import show_image

# Zadanie 1 - Wstęp do progowania
def task_1(image):
    # a. Wczytaj obraz, przekształć do skali szarości
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # i. Zastosuj progowanie dla T = 30
    _, threshold_30 = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
    show_image("Threshold T=30", threshold_30)

    # ii. Zastosuj progowanie dla T = 100
    _, threshold_100 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    show_image("Threshold T=100", threshold_100)

    # iii. Zastosuj progowanie dla T = 200
    _, threshold_200 = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    show_image("Threshold T=200", threshold_200)

    # b. Porównaj uzyskane binarne obrazy
    # Komentarze w kodzie

# Zadanie 2 - Wpływ rozmycia
def task_2(image):
    # a. Do jednego z obrazów z Zadania 1 zastosuj rozmycie Gaussa
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # b. Porównaj wynik progowania z i bez rozmycia
    _, threshold_original = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    _, threshold_blurred = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)

    show_image("Original Threshold", threshold_original)
    show_image("Blurred Threshold", threshold_blurred)

    # Komentarze do analizy wpływu rozmycia

# Zadanie 3 - Efekty erozji
def task_3(image):
    # a. Do jednego z uzyskanych obrazów binarnych z Zadania 2 zastosuj operację erozji
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    eroded_image = cv2.erode(threshold_image, None, iterations=1)

    # b. Opisz jak zmienił się obraz
    show_image("Eroded Image", eroded_image)
    # Komentarze do analizy efektów erozji

# Zadanie 4 - Wpływ oświetlenia
def task_4(image):
    # a. Zwiększ jasność obrazu o wartość 50
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightened_image = cv2.add(gray_image, 50)

    # b. Zastosuj podstawowe progowanie na obrazie oryginalnym i rozjaśnionym
    _, threshold_original = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    _, threshold_brightened = cv2.threshold(brightened_image, 100, 255, cv2.THRESH_BINARY)

    show_image("Original Threshold", threshold_original)
    show_image("Brightened Threshold", threshold_brightened)

    # Komentarze do analizy wpływu oświetlenia na progowanie

# Zadanie 5 - Progowanie metodą Otsu
def task_5(image):
    # a. Zastosuj progowanie metodą Otsu do rozjaśnionego obrazu
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightened_image = cv2.add(gray_image, 50)
    _, otsu_threshold = cv2.threshold(brightened_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    show_image("Otsu Threshold", otsu_threshold)

    # b. Porównaj z wynikami z poprzedniego zadania

# Zadanie 6 - Analiza histogramu
def task_6(image):
    # a. Wygeneruj histogram skali szarości
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # b. Zaznacz na nim wartość progową wyliczoną przez Otsu
    _, otsu_threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_value = _

    # c. Czy na histogramie można wyraźnie dostrzec dwa zbiory intensywności
    # Komentarze w kodzie

# Zadanie 7 - Maska i segmentacja
def task_7(image):
    # a. Zastosuj metodę Otsu i wykorzystaj uzyskaną maskę
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu_threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # b. Użyj maski do "wycięcia" obiektu z oryginalnego obrazu
    result = cv2.bitwise_and(image, image, mask=otsu_threshold)
    show_image("Segmented Object", result)

# Zadanie 8 - Case study – kostka brukowa
def task_8(image):
    # a. Na obrazie kostki brukowej zastosuj progowanie podstawowe z wybraną wartością T
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    # b. Nałóż progowanie Otsu
    _, otsu_threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # c. Porównaj, która metoda lepiej wykrywa wady powierzchni
    show_image("Basic Threshold", threshold_image)
    show_image("Otsu Threshold", otsu_threshold)
    # Komentarze do analizy

