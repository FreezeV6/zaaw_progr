from tasks import *
from util import load_image

PATH = 'images/img.png'
image = load_image(PATH)

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)

    # Task 1: Klasyczne progowanie
    print('\n-- Task 1: Klasyczne progowanie --')
    task1(PATH, thresholds=[100, 140, 180], output_dir='results')

    # Task 2: Eksperyment z cv2.findContours
    print('\n-- Task 2: Eksperyment z cv2.findContours --')
    task2(PATH, threshold_value=140, output_dir='results')

    # Task 3: Eksperyment z rozdzielczością obrazu
    print('\n-- Task 3: Eksperyment z rozdzielczością obrazu --')
    task3(PATH)

    # Task 4: Numeryzacja i segmentacja kostek
    print('\n-- Task 4: Numeryzacja i segmentacja kostek --')
    contours = task4(PATH, output_dir='results')

    # Task 5: Pomiar wymiarów kostek
    print('\n-- Task 5: Pomiar wymiarów kostek --')
    task5(PATH, output_dir='results', contours=contours)

    # Task 6: Filtrowanie konturów po wielkości
    print('\n-- Task 6: Filtrowanie konturów po wielkości --')
    filtered = task6(PATH, min_area=500, max_area=5000, output_dir='results')

    # Task 7: Liczenie i raportowanie kostek
    print('\n-- Task 7: Liczenie i raportowanie kostek --')
    task7(filtered)

    # Inspection: wykrywanie defektów na każdej kostce
    print('\n-- Inspection: wykrywanie defektów na każdej kostce --')
    task_inspect(PATH, contours, output_dir='results')
