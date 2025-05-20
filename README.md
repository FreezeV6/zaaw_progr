# Repo na zaawansowane programowanie w języku Python

# Automatic License Plate Recognition

## Główne rozwiązanie
- Wykrywanie tablicy rejestracyjnej: **ResNet18 (transfer learning), regresja współrzędnych YOLO**
- Odczyt tablicy: **EasyOCR**
- Metryki: **IoU, dokładność OCR, scoring według wytycznych**
- Wizualizacja predykcji: debug_imgs/ z zielonym (pred) i czerwonym (ground truth) boxem

## Uruchomienie
1. Umieść zdjęcia i adnotacje w folderze `dataset/`
2. `pip install -r requirements.txt`
3. `python src/main.py`
4. Wyniki i debug: w konsoli oraz w folderze debug_imgs/

## Autor
FreezeV6
