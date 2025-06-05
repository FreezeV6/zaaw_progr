# Repo na zaawansowane programowanie w języku Python

# Automatic License Plate Recognition

## Główne rozwiązanie
- Wykrywanie tablicy rejestracyjnej: **yolo11**
- Odczyt tablicy: **TesseractOCR**
- Metryki: **IoU, dokładność OCR, scoring według wytycznych**

## Przygotowanie
1. Umieść dane w katalogu `data/` zgodnie ze strukturą.
2. Uruchom `convert_annotations.py` – wygeneruje pliki YOLO i ground-truth CSV.
3. Podziel dane na train/val/test (opcjonalnie).
4. Wytrenuj model YOLO.
5. Skonfiguruj ścieżki w `config.py`.

## Uruchomienie
1. Umieść zdjęcia i adnotacje w folderze `dataset/`
2. `pip install -r requirements.txt`
3. `python src/main.py`
4. Wyniki i debug: w konsoli oraz w folderze debug_imgs/

## Ewaluacja
```bash
python main.py
```

## Autor
FreezeV6




