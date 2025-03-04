import cv2

def load_and_display_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Nie można wczytać obrazu. Sprawdź ścieżkę.")

        cv2.imshow('Obraz', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Błąd: {e}")

load_and_display_image('images/profil1.jpg')
