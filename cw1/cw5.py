import cv2

def show_resizable_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Nie można wczytać obrazu.")
        return

    cv2.namedWindow('Obraz', cv2.WINDOW_NORMAL)
    cv2.imshow('Obraz', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show_resizable_image('images/profil1.jpg')