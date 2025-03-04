import cv2

def show_two_images(image_path1, image_path2):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1 is None or image2 is None:
        print("Nie można wczytać jednego z obrazów.")
        return

    cv2.imshow('Obraz 1', image1)
    cv2.imshow('Obraz 2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show_two_images('images/profil1.jpg', 'images/profil1_grayscale.jpg')
