import cv2

def save_gray_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Nie można wczytać obrazu.")
        return
    cv2.imwrite(output_path, image)
    print(f"Obraz zapisano jako {output_path}")

save_gray_image('images/profil1.jpg', 'images/profil1_grayscale.jpg')
