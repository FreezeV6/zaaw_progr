import cv2

def count_channels(image_path, color=True):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(image_path, flag)
    if image is None:
        print("Nie można wczytać obrazu.")
        return

    channels = 1 if len(image.shape) == 2 else image.shape[2]
    print(f"Liczba kanałów: {channels}")

count_channels('images/profil1.jpg', color=True)
count_channels('images/profil1.jpg', color=False)
