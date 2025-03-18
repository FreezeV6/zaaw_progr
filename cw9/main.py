from tasks import *
from util import load_image

PATH = 'images/mango_suszone.jpg'
PATH_FAMOUS = 'images/mango_suszone_diff.jpg'

if __name__ == "__main__":
    image = load_image(PATH)
    image_f = load_image(PATH_FAMOUS)
    if image is not None:
        task_1(image)
        task_2(image)
        task_3(image)
        task_4(image)
        if image_f is not None:
            task_5(image, image_f)
        else:
            print("Error: Unable to load second image.")
    else:
        print("Error: Unable to load image.")