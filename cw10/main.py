from tasks import *
from util import load_image

PATH = 'images/hamster_call.png'
PATH_FAMOUS = 'images/hmmm.jpg'

if __name__ == "__main__":
    image = load_image(PATH)
    image_f = load_image(PATH_FAMOUS)
    if image is not None:
        task_1(image)
        task_2(image, image_f)
