from tasks import *
from util import load_image

PATH = 'images/mango_suszone.jpg'
PATH_FAMOUS = 'images/hamster_call.png'

if __name__ == "__main__":
    image = load_image(PATH)
    image_f = load_image(PATH_FAMOUS)
    # task_1(image)
    # task_2()
    # task_3()
    # task_4()
    # task_5()
    task_6(image_f)