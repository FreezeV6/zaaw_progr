from tasks import *
from util import load_image

if __name__ == "__main__":
    PATH = 'images/hmmm.jpg'
    image = load_image(PATH)
    task_1(image)
    task_2(image)
    task_3(image)