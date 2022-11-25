"""The main program."""
import os
from glob import glob
import cv2
import numpy as np

from modules import resize

def main():
    INPUT_IMAGES_DIRECTORY = '../data'
    assert(os.path.exists(INPUT_IMAGES_DIRECTORY))
    image_file_names = glob(os.path.join(INPUT_IMAGES_DIRECTORY, '*.jpg'))
    assert(len(image_file_names) > 0)
    images = iter(cv2.imread(file_name) for file_name in image_file_names)

    for image in images:
        img_cpy = image.copy()
        resized = resize(img_cpy)
        cv2.imshow('result', resized)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()