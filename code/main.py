"""The main program."""
import os
from glob import glob
import cv2
import numpy as np

def resize(image):
    def get_resized_dim(img_dim, max_dim):
        if img_dim > max_dim:
            s = max_dim / img_dim
            return s
        return img_dim
    MAX_WIDTH, MAX_HEIGHT = 1000, 1000
    img_w, img_h = image.shape[1], image.shape[0]
    return cv2.resize(image, dsize=None, fx=get_resized_dim(img_w, MAX_WIDTH),
        fy=get_resized_dim(img_h, MAX_HEIGHT))

def filter_black_labels(gray_img):
    thresh_img = cv2.adaptiveThreshold(
        src=gray_img,
        maxValue=255,  # output value where condition met
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # threshold_type
        blockSize=51,  # neighborhood size (a large odd number)
        C=10)
    # apply morhpological closing and opening
    kernel = np.ones((1, 1), np.uint8)
    morph_open = cv2.morphologyEx(thresh_img, 
        cv2.MORPH_OPEN, kernel)
    morph_close = cv2.morphologyEx(morph_open, 
        cv2.MORPH_CLOSE, kernel)
    _, cc_img = cv2.connectedComponents(morph_close)
    # apply connected components
    return cv2.normalize(
        src=cc_img,
        dst=None,
        alpha=0,
        beta=255, norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

def filter_white_labels(gray_img):
    thresh_img = cv2.adaptiveThreshold(
        src=gray_img,
        maxValue=255,  # output value where condition met
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,  # threshold_type
        blockSize=51,  # neighborhood size (a large odd number)
        C=-10)
    # apply morhpological closing and opening
    kernel = np.ones((1, 1), np.uint8)
    morph_open = cv2.morphologyEx(thresh_img, 
        cv2.MORPH_OPEN, kernel)
    morph_close = cv2.morphologyEx(morph_open, 
        cv2.MORPH_CLOSE, kernel)
    # apply connected components
    _, cc_img = cv2.connectedComponents(morph_close)
    return cv2.normalize(
        src=cc_img,
        dst=None,
        alpha=0,
        beta=255, norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

def main():
    INPUT_IMAGES_DIRECTORY = '../data'
    assert(os.path.exists(INPUT_IMAGES_DIRECTORY))
    image_file_names = glob(os.path.join(INPUT_IMAGES_DIRECTORY, '*.jpg'))
    assert(len(image_file_names) > 0)
    images = iter(cv2.imread(file_name) for file_name in image_file_names)

    for image in images:
        # first resize the image so that it is at most 1000 x 1000
        image_resized = resize(image.copy())
        # turn it into grayscale
        gray_img = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        # blur
        blur = cv2.GaussianBlur(
            src=gray_img,
            ksize=(0, 0),
            sigmaX=1, sigmaY=1
        )
        # threshold black
        black_labels = filter_black_labels(blur)
        white_labels = filter_white_labels(blur)
        cv2.imshow('black labels', black_labels)
        cv2.waitKey(0)
        cv2.imshow('white labels', white_labels)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()