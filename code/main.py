"""The main program."""
import os
from glob import glob
import cv2
import numpy as np

# functions to clean up image
def resize(image):
    """Resize image to at most 1000 x 1000."""
    def get_resized_dim(img_dim, max_dim):
        if img_dim > max_dim:
            s = max_dim / img_dim
            return s
        return img_dim
    MAX_WIDTH, MAX_HEIGHT = 1000, 1000
    img_w, img_h = image.shape[1], image.shape[0]
    return cv2.resize(image, dsize=None, fx=get_resized_dim(img_w, MAX_WIDTH),
        fy=get_resized_dim(img_h, MAX_HEIGHT))

def get_black_and_white_labels(gray_img):
    """Return the processed image containing only white and only black labels."""
    thresh_black = cv2.adaptiveThreshold(
        src=gray_img,
        maxValue=255,  # output value where condition met
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # threshold_type
        blockSize=51,  # neighborhood size (a large odd number)
        C=10)
    thresh_white = cv2.adaptiveThreshold(
        src=gray_img,
        maxValue=255,  # output value where condition met
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,  # threshold_type
        blockSize=51,  # neighborhood size (a large odd number)
        C=-10)
    def get_connected_components(thresh_img):
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
    return get_connected_components(thresh_black), get_connected_components(thresh_white)

# functions to get data
def get_test_data():
    """Get all test data."""
    INPUT_IMAGES_DIRECTORY = '../data'
    assert(os.path.exists(INPUT_IMAGES_DIRECTORY))
    image_file_names = glob(os.path.join(INPUT_IMAGES_DIRECTORY, '*.jpg'))
    assert(len(image_file_names) > 0)
    return iter(cv2.imread(file_name) for file_name in image_file_names)

def get_training_data():
    """Get all training data for ORB detection."""
    TRAINIG_DATA_DIRS = ('../training_data/phrases', '../training_data/months', '../training_data/numbers')
    for dir in TRAINIG_DATA_DIRS:
        assert(os.path.exists(dir))
    EXP_PHRASES_DIR, MONTHS_DIR, NUMBERS_DIR = TRAINIG_DATA_DIRS
    def get_images(dir):
        image_file_names = glob(os.path.join(dir, '*.jpg'))
        assert(len(image_file_names) > 0)
        return [cv2.imread(file_name) for file_name in image_file_names]
    return get_images(EXP_PHRASES_DIR), get_images(MONTHS_DIR), get_images(NUMBERS_DIR)

# functions to detect image
def detect_expiry_date(image, training_imgs):
    # 1) process the image so that it is possible to detect
    # first resize the image so that it is at most 1000 x 1000
    image_resized = resize(image.copy())
    # turn it into grayscale
    gray_img = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    # blur to get rid of background noise
    blur = cv2.GaussianBlur(
        src=gray_img,
        ksize=(0, 0), # sobel
        sigmaX=1
    )
    # threshold black and white
    black_labels, white_labels = get_black_and_white_labels(blur)
    # 2) match the processed image
    cv2.imshow('black labels', black_labels)
    cv2.waitKey(0)
    cv2.imshow('white labels', white_labels)
    cv2.waitKey(0)

def main():
    images = get_test_data()
    training_imgs = get_training_data()
    for image in images:
        detect_expiry_date(image, training_imgs)

if __name__ == '__main__':
    main()