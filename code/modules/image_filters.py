"""Module that contains all image filters."""
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

def apply_hsv(image, low_thresholds, high_thresholds):
    img_h, img_w = image.shape[0], image.shape[1]
    # Convert BGR to HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # split into the different bands.
    planes = cv2.split(hsv_img)
    # create output threshold image
    thresh_img = np.full((img_h, img_w), 255, dtype=np.uint8)
    for i in range(3):
        low_val = low_thresholds[i]
        high_val = high_thresholds[i]

        _, low_img = cv2.threshold(planes[i], low_val, 255, cv2.THRESH_BINARY)
        _, high_img = cv2.threshold(planes[i], high_val, 255, cv2.THRESH_BINARY_INV)

        thresh_band_img = cv2.bitwise_and(low_img, high_img)
        
        # AND with output threshold image.
        thresh_img = cv2.bitwise_and(thresh_img, thresh_band_img)
    return thresh_img
