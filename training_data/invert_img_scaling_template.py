"""Inverts the image files"""
import cv2
from glob import glob
import os
import numpy as np

def resize(image):
    """Resize image 100 x 100."""
    def get_resized_dim(img_dim, max_dim):
        if img_dim > max_dim:
            s = max_dim / img_dim
            return s
        return img_dim
    MAX_WIDTH, MAX_HEIGHT = 100, 100
    img_w, img_h = image.shape[1], image.shape[0]
    return cv2.resize(image, dsize=(100, 100), fx=get_resized_dim(img_w, MAX_WIDTH),
        fy=get_resized_dim(img_h, MAX_HEIGHT))

def main():
    dir = "./dot-mat-sq-chars"
    w_dir = "./dot-mat-sq-chars-inv"
    assert(os.path.exists(dir))
    assert(os.path.exists(w_dir))
    img_file_names = glob(os.path.join(dir, '*.jpg'))
    assert(len(img_file_names) > 0)
    for fname in img_file_names:
        img = cv2.imread(fname)
        img = cv2.resize(img, (100, 100), interpolation = cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        fname_wo_path = fname.split('/')[2]
        cv2.imwrite(f'{w_dir}/{fname_wo_path}',
         img)

if __name__ == '__main__':
    main()