"""Generate training images for detecting using ORB."""
import cv2
import numpy as np
import os

def main():
    DIRS_TO_MAKE = ('phrases', 'numbers', 'months')
    for dir in DIRS_TO_MAKE:
        if not os.path.exists(dir):
            os.mkdir(dir)
    PHRASES_DIR, NUMBERS_DIR, MONTHS_DIR = DIRS_TO_MAKE

    date_numbers = [f'{i}' for i in range(0, 10)]
    for num in date_numbers:
        template = np.zeros((50, 50), dtype=np.uint8)
        cv2.putText(template, text=f"{num}", org=(10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5, color=(255, 255, 255), thickness=2)
        cv2.imwrite(f'{NUMBERS_DIR}/FONT_HERSHEY_SIMPLEX-{num}.jpg', template)

    food_exp_phrases = ['Best if Used By', 'Best if Used Before', 'Use-By']
    drug_exp_phrases = ['EXP.', 'EXP', 'EXPIRY', 'EXP DATE', 'Exp. Date']
    all_phrases = []
    all_phrases.extend(food_exp_phrases)
    all_phrases.extend(drug_exp_phrases)
    for i, phrase in enumerate(all_phrases):
        template = np.zeros((50, 550), dtype=np.uint8)
        # triplex since that will be the closest matching font for expiry labels.
        cv2.putText(template, text=phrase, org=(10, 35), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=1.5, color=(255, 255, 255), thickness=2)
        fname = f'{PHRASES_DIR}/FONT_HERSHEY_TRIPLEX-TEXT-{i}-phrase.jpg'
        cv2.imwrite(fname, template)

    three_letrs_months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
        'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for month in three_letrs_months:
        template = np.zeros((50, 120), dtype=np.uint8)
        # triplex since that will be the closest matching font for expiry labels.
        cv2.putText(template, text=month, org=(10, 35), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=1.5, color=(255, 255, 255), thickness=2)
        fname = f'{MONTHS_DIR}/FONT_HERSHEY_TRIPLEX-TEXT-{month}.jpg'
        cv2.imwrite(fname, template)

if __name__ == '__main__':
    main()
