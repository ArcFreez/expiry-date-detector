"""Detect unique letters in a list of strings."""
import cv2
import numpy as np
import os

def add_phrase_chars(_set, phrase):
    for c in phrase:
        _set.add(c)

def main():
    TEMPLATE_CHARS_DIR = 'template_charaters'
    # letters we need to look for based off of research
    date_numbers = [f'{i}' for i in range(0, 10)]
    food_exp_phrases = ['Best if Used By', 'Best if Used Before', 'Use-By']
    drug_exp_phrases = ['EXP.', 'EXP', 'EXPIRY', 'EXP DATE', 'Exp. Date']
    three_letrs_months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
        'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    # creating all strings based off the key words to look for
    # above
    all_strings = []
    all_strings.extend(food_exp_phrases)
    all_strings.extend(drug_exp_phrases)
    all_strings.extend(three_letrs_months)

    unique_chars_set = set()
    for phrase in all_strings:
        add_phrase_chars(unique_chars_set, phrase)
    # make a list of unique characters, and ignore special characters
    list_unique_chars = [c for c in unique_chars_set if c != ' ' and c != '.' and c != '-']
    list_unique_chars.sort()
    if not os.path.exists(TEMPLATE_CHARS_DIR):
        os.mkdir(TEMPLATE_CHARS_DIR)

    # all the letters we want in complex
    for char in list_unique_chars:
        template = np.zeros((50, 50), dtype=np.uint8)
        cv2.putText(template, text=char, org=(10, 35), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=1.5, color=(255, 255, 255), thickness=2)
        fname = f'{TEMPLATE_CHARS_DIR}/FONT_HERSHEY_TRIPLEX-lower-{char}.jpg' if char.islower() else\
            f'{TEMPLATE_CHARS_DIR}/FONT_HERSHEY_TRIPLEX-upper-{char}.jpg'
        cv2.imwrite(fname, template)
    
    # all the numbers we want in simplex
    for num in date_numbers:
        template = np.zeros((50, 50), dtype=np.uint8)
        cv2.putText(template, text=f"{num}", org=(10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5, color=(255, 255, 255), thickness=2)
        cv2.imwrite(f'{TEMPLATE_CHARS_DIR}/FONT_HERSHEY_SIMPLEX-{num}.jpg', template)


if __name__ == '__main__':
    main()
