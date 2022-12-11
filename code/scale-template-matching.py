import cv2
import numpy as np
import os
from glob import glob
import re
from datetime import datetime

# functions to clean up image
def resize(image, MAX_WIDTH=1000, MAX_HEIGHT=1000):
    """Resize image to at most 1000 x 1000."""
    def get_resized_dim(img_dim, max_dim):
        if img_dim > max_dim:
            s = max_dim / img_dim
            return s
        return img_dim
    img_w, img_h = image.shape[1], image.shape[0]
    dim = (int(get_resized_dim(img_w, MAX_WIDTH)), int(get_resized_dim(img_w, MAX_HEIGHT)))
    return cv2.resize(image, dsize=dim, fx=get_resized_dim(img_w, MAX_WIDTH),
        fy=get_resized_dim(img_h, MAX_HEIGHT))


def scale_and_match_template(original, image, template, template_name, color_of_label, show_matches=False):
    template_w, template_h = template.shape[::-1]
    global_max_correlation = 0
    max_loc = None
    max_img = None
    max_scale = 1
    descending_scale_from_100_precent_to_20_percent = np.linspace(0.2, 1.0, 20)[::-1]
    for scale in descending_scale_from_100_precent_to_20_percent:
        resized = resize(image, MAX_WIDTH=int(image.shape[1] * scale), 
            MAX_HEIGHT=int(image.shape[1] * scale))
        if resized.shape[0] < template_w or resized.shape[1] < template_h:
            break
        res = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        loc = np.where(res == max_val)
        if max_val > global_max_correlation:
            global_max_correlation = max_val
            max_loc = loc
            max_img = resize(original, MAX_WIDTH=int(original.shape[1] * scale), 
                MAX_HEIGHT=int(original.shape[1] * scale))
            max_scale = scale
    if max_loc is not None:
        if show_matches:
            for pt in zip(*max_loc[::-1]):
                cv2.rectangle(max_img, pt, (pt[0] + template_w, pt[1] + template_h), (255,255,255), 1)
            cv2.imshow(f'matches for {color_of_label} image with {template_name}', max_img)
            cv2.waitKey(0)
        return [(template_name, global_max_correlation, pt, max_scale) for pt in [pt for pt in zip(*max_loc[::-1])]]
    return []


def determine_if_expired(date_str):
    # mocking date for now, but this can be adjusted
    curr_date = datetime(year=2024, month=12, day=4)
    month_to_num_table = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5,
             'JUN': 6, 'JUL': 7,
            'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    PRODUCT_HAS_EXPIRED_MSG = 'CAUTION: PRODUCT HAS EXPIRED'
    def handle_mm_dd_yyyy_fmt(delimeter):
        split = date_str.split(delimeter)
        if len(split) == 3:
            month, day, year = split[0], \
                split[1], split[2]
            if month.isdigit() and day.isdigit() and year.isdigit():
                date = datetime(year=int(year), month=int(month), day=int(day))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    def handle_yyyy_mm_fmt(delimeter):
        split = date_str.split(delimeter)
        if len(split) == 2:
            year, month = split[0], split[1]
            if year.isdigit() and month.isdigit():
                date = datetime(year=int(year), month=int(month))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    def handle_yyyy_mmm_dd_fmt(delimeter):
        split = date_str.split(delimeter)
        if len(split) == 3:
            year, month, day = split[0], split[1],\
                split[2]
            if year.isdigit() and day.isdigit() and month in month_to_num_table:
                date = datetime(year=int(year), month=month_to_num_table[month],
                        day=int(day))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    def handle_yyyy_mmm_fmt(delimeter):
        split = date_str.split(delimeter)

        if len(split) == 2:
            year, month = split[0], split[1]
            if year.isdigit() and month in month_to_num_table:
                date = datetime(year=int(year), month=month_to_num_table[month])
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    def handle_yyyy_mmm_dd_spaces_fmt():
        if len(date_str) == 9:
            year, month, day = date_str[0:4], date_str[4:7],\
                date_str[7:9]
            if year.isdigit() and day.isdigit() and month in month_to_num_table:
                date = datetime(year=int(year), month=month_to_num_table[month], day=int(day))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    def handle_yyyy_mmm_spaces_fmt():
        if len(date_str) == 7:
            year, month = date_str[0:4], date_str[4:7]
            if year.isdigit() and month in month_to_num_table:
                date = datetime(year=int(year), month=month_to_num_table[month])
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    def handle_dd_mmm_yyyy_spaces_fmt():
        if len(date_str) == 9:
            day, month, year = date_str[0:2], date_str[2:5],\
                date_str[5:9]
            if day.isdigit() and year.isdigit() and month in month_to_num_table:
                date = datetime(year=int(year), month=month_to_num_table[month], day=int(day))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    def handle_mmm_dd_yy_spaces_fmt():
        if len(date_str) == 7:
            month, day, year = date_str[0:3],\
                date_str[3:5],\
                date_str[5:7]
            if day.isdigit() and year.isdigit() and month in month_to_num_table:
                curr_year_in_2_digits = abs(curr_date.year) % 100
                if curr_year_in_2_digits > year:
                    return PRODUCT_HAS_EXPIRED_MSG
                date = datetime(year=curr_date.year, month=month_to_num_table[month], day=int(day))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    # match mm/dd/yyyy, mm-dd-yyyy format
    format_handled = [handle_mm_dd_yyyy_fmt('/'),
        handle_mm_dd_yyyy_fmt('-'), handle_yyyy_mm_fmt('-'),
        handle_yyyy_mmm_dd_fmt('-'),
        handle_yyyy_mmm_fmt('-'),
        handle_yyyy_mmm_dd_spaces_fmt(),
        handle_yyyy_mmm_spaces_fmt(),
        handle_dd_mmm_yyyy_spaces_fmt(),
        handle_mmm_dd_yy_spaces_fmt()]
    for msg in format_handled:
        if msg is not None:
            return msg
    return 'PRODUCT IS SAFE TO CONSUME.'


def match_best_date_format(output_label):
    date_fmt_regexs_and_matches = [
        # labels recommended format by FDA for drug labels
        ('mm/dd/yyyy',re.compile(r'\d{2}\\\d{2}\\\d{4}')),
        ('mm-dd-yyyy', re.compile(r'\d{2}\-\d{2}\-\d{4}')),
        ('yyyy-mm', re.compile(r'\d{4}\-\d{2}'), []),
        ('yyyy-mmm-dd', re.compile(r"""\d{4}\-JAN\-\d{2}|
        \d{4}\-FEB\-\d{2}|
        \d{4}\-MAR\-\d{2}|
        \d{4}\-APR\-\d{2}|
        \d{4}\-MAY\-\d{2}|
        \d{4}\-JUN\-\d{2}|
        \d{4}\-JUL\-\d{2}|
        \d{4}\-AUG\-\d{2}|
        \d{4}\-SEP\-\d{2}|
        \d{4}\-OCT\-\d{2}|
        \d{4}\-NOV\-\d{2}|
        \d{4}\-DEC\-\d{2}""")),
        ('yyyy-mmm', re.compile(r"""\d{4}\-JAN|
        \d{4}\-FEB|
        \d{4}\-MAR|
        \d{4}\-APR|
        \d{4}\-MAY|
        \d{4}\-JUN|
        \d{4}\-JUL|
        \d{4}\-AUG|
        \d{4}\-SEP|
        \d{4}\-OCT|
        \d{4}\-NOV|
        \d{4}\-DEC""")),
        ('yyyy mmm dd', re.compile(r"""\d{4}JAN\d{2}|
        \d{4}FEB\d{2}|
        \d{4}MAR\d{2}|
        \d{4}APR\d{2}|
        \d{4}MAY\d{2}|
        \d{4}JUN\d{2}|
        \d{4}JUL\d{2}|
        \d{4}AUG\d{2}|
        \d{4}SEP\d{2}|
        \d{4}OCT\d{2}|
        \d{4}NOV\d{2}|
        \d{4}DEC\d{2}"""), []),
        ('yyyy mmm', re.compile(r"""\d{4}JAN|
        \d{4}FEB|
        \d{4}MAR|
        \d{4}APR|
        \d{4}MAY|
        \d{4}JUN|
        \d{4}JUL|
        \d{4}AUG|
        \d{4}SEP|
        \d{4}OCT|
        \d{4}NOV|
        \d{4}DEC""")),
        # observed regexs from food labels
        ('dd mmm yyyy', re.compile(r"""\d{2}JAN\d{4}|
        \d{2}FEB\d{4}|
        \d{2}MAR\d{4}|
        \d{2}APR\d{4}|
        \d{2}MAY\d{4}|
        \d{2}JUN\d{4}|
        \d{2}JUL\d{4}|
        \d{2}AUG\d{4}|
        \d{2}SEP\d{4}|
        \d{2}OCT\d{4}|
        \d{2}NOV\d{4}|
        \d{2}DEC\d{4}""")),
        ('mmm dd yy', re.compile(r"""JAN\d{2}\d{2}|
        FEB\d{2}\d{2}|
        MAR\d{2}\d{2}|
        APR\d{2}\d{2}|
        MAY\d{2}\d{2}|
        JUN\d{2}\d{2}|
        JUL\d{2}\d{2}|
        AUG\d{2}\d{2}|
        SEP\d{2}\d{2}|
        OCT\d{2}\d{2}|
        NOV\d{2}\d{2}|
        DEC\d{2}\d{2}"""))
    ]
    date_matches = []
    for item in date_fmt_regexs_and_matches:
        regex_to_match = item[1]
        matches = regex_to_match.findall(output_label)
        if len(matches) != 0:
            date_matches.extend(matches)
    return date_matches


def get_black_and_white_labels(image):
    # 1) first resize the image so that it is at most 1000 x 1000
    image_resized = resize(image.copy())
    # 2) turn it into grayscale
    gray_img = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    # 3) apply gaussian blur to get rid of background noise
    gray_img = cv2.GaussianBlur(
        src=gray_img,
        ksize=(0, 0), # sobel
        sigmaX=1
    )
    # 4) use adaptive thresholding to get rid of lighting issues
    # and threshold both black and white labels
    black_labels = cv2.adaptiveThreshold(
        src=gray_img,
        maxValue=255,  # output value where condition met
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # threshold_type
        blockSize=51,  # neighborhood size (a large odd number)
        C=10)
    white_labels = cv2.adaptiveThreshold(
        src=gray_img,
        maxValue=255,  # output value where condition met
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,  # threshold_type
        blockSize=51,  # neighborhood size (a large odd number)
        C=-10)
    cv2.imshow('black labels', black_labels)
    cv2.waitKey(0)
    cv2.imshow('white labels', white_labels)
    cv2.waitKey(0)
    return image_resized, black_labels,\
        white_labels

def get_template_match_scores_and_locs(image_resized, gray_img, templates_to_match, color_of_label):
    score_and_locs_arr = []
    for template_content in templates_to_match:
        template_img, template_label = template_content
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        score_and_locs_for_character = scale_and_match_template(image_resized, 
            gray_img, template_img, template_label, color_of_label=color_of_label,
            show_matches=False)
        score_and_locs_arr.extend(score_and_locs_for_character)
    # 6) get only the best matches for overlapping points
    # w and h are constant due to all templates being same size.
    def are_overlapping_rect_points(pt1, pt2, w=100, h=100):
        pt3 = (pt1[0] + w, pt1[1] + h)
        return (pt2[0] >= pt1[0] and pt2[0] <= pt3[0])\
            and (pt2[1] >= pt1[1] and pt2[1] <= pt3[1])
    def get_score(item):
        return item[1]
    def get_pt(item):
        return item[2]
    def get_scale(item):
        return item[-1]
    # sort by points to see if
    # a area has been labeled more than once.
    score_and_locs_arr.sort(key=lambda item: get_pt(item)[0])
    score_and_locs_arr.sort(key=lambda item: get_pt(item)[1])
        
    max_score_and_locs_arr = []
    # compare matches to see if they are on the same
    # rectangle for the template, and if they are pick the better match
    for i in range(1,len(score_and_locs_arr)):
        curr = i
        prev = i - 1
        while i < len(score_and_locs_arr) and\
            get_scale(score_and_locs_arr[prev]) == get_scale(score_and_locs_arr[curr]) and\
            are_overlapping_rect_points(get_pt(score_and_locs_arr[prev]),
                get_pt(score_and_locs_arr[curr])):
            if get_score(score_and_locs_arr[prev]) < get_score(score_and_locs_arr[curr]): 
                prev = curr
            i += 1
            curr = i
        max_score_and_locs_arr.append(score_and_locs_arr[prev])
    return max_score_and_locs_arr

def detect_expiry_date(image, image_name, log_file, templates_to_match):
    print(f'detecting expiry date for {image_name}')
    log_file.write(f'detecting expiry date for {image_name}\n')
    # 1)process the image so that it is possible to detect
    image_resized, black_labels, white_labels = get_black_and_white_labels(image)
    # 2) expiry labels can be balck or white, so find the matches
    # on the scaled image for the template which doesn't scale.
    print('MATCHING TEMPLATES...')
    max_score_and_locs_black_labels = get_template_match_scores_and_locs(image_resized, black_labels, templates_to_match, 'black')
    max_score_and_locs_white_labels = get_template_match_scores_and_locs(image_resized, white_labels, templates_to_match, 'white')
    print('DONE MATCHING TEMPLATES')
    def get_label(item):
        return item[0]
    def output_date_formt_matches(date_format_matches):
        for date_format_match in date_format_matches:
            log_file.write(f'{date_format_match}\n')
    # 3) figure out what date the templates spell out based off of sorting points in
    # the image.
    output_label_black = ''.join(get_label(item) for item in max_score_and_locs_black_labels)
    log_file.write(f'Ouput black label based off of matches : {output_label_black}\n')
    output_label_white = ''.join(get_label(item) for item in max_score_and_locs_white_labels)
    log_file.write(f'Ouput white label based off of matches : {output_label_white}\n')
    date_format_matches_black_label = match_best_date_format(output_label_black)
    date_format_matches_white_label = match_best_date_format(output_label_white)
    if len(date_format_matches_black_label) == 1:
        print('labels are in black')
        log_file.write('labels are in black.\n')
        log_file.write(f'for {image_name}: found exact expiry date {date_format_matches_black_label[0]}\n')
        has_exp_msg = determine_if_expired(date_format_matches_black_label[0])
        print(has_exp_msg)
        log_file.write(f'{has_exp_msg}\n')
    elif len(date_format_matches_white_label) == 1:
        print('labels are in white')
        log_file.write('labels are in white.')
        log_file.write(f'for {image_name}: found exact expiry date {date_format_matches_white_label[0]}\n')
        has_exp_msg = determine_if_expired(date_format_matches_white_label[0])
        print(has_exp_msg)
        log_file.write(f'{has_exp_msg}\n')
    elif len(date_format_matches_black_label) == 0 and len(date_format_matches_white_label) == 0:
        print('no matches found for image to show date.')
        log_file.write(f'no matches found for image {image_name}\n')
    else:
        print('ambigious matches for both labels.')
        log_file.write(f'ambigious matches for both labels.\n')
        log_file.write('white label matches\n')
        output_date_formt_matches(date_format_matches_white_label)
        log_file.write('black label matches\n')
        output_date_formt_matches(date_format_matches_black_label)

def get_templates():
    CHARS_TO_MATCH_DIR = "../training_data/dot-mat-sq-chars-inv"
    assert(os.path.exists(CHARS_TO_MATCH_DIR))
    def get_character(fname: str) -> str:
        last_split = fname.split('-')[-1]
        char_name = last_split.split('.')[0]
        if char_name == 'slash':
            return '/'
        if char_name == 'dash':
            return '-'
        return char_name
    
    character_file_names = glob(os.path.join(CHARS_TO_MATCH_DIR, '*.jpg'))
    templates_to_match = [(cv2.imread(fname),
        get_character(fname)) for fname in character_file_names]
    assert(len(templates_to_match) > 0)
    return templates_to_match

def main():
    # get test data
    TEST_DATA_IMAGE_DIR, TRAINING_DATA_IMAGE_DIR = '../data', '../training_data/dot-mat-sq-chars-inv'
    assert(os.path.exists(TEST_DATA_IMAGE_DIR))
    image_file_names = glob(os.path.join(TEST_DATA_IMAGE_DIR, '*.jpg'))
    assert(len(image_file_names) > 0)
    image_file_names.sort()
    # training data image directory
    assert(os.path.exists(TRAINING_DATA_IMAGE_DIR))
    training_data_file_names = glob(os.path.join(TRAINING_DATA_IMAGE_DIR, '*.jpg'))
    assert(len(training_data_file_names) > 0)
    training_data_file_names.sort()
    templates_to_match = get_templates()
    # log file
    log_file = open('logs-scale-template.txt', 'w+')
    for fname in image_file_names:
        image = cv2.imread(fname)
        detect_expiry_date(image, fname, log_file, templates_to_match)
    log_file.close()

if __name__ == '__main__':
    main()