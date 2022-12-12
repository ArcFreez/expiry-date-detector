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

def with_parse_err_handler(parser):
    def wrapper(*args,**kwargs):
        try:
            return parser(*args,**kwargs)
        except ValueError as e:
            print(e)
        return None
    return wrapper

def determine_if_expired(date_str):
    curr_date = datetime(year=2024, month=12, day=1)
    month_to_num_table = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5,
             'JUN': 6, 'JUL': 7,
            'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    PRODUCT_HAS_EXPIRED_MSG = 'CAUTION: PRODUCT HAS EXPIRED'
    @with_parse_err_handler
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
    
    @with_parse_err_handler
    def handle_yyyy_mm_fmt(delimeter):
        split = date_str.split(delimeter)
        if len(split) == 2:
            year, month = split[0], split[1]
            if year.isdigit() and month.isdigit():
                date = datetime(year=int(year), month=int(month))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    
    @with_parse_err_handler
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

    @with_parse_err_handler
    def handle_yyyy_mmm_fmt(delimeter):
        split = date_str.split(delimeter)

        if len(split) == 2:
            year, month = split[0], split[1]
            if year.isdigit() and month in month_to_num_table:
                date = datetime(year=int(year), month=month_to_num_table[month])
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    
    @with_parse_err_handler
    def handle_yyyy_mmm_dd_spaces_fmt():
        if len(date_str) == 9:
            year, month, day = date_str[0:4], date_str[4:7],\
                date_str[7:9]
            if year.isdigit() and day.isdigit() and month in month_to_num_table:
                date = datetime(year=int(year), month=month_to_num_table[month], day=int(day))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None
    
    @with_parse_err_handler
    def handle_yyyy_mmm_spaces_fmt():
        if len(date_str) == 7:
            year, month = date_str[0:4], date_str[4:7]
            if year.isdigit() and month in month_to_num_table:
                date = datetime(year=int(year), month=month_to_num_table[month])
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None

    @with_parse_err_handler
    def handle_dd_mmm_yyyy_spaces_fmt():
        if len(date_str) == 9:
            day, month, year = date_str[0:2], date_str[2:5],\
                date_str[5:9]
            if day.isdigit() and year.isdigit() and month in month_to_num_table:
                date = datetime(year=int(year), month=month_to_num_table[month], day=int(day))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None

    @with_parse_err_handler
    def handle_mmm_dd_yy_spaces_fmt():
        if len(date_str) == 7:
            month, day, year = date_str[0:3],\
                date_str[3:5],\
                date_str[5:7]
            if day.isdigit() and year.isdigit() and month in month_to_num_table:
                curr_year_in_2_digits = abs(curr_date.year) % 100
                if curr_year_in_2_digits > year:
                    return PRODUCT_HAS_EXPIRED_MSG
                if curr_year_in_2_digits < year:
                    return None
                date = datetime(year=curr_date.year, month=month_to_num_table[month], day=int(day))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None

    @with_parse_err_handler
    def handle_mm_dd_yy_spaces_fmt():
        if len(date_str) == 6:
            month, day, year = date_str[0:2], \
                date_str[2:4],\
                date_str[4:6]
            if month.isdigit() and year.isdigit() and day.isdigit():
                curr_year_in_2_digits = abs(curr_date.year) % 100
                if curr_year_in_2_digits > year:
                    return PRODUCT_HAS_EXPIRED_MSG
                if curr_year_in_2_digits < year:
                    return None
                date = datetime(year=curr_date.year, month=int(month), day=int(day))
                if curr_date > date:
                    return PRODUCT_HAS_EXPIRED_MSG
        return None

    @with_parse_err_handler
    def handle_mm_dd_yy_slash_fmt():
        split = date_str.split('/')
        if len(split) == 3:
            month, day, year = split[0], split[1], split[2]
            if all(len(item) == 2 for item in split):
                if month.isdigit() and year.isdigit() and day.isdigit():
                    curr_year_in_2_digits = abs(curr_date.year) % 10
                    if curr_year_in_2_digits > year:
                        return PRODUCT_HAS_EXPIRED_MSG
                    if curr_year_in_2_digits < year:
                        return None
                    date = datetime(year=curr_date.year, month=int(month), day=int(day))
                    if curr_date > date:
                        return PRODUCT_HAS_EXPIRED_MSG
        return None

    # match mm/dd/yyyy, mm-dd-yyyy format
    format_handled = [
        handle_mm_dd_yyyy_fmt('/'),
        handle_mm_dd_yyyy_fmt('-'),
        handle_dd_mmm_yyyy_spaces_fmt(),
        handle_yyyy_mmm_dd_fmt('-'),
        handle_yyyy_mmm_dd_fmt('/'),
        handle_yyyy_mmm_dd_spaces_fmt(),
        handle_mmm_dd_yy_spaces_fmt(),
        handle_yyyy_mmm_fmt('-'),
        handle_yyyy_mmm_spaces_fmt(),
        handle_mmm_dd_yy_spaces_fmt(),
        handle_mm_dd_yy_spaces_fmt(),
        handle_mm_dd_yy_slash_fmt(),
        handle_yyyy_mm_fmt('/'),
        handle_yyyy_mm_fmt('-'),
    ]
    for msg in format_handled:
        if msg is not None:
            return msg
    return 'PRODUCT IS SAFE TO CONSUME.'


def match_best_date_format(output_label):
    date_fmt_regexs_and_matches = [
        # labels recommended format by FDA for drug labels
        ('mm/dd/yyyy',re.compile(r'\d{2}\/\d{2}\/\d{4}')),
        ('mm-dd-yyyy', re.compile(r'\d{2}\-\d{2}\-\d{4}')),
        ('yyyy-mm', re.compile(r'\d{4}\-\d{2}')),
        ('yyyy/mm', re.compile(r'\d{4}\\\d{2}')),
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
        ('yyyy/mmm/dd', re.compile(r"""\d{4}\/JAN\/\d{2}|
        \d{4}\/FEB\/\d{2}|
        \d{4}\/MAR\/\d{2}|
        \d{4}\/APR\/\d{2}|
        \d{4}\/MAY\/\d{2}|
        \d{4}\/JUN\/\d{2}|
        \d{4}\/JUL\/\d{2}|
        \d{4}\/AUG\/\d{2}|
        \d{4}\/SEP\/\d{2}|
        \d{4}\/OCT\/\d{2}|
        \d{4}\/NOV\/\d{2}|
        \d{4}\/DEC\/\d{2}""")),
        ('yyyy/mmm', re.compile(r"""\d{4}\/JAN|
        \d{4}\/FEB|
        \d{4}\/MAR|
        \d{4}\/APR|
        \d{4}\/MAY|
        \d{4}\/JUN|
        \d{4}\/JUL|
        \d{4}\/AUG|
        \d{4}\/SEP|
        \d{4}\/OCT|
        \d{4}\/NOV|
        \d{4}\/DEC""")),
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
        # observed 
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
        DEC\d{2}\d{2}""")),
        ('mm/dd/yy', re.compile(r'\d{2}\/d{2}\/d{2}')),
        ('mm dd yy', re.compile(r''))
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
    return image_resized, black_labels,\
        white_labels

def get_possible_date_CCs(binary_img):
    ksize = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    binary_img = cv2.dilate(binary_img, kernel)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary_img)
    binary_img = cv2.normalize(
        src= binary_img, dst= None, alpha = 0, beta = 255,
        norm_type= cv2.NORM_MINMAX, dtype= cv2.CV_8U)
    height_to_cc_table = {}
    # grab connected components with equal heights
    # one of these components can have multiple or one
    # template, and could possibly represent date
    for n, item in enumerate(zip(stats, centroids)):
        stat, centroid = item
        x0 = stat[cv2.CC_STAT_LEFT]
        y0 = stat[cv2.CC_STAT_TOP]
        w = stat[cv2.CC_STAT_WIDTH]
        h = stat[cv2.CC_STAT_HEIGHT]
        area = stats[n, cv2.CC_STAT_AREA]
        if w != binary_img.shape[1]:
            # too small to be considered a date label
            if area <= 0.05:
                continue
            if h in height_to_cc_table:
                height_to_cc_table[h].append(((x0, y0), (w, h), centroid))
            else:
                height_to_cc_table[h] = [((x0, y0), (w, h), centroid)]
    eq_height_ccs = []
    for _, items in height_to_cc_table.items():
        if len(items) > 1:
            eq_height_ccs.extend(items)
    # sort by x and y
    eq_height_ccs.sort(key=lambda item: item[0][0])
    eq_height_ccs.sort(key=lambda item: item[0][1])
    bgr_display = binary_img.copy()
    for cc in eq_height_ccs:
        pt, dim, _ = cc
        w, h = dim
        cv2.rectangle(bgr_display, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)
    cv2.imshow('ccs', bgr_display)
    cv2.waitKey(0)
    return eq_height_ccs

def get_all_locs_with_matches_from_cc(binary_img, possible_date_ccs, templates_to_match):
    def resize_cc(connected_component, MAX_HEIGHT=100, MAX_WIDTH=100):
        width = connected_component.shape[0]
        if width < MAX_WIDTH:
            width = MAX_WIDTH
        return cv2.resize(connected_component, (width, MAX_HEIGHT), interpolation = cv2.INTER_AREA)
    def get_cc_img(cc):
        pt, dim, _ = cc
        x, y = pt
        w, h = dim
        img = resize_cc(binary_img[x:x+w, y:y+h])
        return img
    def get_template_locations_with_max_score(cc, temp):
        img = get_cc_img(cc)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return np.where(res == max_val), max_val
    # all templates are 100 x 100 size
    def are_overlapping_rect_points(pt1, pt2, w=100, h=100):
        pt3 = (pt1[0] + w, pt1[1] + h)
        return (pt2[0] >= pt1[0] and pt2[0] <= pt3[0])\
            and (pt2[1] >= pt1[1] and pt2[1] <= pt3[1])
    def get_score(item):
        return item[1]
    def get_pt(item):
        return item[2]
    all_locations_with_matches = []
    for cc in possible_date_ccs:
        all_cc_temp_matches = []
        for temp_content in templates_to_match:
            temp, temp_name = temp_content
            loc, max_correlation = get_template_locations_with_max_score(cc, temp)
            all_cc_temp_matches.extend([(temp_name, max_correlation, pt) for pt in [pt for pt in zip(*loc[::-1])]])
        # sort by coordinates the cc temp matches
        all_cc_temp_matches.sort(key=lambda item: get_pt(item)[0])
        all_cc_temp_matches.sort(key=lambda item: get_pt(item)[1])
        # check if there are any overlapping matches, and if
        # there are then pick the one with the better score.
        best_score_temp_matches = []
        for i in range(1, len(all_cc_temp_matches)):
            curr = i
            prev = i - 1
            while i < len(all_cc_temp_matches) and\
                are_overlapping_rect_points(all_cc_temp_matches[prev][2],
                    all_cc_temp_matches[curr][2]):
                if get_score(all_cc_temp_matches[prev]) < get_score(all_cc_temp_matches[curr]):
                    prev = curr
                i += 1
                curr = i
            best_score_temp_matches.append(all_cc_temp_matches[prev])
        all_locations_with_matches.extend([(item[0], cc[0], cc[1], cc[2]) for item in best_score_temp_matches])
    return all_locations_with_matches




def detect_expiry_date(image, image_name, log_file, templates_to_match):
    print(f'detecting expiry date for {image_name}')
    log_file.write(f'detecting expiry date for {image_name}\n')
    # 1)process the image so that it is possible to detect
    _, black_labels, white_labels = get_black_and_white_labels(image)
    cv2.imshow('black labels image', black_labels)
    cv2.waitKey(0)
    cv2.imshow('white labels image', white_labels)
    cv2.waitKey(0)
    print('MATCHING TEMPLATES...')
    # 2) the expiry date labels can be either in black or white,
    # so process them both
    all_cc_locs_with_matches_black_labels = get_all_locs_with_matches_from_cc(black_labels, get_possible_date_CCs(black_labels),
        templates_to_match)
    all_cc_locs_with_matches_white_labels = get_all_locs_with_matches_from_cc(white_labels, get_possible_date_CCs(white_labels),
        templates_to_match)
    print('DONE MATCHING TEMPLATES')
    def get_label(item):
        return item[0]
    def output_date_formt_matches(date_format_matches):
        for date_format_match in date_format_matches:
            log_file.write(f'{date_format_match}\n')
    # 3) figure out what date the templates spell out based off of sorting points in
    # the image.
    output_label_black = ''.join(get_label(item) for item in all_cc_locs_with_matches_black_labels)
    log_file.write(f'Ouput black label based off of matches : {output_label_black}\n')
    output_label_white = ''.join(get_label(item) for item in all_cc_locs_with_matches_white_labels)
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
    log_file = open('logs-scale-ccs.txt', 'w+')
    for fname in image_file_names:
        image = cv2.imread(fname)
        detect_expiry_date(image, fname, log_file, templates_to_match)
    log_file.close()

if __name__ == '__main__':
    main()