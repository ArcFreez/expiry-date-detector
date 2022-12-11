"""Grab the unique characters to match for template matching."""
def grab_unique_letrs(word, set_):
    for char in word:
        set_.add(char)
    
def main():
    date_numbers = [f'{i}' for i in range(0, 10)]
    months = ['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'JUL', 'AUG', 'SEP', 
        'OCT', 'NOV', 'DEC']
    special_chars = ['/', '-']
    set_ = set()
    log = open('unique_chars.txt', 'w+')
    for num in date_numbers:
        set_.add(f'{num}')
    for p in special_chars:
        grab_unique_letrs(p, set_)
    for m in months:
        grab_unique_letrs(m, set_)
    for item in set_:
        log.write(f'{item}\n')

if __name__ == '__main__':
    main()