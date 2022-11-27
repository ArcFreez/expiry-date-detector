def grab_unique_letrs(word, set_):
    for char in word:
        set_.add(char)
    
def main():
    date_numbers = [f'{i}' for i in range(0, 10)]
    food_exp_phrases = ['Best if Used By', 'BEST IF USED BY',
         'Best if Used Before', 'BEST IF USED BEFORE', 'Use-By', 'USE-BY']
    drug_exp_phrases = ['EXP', 'EXPIRY', 'EXP DATE', 'Exp. Date']
    special_chars = ['/', '-']
    set_ = set()
    log = open('unique_chars.txt', 'w+')
    for num in date_numbers:
        set_.add(f'{num}')
    for p in food_exp_phrases:
        grab_unique_letrs(p, set_)
    for p in drug_exp_phrases:
        grab_unique_letrs(p, set_)
    for p in special_chars:
        grab_unique_letrs(p, set_)
    for item in set_:
        log.write(f'{item}\n')

if __name__ == '__main__':
    main()