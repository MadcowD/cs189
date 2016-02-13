from collections import defaultdict
import glob
import re

WORD_COUNTS = {}
def count(text, freq):
    for word, count in freq.items():
        word = word.lower()
        if word not in WORD_COUNTS:
            WORD_COUNTS[word] = 0
        WORD_COUNTS[word] += count

filenames = glob.glob('*.txt')
for filename in filenames:
    with open(filename) as f:
        text = f.read() # Read in text from file
        text = text.replace('\r\n', ' ') # Remove newline character
        words = re.findall(r'\w+', text)
        freq = defaultdict(int) # Frequency of all words
        for word in words:
            freq[word] += 1
        count(text, freq)

WORDS = sorted(WORD_COUNTS, reverse=True, key=lambda x: WORD_COUNTS[x])

with open('counts.txt', 'w') as f:
    for word in WORDS:
        f.write('{} {}\n'.format(word, WORD_COUNTS[word]))
