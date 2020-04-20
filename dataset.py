#!/usr/bin/env python3
from collections import defaultdict
import csv


WORDS_FILE = "words.csv"


def get_annotated_words(get_list=False):
    if get_list:
        words = []
    else:
        words = defaultdict(None)

    with open(WORDS_FILE, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')

        for row in csv_reader:
            word = row['word']
            tags_raw = row['tags']
            tag_start = tags_raw.find("{")
            tag_end = tags_raw.find("}")
            tags = list(map(int, tags_raw[tag_start + 1:tag_end].split(',')))
            if get_list:
                words.append((word, tags))
            else:
                words[word] = (word, tags)

    return words


def get_char_mapping():
    """
    Maps lowercase latin characters to a feature array.

    :return:
    """
    from string import ascii_lowercase

    C_LEN = len(ascii_lowercase)

    char_map = {" ": C_LEN}
    for i, c in enumerate(ascii_lowercase):
        char_map[c] = i

    return char_map


def main():
    annotated_words = get_annotated_words()
    print(annotated_words)


if __name__ == "__main__":
    main()
