import shlex

import dataset as ds
import models as mdls


def print_usage():
    pass


def print_help():
    pass


def main(h_layers):
    tag = mdls.get_bidirectional_tag(h_layers)
    char_map = ds.get_char_mapping()
    len_map = len(char_map)

    print_usage()
    while True:
        cmd, *args = shlex.split(input('> '))

        if cmd == "help":
            print_help()
        elif cmd == "train":
            word = args[0]
            root = list(map(int, args[1]))
            X, y = mdls.get_bidirectional_model_input(word, root, char_map)

            model = mdls.get_bidirectional_model(h_layers, len(y))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif cmd == "predict":
            pass
        else:
            pass

        # https://docs.python.org/3/library/readline.html
