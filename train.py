#!/usr/bin/env python3

from word_etymologist import interactive_train as intt, batch_train as batt, GUI_train as guit


class Program:
    TRAIN = 1
    EVALUATE = 2
    INTERACTIVE = 3
    GUI_INTERACTIVE = 4


def main(args):
    h_layers = args.hlayers
    window = args.window

    program_mode = args.program

    if program_mode == Program.TRAIN:
        batt.train(h_layers, window)
    elif program_mode == Program.EVALUATE:
        batt.evaluate(h_layers, window)
    elif program_mode == Program.INTERACTIVE:
        intt.main(h_layers)
    elif program_mode == Program.GUI_INTERACTIVE:
        guit.main(h_layers)
    else:
        print(f'Program {program_mode} is not a valid option')


def parse_cmd():
    import argparse

    # number of characters to look ahead (including current)
    # size of window of characters to use as input
    # pads with empty characters at the end of a word
    WINDOW = 4
    H_LAYERS = 100
    PROGRAM = Program.TRAIN

    parser = argparse.ArgumentParser()
    parser.add_argument("--hlayers", help='number of hidden layers of LSTM',
                        nargs='?', const=H_LAYERS, type=float, default=H_LAYERS)
    parser.add_argument("--window", help='size of character window that is used as input',
                        nargs='?', const=WINDOW, type=float, default=WINDOW)
    parser.add_argument("--program", help='1 for training, 2 for evaluation, '
                                          '3 for interactive training,'
                                          '4 for interactive training with GUI',
                        nargs='?', const=PROGRAM, type=int, default=PROGRAM)
    return parser.parse_args()


if __name__ == "__main__":

    main(args=parse_cmd())
