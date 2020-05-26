#!/usr/bin/env python3

import interactive_train as intt
import batch_train as batt


class Program:
    TRAIN = 1
    EVALUATE = 2
    INTERACTIVE = 3


class MODEL:
    SEQUENTIAL = 'seq'
    BIDIRECTIONAL = 'bid'


def main(args):
    h_layers = args.hlayers
    window = args.window

    model_type = args.model
    program_mode = args.program
    # if model_type == MODEL.SEQUENTIAL:
    #     tag = mdls.get_sequential_tag(window, hlayers)
    # else:
    #     tag = mdls.get_bidirectional_tag(hlayers)

    if program_mode == Program.TRAIN and model_type == MODEL.SEQUENTIAL:
        batt.train(h_layers, window)
    elif program_mode == Program.EVALUATE and model_type == MODEL.SEQUENTIAL:
        batt.evaluate(h_layers, window)
    elif program_mode == Program.INTERACTIVE and model_type == MODEL.BIDIRECTIONAL:
        intt.main(h_layers)
    else:
        print(f'Model {model_type} is not supported for program {program_mode}')


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
    parser.add_argument("--model", choices=[MODEL.SEQUENTIAL, MODEL.BIDIRECTIONAL],
                        default=MODEL.SEQUENTIAL)
    parser.add_argument("--program", help='1 for training, 2 for evaluation, '
                                          '3 for interactive training',
                        nargs='?', const=PROGRAM, type=int, default=PROGRAM)
    return parser.parse_args()


if __name__ == "__main__":

    main(args=parse_cmd())
