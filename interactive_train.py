import shlex

import dataset as ds
import models as mdls


class Command:
    HELP = "help"
    TRAIN = "train"
    PREDICT = "predict"
    INSPECT = "inspect"
    EXIT = "exit"

def print_cmd_help():
    print("help <cmd>: Prints details of command <cmd>")


def print_cmd_train():
    print("train <word> <root>: Trains model for the given datapoint.")
    print("                       <word> consists of latin characters.")
    print("                       <root> is a sequence of 1 or 0 which")
    print("                       indicates if corresponding character in <word> has Greek root.")


def print_cmd_predict():
    print("predict <word>: Predicts whether each character in <word> has Greek root.")
    print("                  Without training prediction will be inaccurate.")
    print("                  <word> consists of latin characters.")


def print_cmd_inspect():
    print("inspect: Prints information about the model")


def print_cmd_exit():
    print("exit: Exits")


def print_usage():
    print("Usage: <cmd> [<args>]")
    print("Commands:")
    print_cmd_help()
    print_cmd_train()
    print_cmd_predict()
    print_cmd_inspect()
    print_cmd_exit()


def print_help(cmd=None):
    if cmd == Command.HELP:
        print_cmd_help()
    elif cmd == Command.TRAIN:
        print_cmd_train()
    elif cmd == Command.PREDICT:
        print_cmd_predict()
    elif cmd == Command.EXIT:
        print_cmd_exit()
    else:
        print_usage()


def new_model(model_dict, h_layers, len_map, len_word):
    model = mdls.get_bidirectional_model(h_layers, len_map, n_timesteps=len_word)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_dict[len_word] = model


def main(h_layers):
    tag = mdls.get_bidirectional_tag(h_layers)
    char_map = ds.get_char_mapping()
    len_map = len(char_map)

    base_model = mdls.get_bidirectional_model(h_layers, len_map, n_timesteps=1)
    mdls.load_weights(base_model, tag)

    model_dict = {}
    print_usage()
    while True:
        try:
            cmd, *args = shlex.split(input('> '))
        except ValueError:
            # empty line
            continue

        if cmd == Command.HELP:
            try:
                print_help(args[0])
            except IndexError:
                print_help()

        elif cmd == Command.TRAIN:
            try:
                word = args[0].lower()
            except IndexError:
                print(f"Command {Command.TRAIN} requires two arguments")
                print_cmd_train()
                continue

            len_word = len(word)
            try:
                root = list(map(int, args[1]))
            except IndexError:
                print(f"Command {Command.TRAIN} requires two arguments")
                print_cmd_train()
                continue
            except ValueError:
                print(f"Argument after '{word}' should be a string of 0 and 1 "
                      f"of same length as '{word}'.")
                continue

            try:
                X, y = mdls.get_bidirectional_model_input(char_map, word, root)
            except KeyError:
                print(f"Invalid word '{word}'")
                continue

            if len_word not in model_dict:
                new_model(model_dict, h_layers, len_map, len_word)

            model = model_dict[len_word]
            model.set_weights(base_model.get_weights())

            try:
                model.fit(X, y, epochs=1, batch_size=1, verbose=0)
            except Exception as ex:
                print(f"Exception of type {type(ex).__name__} "
                      f"while training with word: {word}")
                print(f"Exception Arguments:\n{ex.args!r}")

            base_model.set_weights(model.get_weights())
        elif cmd == Command.PREDICT:
            try:
                word = args[0].lower()
            except IndexError:
                print(f"Command {Command.PREDICT} requires one argument")
                print_cmd_predict()
                continue

            len_word = len(word)

            try:
                X = mdls.get_bidirectional_model_input(char_map, word)
            except KeyError:
                print(f"Invalid word '{word}'")
                continue

            if len_word not in model_dict:
                new_model(model_dict, h_layers, len_map, len_word)

            model = model_dict[len_word]
            model.set_weights(base_model.get_weights())

            yhat = model.predict_classes(X, verbose=0)
            print(f"Predicted: {yhat[0].flatten()}")
        elif cmd == Command.INSPECT:
            print(base_model.summary())
        elif cmd == Command.EXIT:
            print("Saving weights")
            mdls.save_weights(base_model, tag)
            print("Exiting")
            break
        else:
            print(f"Unknown command: {cmd}")
