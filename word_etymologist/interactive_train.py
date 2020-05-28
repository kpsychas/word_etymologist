import shlex

from word_etymologist import models as mdls


class Command:
    HELP = "help"
    TRAIN = "train"
    PREDICT = "predict"
    INSPECT = "inspect"
    SAVE = "save"
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


def print_cmd_save():
    print("save: Saves trained weights (weights are save by default on exit)")


def print_cmd_exit():
    print("exit: Exits")


def print_usage():
    print("Usage: <cmd> [<args>]")
    print("Commands:")
    print([Command.HELP, Command.TRAIN, Command.PREDICT, Command.INSPECT,
        Command.SAVE, Command.EXIT])
    print_cmd_help()


def print_help(cmd=None):
    if cmd == Command.HELP:
        print_cmd_help()
    elif cmd == Command.TRAIN:
        print_cmd_train()
    elif cmd == Command.PREDICT:
        print_cmd_predict()
    elif cmd == Command.INSPECT:
        print_cmd_inspect()
    elif cmd == Command.SAVE:
        print_cmd_save()
    elif cmd == Command.EXIT:
        print_cmd_exit()
    else:
        print_usage()


def new_model(model_dict, h_layers, len_map, len_word):
    model = mdls.get_bidirectional_model(h_layers, len_map, n_timesteps=len_word)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_dict[len_word] = model


def main(h_layers):
    model_wrapper = mdls.ModelWrapper(h_layers)

    print_usage()
    while True:
        try:
            cmd, *args = shlex.split(input('> '))
        except ValueError:
            # skip empty line
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
                model_wrapper.train(word, root)
            except KeyError:
                print(f"Invalid characters in word '{word}'. "
                      f"Only Latin characters are allowed")
                continue

        elif cmd == Command.PREDICT:
            try:
                word = args[0].lower()
            except IndexError:
                print(f"Command {Command.PREDICT} requires one argument")
                print_cmd_predict()
                continue

            try:
                yhat = model_wrapper.predict(word)
            except KeyError:
                print(f"Invalid characters in word '{word}'. "
                      f"Only Latin characters are allowed")
                continue
            else:
                print(f"Predicted: {yhat[0].flatten()}")

        elif cmd == Command.INSPECT:
            model_wrapper.inspect()
        elif cmd == Command.SAVE:
            model_wrapper.save()
        elif cmd == Command.EXIT:
            print("Saving weights")
            model_wrapper.save()
            print("Exiting")
            break
        else:
            print(f"Unknown command: {cmd}")
