import random

import numpy as np

from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from keras.layers import Dense, LSTM, TimeDistributed

import dataset as ds


class Program:
    TRAIN = 1
    EVALUATE = 2


M = 4  # number of look ahead characters (including current)
H_LAYERS = 200
PROGRAM = Program.EVALUATE

MODEL_FILE = f"model_{M}_{H_LAYERS}.json"
MODEL_WEIGHTS_FILE = f"model_{M}_{H_LAYERS}.h5"

# def get_random_word(words, char_map):
#     word = random.choice(words)
#
#     LW = len(word)
#     LC = len(char_map)-1
#     word += (M-1)*[(" ", 0)]
#     # create a sequence of random numbers in [0,1]
#     X = np.array([char_map[c] for i in range(LW) for c, _ in word[i:i+M]])
#     y = np.array([rid > 0 for _, rid in word[:LW]])
#
#     # (batch_size, timesteps, input_dim)
#     X = X.reshape((1, -1, LC*M))
#     y = y.reshape((1, -1, 1))
#     return X, y


def get_random_word(words, char_map):
    word, root = random.choice(words)

    LW = len(word)
    LC = len(char_map)

    word += (M-1)*" "
    X = np.array([char_map[c] for i in range(LW) for c in word[i:i+M]])
    X = to_categorical(X, num_classes=LC)
    y = np.array([root_id > 0 for root_id in root])

    # (batch_size, timesteps, input_dim)
    X = X.reshape((1, -1, LC*M))
    y = y.reshape((1, -1, 1))
    return X, y, word


def get_test_word(char_map):
    word, root = 'hypertension', [1,1,1,1,1,0,0,0,0,0,0,0]


    LW = len(word)
    LC = len(char_map)
    word += (M-1)*" "

    # create a sequence of random numbers in [0,1]
    X = np.array([char_map[c] for i in range(LW) for c in word[i:i+M]])
    X = to_categorical(X, num_classes=LC)
    y = np.array([root_id > 0 for root_id in root])

    X = X.reshape((1, -1, LC*M))
    y = y.reshape((1, -1, 1))

    return X, y


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(MODEL_FILE, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(MODEL_WEIGHTS_FILE)
    print("Saved model to disk")


def load_model():
    with open(MODEL_FILE, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(MODEL_WEIGHTS_FILE)
    print("Loaded model from disk")
    return model


def main_train():
    char_map = ds.get_char_mapping()
    words = ds.get_annotated_words(get_list=True)

    LC = len(char_map)

    # return_sequences will return full y sequence and that is what we
    # are interested in

    # define LSTM
    try:
        model = load_model()
    except FileNotFoundError:
        model = Sequential()
        model.add(LSTM(H_LAYERS, return_sequences=True, input_shape=(None, LC*M)))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    print(model.summary(90))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train LSTM
    for epoch in range(1000):
        # generate new random sequence
        X, y, word = get_random_word(words, char_map)
        # fit model for one epoch on this sequence
        try:
            model.fit(X, y, epochs=1, batch_size=1, verbose=0)
        except:
            print(f"Error while training with word: {word} - Skipping word")

    save_model(model)


def main_evaluate():
    char_map = ds.get_char_mapping()
    model = load_model()

    # evaluate LSTM
    X, y = get_test_word(char_map)
    yhat = model.predict_classes(X, verbose=0)
    for yi, yhati in zip(y, yhat):
        print(f"Expected: {yi}, Predicted: {yhati}")


def main():
    if PROGRAM == Program.TRAIN:
        main_train()
    elif PROGRAM == Program.EVALUATE:
        main_evaluate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hlayers", nargs='?', const=H_LAYERS, type=float, default=H_LAYERS)
    parser.add_argument("--window", nargs='?', const=M, type=float, default=M)
    parser.add_argument("--program", nargs='?', const=PROGRAM, type=int, default=PROGRAM)
    args = parser.parse_args()

    H_LAYERS = args.hlayers
    M = args.window
    PROGRAM = args.program
    main()
