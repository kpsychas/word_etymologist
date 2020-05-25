import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Bidirectional, Dense, LSTM, TimeDistributed


def get_sequential_model_input(word, root, char_map, window):
    len_word = len(word)
    len_map = len(char_map)
    word += (window - 1) * " "

    # create a sequence of random numbers in [0,1]
    X = np.array([char_map[c] for i in range(len_word) for c in word[i:i + window]])
    X = to_categorical(X, num_classes=len_map)
    y = np.array(root)

    X = X.reshape((1, -1, len_map * window))
    y = y.reshape((1, -1, 1))

    return X, y


def get_bidirectional_model_input(word, root, char_map):
    len_map = len(char_map)

    # create a sequence of random numbers in [0,1]
    X = np.array([char_map[c] for c in word])
    X = to_categorical(X, num_classes=len_map)
    y = np.array(root)

    X = X.reshape((1, -1, len_map))
    y = y.reshape((1, -1, 1))

    return X, y


def get_sequential_tag(window, hidden_layers):
    return f"model_{window}_{hidden_layers}"


def get_bidirectional_tag(hidden_layers):
    return f"model_{hidden_layers}"


def get_weights_file(tag):
    return f"{tag}.h5"


def get_model_file(tag):
    return f"{tag}.json"


def load_model(tag):
    model_file = get_model_file(tag)

    with open(model_file, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    # load weights into new model
    load_weights(model, tag)
    print("Loaded model from disk")

    return model


def save_sequential_model(model, tag):
    # serialize model to JSON
    model_file = get_model_file(tag)

    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)

    save_weights(model, tag)
    print("Saved model to disk")


def load_weights(model, tag):
    model_weights_file = get_weights_file(tag)
    model.load_weights(model_weights_file)


def save_weights(model, tag):
    # serialize weights to HDF5
    model_weights_file = get_weights_file(tag)
    model.save_weights(model_weights_file)


def get_sequential_model(tag, h_layers, window, len_map):
    try:
        model = load_model(tag)
    except FileNotFoundError:
        # define LSTM
        model = Sequential()
        model.add(LSTM(h_layers, return_sequences=True, input_shape=(None, len_map * window)))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    return model


def get_bidirectional_model(h_layers, len_map, n_timesteps):
    model = Sequential()
    model.add(Bidirectional(LSTM(h_layers, return_sequences=True), input_shape=(n_timesteps, len_map)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
