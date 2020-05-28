import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Bidirectional, Dense, LSTM, TimeDistributed

from word_etymologist import dataset as ds


def get_sequential_model_input(char_map, window, word, root=None):
    len_word = len(word)
    len_map = len(char_map)
    word += (window - 1) * " "

    # create a sequence of random numbers in [0,1]
    X = np.array([char_map[c] for i in range(len_word) for c in word[i:i + window]])
    X = to_categorical(X, num_classes=len_map)
    X = X.reshape((1, -1, len_map * window))

    if root is not None:
        y = np.array(root)
        y = y.reshape((1, -1, 1))

        return X, y
    else:
        return X


def get_bidirectional_model_input(char_map, word, root=None):
    len_map = len(char_map)

    # create a sequence of random numbers in [0,1]
    X = np.array([char_map[c] for c in word])
    X = to_categorical(X, num_classes=len_map)
    X = X.reshape((1, -1, len_map))

    if root is not None:
        y = np.array(root)
        y = y.reshape((1, -1, 1))

        return X, y
    else:
        return X


def get_sequential_tag(hidden_layers, window):
    return f"model_{hidden_layers}_{window}"


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
    try:
        model.load_weights(model_weights_file)
    except OSError as e:
        print(f"Failed to load weights from file {model_weights_file}, "
              f"weights remain unchanged in their default initialization.")
        print(e)


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

    return model


class ModelWrapper:
    def __init__(self, h_layers):
        self.h_layers = h_layers
        self.char_map = ds.get_char_mapping(include_space=False)
        self.len_map = len(self.char_map)

        self.base_model = get_bidirectional_model(h_layers, self.len_map, n_timesteps=1)
        self.tag = get_bidirectional_tag(h_layers)
        load_weights(self.base_model, self.tag)

        self.model_dict = {}

    def _add_model(self, len_word):
        model = get_bidirectional_model(self.h_layers, self.len_map, n_timesteps=len_word)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model_dict[len_word] = model

    def _get_model(self, word):
        model_dict = self.model_dict
        len_word = len(word)
        if len_word not in model_dict:
            self._add_model(len_word)

        return model_dict[len_word]

    def train(self, word, root):
        base_model = self.base_model
        model = self._get_model(word)

        model.set_weights(base_model.get_weights())

        try:
            X, y = get_bidirectional_model_input(self.char_map, word, root)
        except KeyError:
            raise

        try:
            model.fit(X, y, epochs=1, batch_size=1, verbose=0)
        except Exception as ex:
            print(f"Exception of type {type(ex).__name__} "
                  f"while training with word: {word}")
            print(f"Exception Arguments:\n{ex.args!r}")
            return

        base_model.set_weights(model.get_weights())

    def predict(self, word):
        base_model = self.base_model
        model = self._get_model(word)
        model.set_weights(base_model.get_weights())

        try:
            X = get_bidirectional_model_input(self.char_map, word)
        except KeyError:
            raise

        return model.predict_classes(X, verbose=0)

    def save(self):
        save_weights(self.base_model, self.tag)

    def inspect(self):
        print(self.base_model.summary())
