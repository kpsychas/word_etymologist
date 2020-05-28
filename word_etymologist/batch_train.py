import random

from word_etymologist import models as mdls, dataset as ds


def get_random_word(words):
    return random.choice(words)


def get_test_word():
    return 'hypertension', [1,1,1,1,1,0,0,0,0,0,0,0]


def train(h_layers, window):
    tag = mdls.get_sequential_tag(h_layers, window)
    char_map = ds.get_char_mapping()
    words = ds.get_annotated_words(get_list=True)

    len_map = len(char_map)

    """ 
    Hidden layers of LSTM can be customized 
    Parameter return_sequences is set to True because we are interested in 
    classifying every output of the input sequence not just the final one.
    """

    model = mdls.get_sequential_model(tag, h_layers, window, len_map)
    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train LSTM
    for epoch in range(100):
        # get new random word
        word, root = get_random_word(words)
        X, y = mdls.get_sequential_model_input(char_map, window, word, root)

        # fit model for one epoch on this sequence
        try:
            model.fit(X, y, epochs=1, batch_size=1, verbose=0)
        except ValueError as ex:
            print(f"Exception while training with word: {word}")
            print(ex)
        except Exception as ex:
            print(f"Exception of type {type(ex).__name__} "
                  f"while training with word: {word}")
            print(f"Exception Arguments:\n{ex.args!r}")

    mdls.save_sequential_model(model, tag)


def evaluate(h_layers, window):
    tag = mdls.get_sequential_tag(h_layers, window)
    char_map = ds.get_char_mapping()
    try:
        model = mdls.load_model(tag)
    except FileNotFoundError as e:
        print(e)
        print(f"Make sure that sequential model with {h_layers} hidden layers "
              f"and {window} window size is trained")
        return

    # evaluate LSTM
    word, root = get_test_word()
    X, y = mdls.get_sequential_model_input(char_map, window, word, root)
    # print(X.shape, y.shape)
    yhat = model.predict_classes(X, verbose=0)
    for yi, yhati in zip(y, yhat):
        print(f"Expected: {yi.flatten()}")
        print(f"Predicted: {yhati.flatten()}")

