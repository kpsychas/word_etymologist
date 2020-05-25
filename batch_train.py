import random

import dataset as ds
import models as mdls


def get_random_word(words):
    return random.choice(words)


def get_test_word():
    return 'hypertension', [1,1,1,1,1,0,0,0,0,0,0,0]


def train(h_layers, window):
    tag = mdls.get_sequential_tag(window, h_layers)
    char_map = ds.get_char_mapping()
    words = ds.get_annotated_words(get_list=True)

    len_map = len(char_map)

    """ 
    Hidden layers of LSTM can be customized 
    Parameter return_sequences is set to True because we are interested in 
    classifying every output of the input sequence not just the final one.
    """

    model = mdls.get_model(tag, len_map, h_layers, window)
    print(model.summary(90))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train LSTM
    for epoch in range(1000):
        # get new random word
        word, root = get_random_word(words)
        X, y = get_model_input(word, root, char_map, window)

        # fit model for one epoch on this sequence
        try:
            model.fit(X, y, epochs=1, batch_size=1, verbose=0)
        except:
            print(f"Error while training with word: {word} - Skipping word")

    mdls.save_sequential_model(model, tag)


def evaluate(window, h_layers):
    tag = mdls.get_sequential_tag(window, h_layers)
    char_map = ds.get_char_mapping()
    model = mdls.load_sequential_model(tag)

    # evaluate LSTM
    word, root = get_test_word()
    X, y = get_model_input(word, root, char_map, window)
    yhat = model.predict_classes(X, verbose=0)
    for yi, yhati in zip(y, yhat):
        print(f"Expected: {yi.flatten()}")
        print(f"Predicted: {yhati.flatten()}")

