import PySimpleGUI as sg

from word_etymologist import models as mdls, dataset as ds


def main(h_layers):
    model_wrapper = mdls.ModelWrapper(h_layers)

    layout = [
        [sg.Text('Close the browser tab to exit (progress is saved)')],
        [sg.Text('Word'), sg.Input(key='-WORD-')],
        [sg.Text('Expected Output (Only required for training)'), sg.Input(key='-EXPECTED-')],
        [sg.Text('Output'), sg.Output(key='-OUTPUT-', size=(80, 20))],
        [sg.Button('Train'), sg.Button('Predict'), sg.Button('Inspect'), sg.Save(), sg.Exit()]
    ]

    window = sg.Window('Interactive Training and Evaluation', layout)
    while True:
        event, values = window.read()

        window['-OUTPUT-'].update("")
        if event is None:
            model_wrapper.save()
            break
        elif event == "Train":

            word = values['-WORD-'].lower()
            if not word:
                print("No word was provided for training")
                continue

            try:
                root = list(map(int, values['-EXPECTED-']))
            except ValueError:
                print(f"Train requires Expected Output to be set to "
                      f"a sequence of 0 and 1 same length as the Word.")
                continue

            try:
                model_wrapper.train(word, root)
            except KeyError:
                print(f"Invalid characters in word '{word}'. "
                      f"Only Latin characters are allowed")
                continue

        elif event == "Predict":
            word = values['-WORD-'].lower()
            if not word:
                print("No word was provided for prediction")
                continue

            try:
                yhat = model_wrapper.predict(word)
            except KeyError:
                print(f"Invalid characters in word '{word}'. "
                      f"Only Latin characters are allowed")
                continue
            else:
                window['-OUTPUT-'].update(f"Prediction: {yhat[0].flatten()}")

        elif event == "Save":
            model_wrapper.save()
            print("Saved Weights")
        elif event == "Inspect":
            model_wrapper.inspect()
        elif event == "Exit":
            model_wrapper.save()
            print("Exiting")
            break

    window.close()


if __name__ == '__main__':
    main(200)
