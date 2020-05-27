import PySimpleGUIWeb as sg


def main(h_layers):
    layout = [
        [sg.Text('Close the browser tab to exit')],
        [sg.Text('Word'), sg.Input(key='-WORD-')],
        [sg.Text('Expected Output (Only required for training)'), sg.Input(key='-EXPECTED-')],
        [sg.Text('Predicted'), sg.Output(key='-PREDICTED-')],
        [sg.Button('Train'), sg.Button('Predict'), sg.Save(), sg.Exit()]
    ]

    window = sg.Window('Interactive Training and Evaluation', layout)
    while True:
        event, values = window.read()

        if event is None:
            # TODO save first
            break
        elif event == "Train":
            pass
        elif event == "Predict":
            pass
        elif event == "Save":
            pass

    window.close()


if __name__ == '__main__':
    main(0)