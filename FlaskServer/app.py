from random import random, randint

from flask import Flask

app = Flask(__name__)

predictions = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@app.route('/')
def index():
    return 'Index page!'


@app.route('/hello')
def hello_world():
    return 'Hello world page!'


@app.route('/predict/<phrase>')
def predict_request(phrase):


    return predict(phrase)


def predict(phrase):
    rnd = randint(0, len(predictions)-1)

    response = {'phrase': phrase, 'prediction': predictions[rnd]}
    return response


if __name__ == '__main__':
    app.run()
