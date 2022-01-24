from random import random, randint

from flask import Flask
from tensorflow.keras.models import load_model
import tensorflow as tf
import string
import numpy as np

app = Flask(__name__)


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[^%s]' % (string.ascii_letters + string.whitespace), '')


print("loading ..........")
models_dir = 'saved_models'
predictions = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

toxic_model = load_model(f'{models_dir}/toxic_model')
severe_toxic_model = load_model(f'{models_dir}/severe_toxic_model')
obscene_model = load_model(f'{models_dir}/obscene_model')
insult_model = load_model(f'{models_dir}/insult_model')
threat_model = load_model(f'{models_dir}/threat_model')
hate_model = load_model(f'{models_dir}/identity_hate_model')

print("all models loaded")
print(toxic_model.summary())


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
    # p = tf.data.Dataset.from_tensor_slices([phrase])
    # p = np.array(phrase)
    # print("phrase as no array: ", p)
    is_toxic = round(float(toxic_model.predict([phrase])))
    is_s_toxic = round(float(severe_toxic_model.predict([phrase])))
    is_obscene = round(float(obscene_model.predict([phrase])))
    is_insult = round(float(insult_model.predict([phrase])))
    is_threat = round(float(threat_model.predict([phrase])))
    is_hate = round(float(hate_model.predict([phrase])))
    print(is_toxic)
    response = {'phrase': phrase,
                'predictions': {'toxic': is_toxic,
                                'severe_toxic': is_s_toxic,
                                'obscene': is_obscene,
                                'threat': is_insult,
                                'insult': is_threat,
                                'identity_hate': is_hate, }}
    return response


if __name__ == '__main__':
    app.run()
