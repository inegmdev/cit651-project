import io
import os

import string
import time
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import save_model, load_model

from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)

AUTOTUNE = tf.data.AUTOTUNE


def load_train(data_folder, filename, shuffle_data=True, remove_non_toxic=False,
               num_of_records=-1, batch_size=16, clean_prct=50, class_column_name=""):
    print(
        f"load_train({data_folder}, {filename}, {shuffle_data}, {remove_non_toxic}, {num_of_records}, {batch_size}, {clean_prct}, {class_column_name}""):")
    data_path = os.path.join(data_folder, filename)
    df = pd.read_csv(data_path, encoding='utf8')

    print(f'dataframe shape: {df.shape}')

    if remove_non_toxic == True and clean_prct == 0:
        # keep only the non clean text
        df = df.loc[df.sum(axis=1, numeric_only=True) != 0]
    else:
        clean_df = df.loc[df.sum(axis=1, numeric_only=True) == 0]
        non_clean_df = df.loc[df[class_column_name] != 0]
        print(f"adding {len(non_clean_df)} records of toxic text")
        no_of_clean_records = clean_prct * len(clean_df) // 100
        print(f"adding {no_of_clean_records} records of non-toxic text")
        clean_df = clean_df.sample(n=no_of_clean_records)
        df = non_clean_df.append(clean_df)
    print(f'dataframe shape again: {df.shape}')

    if num_of_records == -1:
        num_of_records = df.shape[0]
    print(f'loading {num_of_records} records')
    if shuffle_data:
        df = df.sample(n=num_of_records)
    else:
        df = df[df.columns][:num_of_records]

    print(df.head())
    train_df, valid_df = train_test_split(df, test_size=0.2)
    print("train count:", len(train_df))
    print("valid count:", len(valid_df))

    train_id = train_df['id'].to_numpy().reshape(-1, 1)
    train_text = train_df['comment_text'].to_numpy().reshape(-1, 1)
    train_non_clean_lbl = train_df[class_column_name].to_numpy().reshape(-1, 1)

    valid_id = valid_df['id'].to_numpy().reshape(-1, 1)
    valid_text = valid_df['comment_text'].to_numpy().reshape(-1, 1)
    valid_non_clean_lbl = valid_df[class_column_name].to_numpy().reshape(-1, 1)

    train_id_data_set = tf.data.Dataset.from_tensor_slices((train_id, train_text))
    train_id_data_set = train_id_data_set.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

    non_clean_train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_non_clean_lbl))
    non_clean_train_dataset = non_clean_train_dataset.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

    valid_id_data_set = tf.data.Dataset.from_tensor_slices((valid_id, valid_text))
    valid_id_data_set = valid_id_data_set.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

    non_clean_valid_dataset = tf.data.Dataset.from_tensor_slices((valid_text, valid_non_clean_lbl))
    non_clean_valid_dataset = non_clean_valid_dataset.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

    data_tuple = (
        train_id_data_set,
        valid_id_data_set,
        non_clean_train_dataset,
        non_clean_valid_dataset

    )
    return data_tuple


def load_test(data_folder, test_text_filename, test_lbl_filename, shuffle_data=True, remove_unused=False,
              num_of_records=-1, batch_size=16):
    data_path = os.path.join(data_folder, test_text_filename)
    df_text = pd.read_csv(data_path)

    text_id = df_text['id'].to_numpy().reshape(-1, 1)
    text = df_text['comment_text'].to_numpy().reshape(-1, 1)

    toxic_test_dataset = tf.data.Dataset.from_tensor_slices((text))
    toxic_test_dataset = toxic_test_dataset.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)

    data_tuple = (
        text_id, toxic_test_dataset)

    return data_tuple


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    only_ascii = tf.strings.regex_replace(stripped_html, '[^%s]' % (string.ascii_letters + string.whitespace), ' ')
    no_repeated_letters = only_ascii
    for letter in string.ascii_letters + string.whitespace:
        no_repeated_letters = tf.strings.regex_replace(no_repeated_letters, '(%s){4,}' % letter, '\\1')

    short_words = tf.strings.regex_replace(no_repeated_letters, '(\w{20})\w{1,}', '\\1')
    return short_words

# https://wandb.ai/site

def build_model(vectorize_layer, max_features, embedding_dim):
    model = tf.keras.Sequential([layers.Input(shape=(1,), dtype=tf.string),
                                 vectorize_layer,
                                 layers.Embedding(max_features, embedding_dim, name="embedding"),
                                 layers.Dropout(0.2),
                                 layers.GlobalAveragePooling1D(),
                                 layers.Dropout(0.2),
                                 layers.Dense(1, activation='sigmoid')])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(from_logits=True, name='auc')])

    return model


def plot_history(history, model_name):
    history_dict = history.history
    keylist = list(history_dict.keys())
    print(keylist)
    acc = history_dict[keylist[1]]
    val_acc = history_dict[keylist[3]]
    loss = history_dict[keylist[0]]
    val_loss = history_dict[keylist[2]]

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, c='b', label='Training loss')
    plt.plot(epochs, val_loss, c='r', label='Validation loss')
    plt.title(f'Training and validation loss for {model_name} model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(
        f"./plots/Train_validatn_loss_for_{model_name}_model_{datetime.datetime.now().strftime('%Y_%m_%d%H_%M_%S')}.png")

    plt.show()

    plt.plot(epochs, acc, c='b', label='Training acc')
    plt.plot(epochs, val_acc, c='r', label='Validation acc')
    plt.title(f'Training and validation AUC for {model_name} model')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend(loc='lower right')
    plt.savefig(
        f"./plots/Train_validatn_AUC_for_{model_name}_model_{datetime.datetime.now().strftime('%Y_%m_%d%H_%M_%S')}.png")

    plt.show()


def run2(data_dir, models_dir, train_record_count=-1, test_record_count=-1, shuffle_data=True, remove_not_toxic=False):
    max_features = 10000
    sequence_length = 100  # 75% of the have 76 words
    embedding_dim = 16
    epochs = 10

    classifiers = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    clean_percent = [20, 10, 10, 10, 10, 10]
    models = []
    if not os.path.exists(models_dir):  # or len(os.listdir(models_dir)) != 6:

        print("generating toxic_models ...")
        # print('[%s]' % string.punctuation)

        for cls in range(len(classifiers)):
            classifier = classifiers[cls]
            vectorize_layer = layers.TextVectorization(
                standardize=custom_standardization,
                max_tokens=max_features,
                output_mode='int',
                output_sequence_length=sequence_length)
            tpl = load_train(data_dir, 'train.csv', shuffle_data, remove_not_toxic, train_record_count,
                             clean_prct=clean_percent[cls], class_column_name=classifier)
            train_train_id = tpl[0]
            valid_train_id = tpl[1]
            toxic_train_dataset = tpl[2]
            toxic_valid_dataset = tpl[3]

            train_text = train_train_id.map(lambda x, y: y)
            vectorize_layer.adapt(train_text)

            print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

            vocab = vectorize_layer.get_vocabulary()
            out_m = io.open(f'./metadata/{classifier}_metadata.tsv', 'w', encoding='utf-8')

            for index, word in enumerate(vocab):
                if index == 0:
                    continue  # skip 0, it's padding.
                # vec = weights[index]
                # out_v.write('\t'.join([str(x) for x in vec]) + "\n")
                out_m.write(word + "\n")
            # out_v.close()
            out_m.close()

            toxic_model = build_model(vectorize_layer, max_features, embedding_dim)

            st = time.time()
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.01)

            history = toxic_model.fit(toxic_train_dataset, validation_data=toxic_valid_dataset,
                                      epochs=epochs, callbacks=[callback])
            et = time.time()
            print(f"{classifier} model training time :{et - st:0.2f} seconds")
            plot_history(history, classifier)
            save_model(model=toxic_model, filepath=f'/saved_models/{classifier}_model')
            models += [toxic_model]

        print("finished training models")

    else:
        for classifier in classifiers:
            model = load_model(f'{models_dir}/{classifier}_model')
            models += [model]

        print("all models were loaded")

    tpl_test = load_test(data_dir, 'test.csv', 'test_labels.csv', False, False, test_record_count)
    test_id = tpl_test[0]
    test_dataset = tpl_test[1]

    print('toxic_model test started')
    predictions = models[0].predict(test_dataset)
    print('toxic_model test ended')

    for i in range(1, len(models)):
        print(f'{classifiers[i]}  test started')
        predictions = np.hstack((predictions, models[i].predict(test_dataset)))
        print(f'{classifiers[i]} test ended')

    np.nan_to_num(predictions, nan=0.0, copy=False)
    output = pd.DataFrame({'id': test_id.flatten(),
                           'toxic': predictions[:, 0],
                           'severe_toxic': predictions[:, 1],
                           'obscene': predictions[:, 2],
                           'threat': predictions[:, 3],
                           'insult': predictions[:, 4],
                           'identity_hate': predictions[:, 5]
                           })
    output.to_csv('submission.csv', index=False)


run2(data_dir='data', models_dir='/saved_models',
     train_record_count=-1, test_record_count=-1, shuffle_data=True, remove_not_toxic=False)
