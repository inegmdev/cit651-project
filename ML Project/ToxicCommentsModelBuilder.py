import io
import os

import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import losses, Input
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import save_model, load_model
from explore_data import get_num_words_per_sample, min_max_sample_length, plot_sample_length_distribution, \
    plot_frequency_distribution_of_ngrams
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)

AUTOTUNE = tf.data.AUTOTUNE


def load_train(data_folder, filename, shuffle_data=True, remove_not_toxic=False, num_of_records=-1):
    data_path = os.path.join(data_folder, filename)
    df = pd.read_csv(data_path, encoding='utf8')
    print(f'dataframe shape: {df.shape}')
    if remove_not_toxic:
        df = df.loc[df.sum(axis=1, numeric_only=True) != 0]
    print(f'dataframe shape again: {df.shape}')

    if num_of_records == -1:
        num_of_records = df.shape[0]
    print(f'loading {num_of_records} records')
    if shuffle_data:
        df = df.sample(n=num_of_records)
    else:
        df = df[df.columns][:num_of_records]

    train_df, valid_df = train_test_split(df, test_size=0.2)
    print("train count:", len(train_df))
    print("valid count:", len(valid_df))

    # print(df.shape)
    # print("df['comment_text']",df['comment_text'].to_numpy().reshape(-1, 1))

    train_id = train_df['id'].to_numpy().reshape(-1, 1)
    train_text = train_df['comment_text'].to_numpy().reshape(-1, 1)
    train_toxic_lbl = train_df['toxic'].to_numpy().reshape(-1, 1)
    train_severe_toxic_lbl = train_df['severe_toxic'].to_numpy().reshape(-1, 1)
    train_obscene_lbl = train_df['obscene'].to_numpy().reshape(-1, 1)
    train_threat_lbl = train_df['threat'].to_numpy().reshape(-1, 1)
    train_insult_lbl = train_df['insult'].to_numpy().reshape(-1, 1)
    train_identity_hate_lbl = train_df['identity_hate'].to_numpy().reshape(-1, 1)

    valid_id = valid_df['id'].to_numpy().reshape(-1, 1)
    valid_text = valid_df['comment_text'].to_numpy().reshape(-1, 1)
    valid_toxic_lbl = valid_df['toxic'].to_numpy().reshape(-1, 1)
    valid_severe_toxic_lbl = valid_df['severe_toxic'].to_numpy().reshape(-1, 1)
    valid_obscene_lbl = valid_df['obscene'].to_numpy().reshape(-1, 1)
    valid_threat_lbl = valid_df['threat'].to_numpy().reshape(-1, 1)
    valid_insult_lbl = valid_df['insult'].to_numpy().reshape(-1, 1)
    valid_identity_hate_lbl = valid_df['identity_hate'].to_numpy().reshape(-1, 1)

    train_id_data_set = tf.data.Dataset.from_tensor_slices((train_id, train_text))
    train_id_data_set = train_id_data_set.cache().prefetch(buffer_size=AUTOTUNE)

    toxic_train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_toxic_lbl))
    toxic_train_dataset = toxic_train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    # toxic_train_dataset = toxic_train_dataset.batch(10)

    severe_toxic_train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_severe_toxic_lbl))
    severe_toxic_train_dataset = severe_toxic_train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    obscene_train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_obscene_lbl))
    obscene_train_dataset = obscene_train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    threat_train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_threat_lbl))
    threat_train_dataset = threat_train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    insult_train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_insult_lbl))
    insult_train_dataset = insult_train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    hate_train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_identity_hate_lbl))
    hate_train_dataset = hate_train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    valid_id_data_set = tf.data.Dataset.from_tensor_slices((valid_id, valid_text))
    valid_id_data_set = valid_id_data_set.cache().prefetch(buffer_size=AUTOTUNE)

    toxic_valid_dataset = tf.data.Dataset.from_tensor_slices((valid_text, valid_toxic_lbl))
    toxic_valid_dataset = toxic_valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    # toxic_valid_dataset = toxic_valid_dataset.batch(10)

    severe_toxic_valid_dataset = tf.data.Dataset.from_tensor_slices((valid_text, valid_severe_toxic_lbl))
    severe_toxic_valid_dataset = severe_toxic_valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    obscene_valid_dataset = tf.data.Dataset.from_tensor_slices((valid_text, valid_obscene_lbl))
    obscene_valid_dataset = obscene_valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    threat_valid_dataset = tf.data.Dataset.from_tensor_slices((valid_text, valid_threat_lbl))
    threat_valid_dataset = threat_valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    insult_valid_dataset = tf.data.Dataset.from_tensor_slices((valid_text, valid_insult_lbl))
    insult_valid_dataset = insult_valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    hate_valid_dataset = tf.data.Dataset.from_tensor_slices((valid_text, valid_identity_hate_lbl))
    hate_valid_dataset = hate_valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    data_tuple = (
        train_id_data_set,
        valid_id_data_set,
        toxic_train_dataset,
        toxic_valid_dataset,
        severe_toxic_train_dataset,
        severe_toxic_valid_dataset,
        obscene_train_dataset,
        obscene_valid_dataset,
        threat_train_dataset,
        threat_valid_dataset,
        insult_train_dataset,
        insult_valid_dataset,
        hate_train_dataset,
        hate_valid_dataset)
    return data_tuple




def load_test(data_folder, test_text_filename, test_lbl_filename, shuffle_data=True, remove_unused=False,
              num_of_records=-1):
    data_path = os.path.join(data_folder, test_text_filename)
    df_text = pd.read_csv(data_path)

    # print("type of df['comment_text']: ",type(df['comment_text'][0]))

    data_path = os.path.join('data', test_lbl_filename)
    df_lbl = pd.read_csv(data_path)
    df_merge = df_text.merge(df_lbl)

    if num_of_records == -1:
        num_of_records = df_merge.shape[0]
    # df = df[df.columns][:num_of_records]
    if remove_unused:
        df_merge = df_merge.loc[df_merge["toxic"] != -1]
    if shuffle_data:
        df_merge = df_merge.sample(n=num_of_records)
    else:
        df_merge = df_merge[df_merge.columns][:num_of_records]

    text_id = df_merge['id'].to_numpy().reshape(-1, 1)
    text = df_merge['comment_text'].to_numpy().reshape(-1, 1)
    data_tuple = (
        text_id, text, df_merge['toxic'].to_numpy().reshape(-1, 1), df_merge['severe_toxic'].to_numpy().reshape(-1, 1),
        df_merge['obscene'].to_numpy().reshape(-1, 1), df_merge['threat'].to_numpy().reshape(-1, 1),
        df_merge['insult'].to_numpy().reshape(-1, 1), df_merge['identity_hate'].to_numpy().reshape(-1, 1))

    # print(df_text.head())
    # print('-' * 100)
    #
    # print(df_lbl.head())
    # print('-' * 100)

    # print(df_merge.head())
    # print('-' * 100)
    # if shuffle_data:
    #     shuffle(data_tuple)

    return data_tuple


# def shuffle(data_tuple, seed=123):
#     np.random.seed(seed)
#     for d in data_tuple:
#         # print("d:" ,d)
#         np.random.shuffle(d)
#         np.random.seed(seed)


def train(data):
    return None


def test(data):
    return None


def plot_bar_chart(tup):
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    classes = tup[0]
    numbers = tup[1]
    # print(classes, numbers)
    plt.bar(classes, numbers)
    plt.show()


# def run():
#     tpl = load_train('train.csv', False)
#     train_id = tpl[0]
#     train_text = tpl[1]
#     train_toxic_lbl = tpl[2]
#     train_severe_toxic_lbl = tpl[3]
#     train_obscene_lbl = tpl[4]
#     train_threat_lbl = tpl[5]
#     train_insult_lbl = tpl[6]
#     train_identity_hate_lbl = tpl[7]
#
#     # print("{0:<32}: {1}".format("train_id",train_id[0]) )
#     print("{0:<32}: {1}".format("sum of train_toxic_lbl:", np.sum(train_toxic_lbl)))
#     print("{0:<32}: {1}".format("sum of train_severe_toxic_lbl: ",
#                                 np.sum(train_severe_toxic_lbl)))
#     print("{0:<32}: {1}".format("sum of train_obscene_lbl: ", np.sum(train_obscene_lbl)))
#     print("{0:<32}: {1}".format("sum of train_threat_lbl: ", np.sum(train_threat_lbl)))
#     print("{0:<32}: {1}".format("sum of train_insult_lbl: ", np.sum(train_insult_lbl)))
#     print("{0:<32}: {1}".format("sum of train_identity_hate_lbl: ", np.sum(train_identity_hate_lbl)))
#     numbers_list = [int(np.sum(train_toxic_lbl)),
#                     int(np.sum(train_severe_toxic_lbl)),
#                     int(np.sum(train_obscene_lbl)),
#                     int(np.sum(train_threat_lbl)),
#                     int(np.sum(train_insult_lbl)),
#                     int(np.sum(train_identity_hate_lbl))
#                     ]
#     classes_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
#                     ]
#     plot_bar_chart((classes_list, numbers_list))
#     # print(df.describe(include=['int64']))
#
#     # print(train_id.shape)
#     # print(train_text.shape)
#     # print(train_toxic_lbl.shape)
#     # print(train_severe_toxic_lbl.shape)
#     # print(train_obscene_lbl.shape)
#     # print(train_threat_lbl.shape)
#     # print(train_insult_lbl.shape)
#     # print(train_identity_hate_lbl.shape)
#
#     tpl = load_test('test.csv', 'test_labels.csv', True)
#     test_id = tpl[0]
#     test_text = tpl[1]
#     test_toxic_lbl = tpl[2]
#     test_severe_toxic_lbl = tpl[3]
#     test_obscene_lbl = tpl[4]
#     test_threat_lbl = tpl[5]
#     test_insult_lbl = tpl[6]
#     test_identity_hate_lbl = tpl[7]
#
#     # print("test_id: ", test_id[0])
#
#     # print(df.describe(include=['int64']))
#
#     # print(test_id.shape)
#     # print(test_text.shape)
#     # print(test_toxic_lbl.shape)
#     # print(test_severe_toxic_lbl.shape)
#     # print(test_obscene_lbl.shape)
#     # print(test_threat_lbl.shape)
#     # print(test_insult_lbl.shape)
#     # print(test_identity_hate_lbl.shape)
#
#     median_training_num_of_words = get_num_words_per_sample(train_text.flatten())
#     median_test_num_of_words = get_num_words_per_sample(test_text.flatten())
#     # print("training num of words per sample: ", median_training_num_of_words)
#     # print("testing num of words per sample: ", median_test_num_of_words)
#
#     min_words, max_word = min_max_sample_length(train_text.flatten())
#     # print(min_words, max_word)
#
#     # plot_sample_length_distribution(train_data)
#     # plot_frequency_distribution_of_ngrams(train_data)
#     print(
#         f"S/W ratio: no of documents/ median num of words =  {len(train_text)} / {median_training_num_of_words} "
#         f"= {len(train_text) / median_training_num_of_words}")
#     lst = train_text.tolist()
#
#     print("shape of train_text: ", train_text.shape)
#     print("type of lst[0]: ", type(lst[0]))
#     print("type of lst[0]: ", type(lst[0]))
#     print("type of train_text[0]: ", type(train_text[0]))
#     print("shape of train_text[0]: ", train_text[0].shape)
#     print(train_text[0])
#     vectorizer = CountVectorizer()
#
#     vectorized_texts = vectorizer.fit_transform(train_text[:3, 0])
#     # print(vectorizer.vocabulary_)
#     print(train_id[:3], train_text[:3], vectorized_texts)
#     print(type(vectorized_texts))
#     for entry in vectorized_texts.data:
#         print(entry)


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[^%s]' % (string.ascii_letters + string.whitespace),
                                    '')


# ,output_sequence_length=sequence_length)

def build_model(vectorize_layer, max_features, embedding_dim):
    model = tf.keras.Sequential([layers.Input(shape=(1,), dtype=tf.string),
                                 vectorize_layer,
                                 layers.Embedding(max_features, embedding_dim, name="embedding"),
                                 layers.Dropout(0.2),
                                 layers.GlobalAveragePooling1D(),
                                 layers.Dropout(0.2),
                                 layers.Dense(1, activation='sigmoid')])

    # model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
    #               optimizer='adam',
    #               metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy',
    #     metrics=['binary_accuracy'],
    # )

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    return model


def run2(data_dir, models_dir, train_record_count=-1, test_record_count=-1, shuffle_data=True, remove_not_toxic=False):
    max_features = 20000
    # sequence_length = 250

    # vocab_size = 10000
    sequence_length = 100
    embedding_dim = 100
    epochs = 10

    # vectorize_layer = layers.TextVectorization(
    #     standardize=custom_standardization,
    #     max_tokens=max_features, pad_to_max_tokens=True,
    #     output_mode='tf_idf')

    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    # def vectorize_text(text, label):
    #     text = tf.expand_dims(text, -1)
    #     return vectorize_layer(text), label

    if not os.path.exists(models_dir):  # or len(os.listdir(models_dir)) != 6:
        print("generating toxic_models ...")
        # print('[%s]' % string.punctuation)
        tpl = load_train(data_dir, 'train.csv', shuffle_data, remove_not_toxic, train_record_count)
        train_train_id = tpl[0]
        valid_train_id = tpl[1]
        # train_text = tpl[1]
        toxic_train_dataset = tpl[2]
        toxic_valid_dataset = tpl[3]
        severe_toxic_train_dataset = tpl[4]
        severe_toxic_valid_dataset = tpl[5]
        obscene_train_dataset = tpl[6]
        obscene_valid_dataset = tpl[7]
        threat_train_dataset = tpl[8]
        threat_valid_dataset = tpl[9]
        insult_train_dataset = tpl[10]
        insult_valid_dataset = tpl[11]
        hate_train_dataset = tpl[12]
        hate_valid_dataset = tpl[13]
        train_text = train_train_id.map(lambda x, y: y)
        vectorize_layer.adapt(train_text)

        # print("toxic_train_dataset", toxic_train_dataset)
        # txtitr = iter(toxic_train_dataset)
        # text_batch, label_batch = next(txtitr)
        # first_review, first_label = text_batch[0], label_batch[0]
        #
        # print("Review++", first_review)
        # print("standard Review++", custom_standardization(first_review))
        # print("Label++", first_label)
        # for i in range(6 - 1):
        #     text_batch, label_batch = next(txtitr)
        #
        #     first_review, first_label = text_batch[0], label_batch[0]
        #     print("Review--", first_review)
        #     print("standard Review--", custom_standardization(first_review))
        #
        #     print("Label--", first_label)

        # print("Review", first_review)
        # print("Label", first_label)
        # print("Vectorized review", vectorize_text(first_review, first_label))

        # print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
        # print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
        print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
        # print('Vocabulary : {}'.format(type(vectorize_layer.get_vocabulary())))
        # pickle.dump(vectorize_layer.get_vocabulary(), open('vocab.dat', 'wb'))
        # pickle.dump(vectorize_layer.(), open('vocab.dat', 'wb'))
        # toxic_train_ds = toxic_train_dataset.map(vectorize_text)
        # print("toxic_test_ds", next(iter(toxic_train_ds)))

        # toxic_train_ds = toxic_train_ds.cache().prefetch(buffer_size=AUTOTUNE)



        toxic_model = build_model(vectorize_layer, max_features, embedding_dim)

        severe_toxic_model = build_model(vectorize_layer, max_features, embedding_dim)
        obscene_model = build_model(vectorize_layer, max_features, embedding_dim)
        threat_model = build_model(vectorize_layer, max_features, embedding_dim)
        insult_model = build_model(vectorize_layer, max_features, embedding_dim)
        hate_model = build_model(vectorize_layer, max_features, embedding_dim)

        # print(toxic_model.summary())

        history = toxic_model.fit(toxic_train_dataset,
                                  validation_data=toxic_valid_dataset,
                                  epochs=epochs)
        save_model(model=toxic_model, filepath='saved_models/toxic_model')

        history = severe_toxic_model.fit(severe_toxic_train_dataset,
                                         validation_data=severe_toxic_valid_dataset,
                                         epochs=epochs)
        save_model(model=severe_toxic_model, filepath='saved_models/severe_toxic_model')
        history = obscene_model.fit(obscene_train_dataset,
                                    validation_data=obscene_valid_dataset,
                                    epochs=epochs)
        save_model(model=obscene_model, filepath='saved_models/obscene_model')

        history = threat_model.fit(threat_train_dataset,
                                   validation_data=threat_valid_dataset,
                                   epochs=epochs)
        save_model(model=threat_model, filepath='saved_models/threat_model')
        history = insult_model.fit(insult_train_dataset,
                                   validation_data=insult_valid_dataset,
                                   epochs=epochs)
        save_model(model=insult_model, filepath='saved_models/insult_model')

        history = hate_model.fit(hate_train_dataset,
                                 validation_data=hate_valid_dataset,
                                 epochs=epochs)
        save_model(model=hate_model, filepath='saved_models/hate_model')

        weights = toxic_model.get_layer('embedding').get_weights()[0]
        vocab = vectorize_layer.get_vocabulary()
        out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
        out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

        for index, word in enumerate(vocab):
            if index == 0:
                continue  # skip 0, it's padding.
            vec = weights[index]
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
            out_m.write(word + "\n")
        out_v.close()
        out_m.close()

        # toxic_model.add(vectorize_layer)

        # tpl_test = load_test('test.csv', 'test_labels.csv', False)
        # test_text = tpl_test[1]
        # test_toxic_lbl = tpl_test[2]
        # toxic_test_dataset = tf.data.Dataset.from_tensor_slices((test_text, test_toxic_lbl))
        # toxic_test_ds = toxic_test_dataset.map(vectorize_text)
        # toxic_test_ds = toxic_test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # loss, accuracy = toxic_model.evaluate(toxic_test_ds)
        # print("Loss: ", loss)
        # print("Accuracy: ", accuracy)


    else:
        toxic_model = load_model('saved_models/toxic_model')
        severe_toxic_model = load_model('saved_models/severe_toxic_model')
        obscene_model = load_model('saved_models/obscene_model')
        insult_model = load_model('saved_models/insult_model')
        threat_model = load_model('saved_models/threat_model')
        hate_model = load_model('saved_models/hate_model')
        # vocab = pickle.load(open('vocab.dat', 'rb'))
        # vectorize_layer.set_vocabulary(vocab)
        print("all models were loaded")
        # print(toxic_model.summary())
        # vectorize_layer = toxic_model.pop()

    # tpl_test = load_test('test.csv', 'test_labels.csv', False)
    tpl_test = load_test(data_dir, 'test.csv', 'test_labels.csv', False, False, test_record_count)
    test_id = tpl_test[0]
    test_text = tpl_test[1]
    # print("test_text[0]",test_text[0])
    test_toxic_lbl = tpl_test[2]
    # print(test_toxic_lbl[0])
    toxic_test_dataset = tf.data.Dataset.from_tensor_slices((test_text, test_toxic_lbl))
    # x = next(iter(toxic_test_dataset))
    # print("toxic_test_dataset.take(1)",x[1])
    # print("vectorize_layer.is_adapted:",vectorize_layer.is_adapted)
    # print("vectorize_text():", vectorize_text(x[0],x[1]))
    # toxic_test_ds = toxic_test_dataset.map(vectorize_text)
    # print("toxic_test_ds",next(iter(toxic_test_ds)))
    # toxic_test_ds = toxic_test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    predictions = toxic_model.predict(toxic_test_dataset)
    # print(predictions.shape)
    # print("toxic_model performance:")
    # print("Loss: ", loss,"Accuracy: ", accuracy)
    predictions = np.hstack((predictions, severe_toxic_model.predict(toxic_test_dataset)))
    # print("severe_toxic_model performance:")
    # print("Loss: ", loss,"Accuracy: ", accuracy)
    predictions = np.hstack((predictions, obscene_model.predict(toxic_test_dataset)))
    # print("obscene_model performance:")
    # print("Loss: ", loss,"Accuracy: ", accuracy)

    predictions = np.hstack((predictions, threat_model.predict(toxic_test_dataset)))
    # print("threat_model performance:")
    # print("Loss: ", loss,"Accuracy: ", accuracy)
    predictions = np.hstack((predictions, insult_model.predict(toxic_test_dataset)))
    # print("insult_model performance:")
    # print("Loss: ", loss,"Accuracy: ", accuracy)
    predictions = np.hstack((predictions, hate_model.predict(toxic_test_dataset)))
    # print("hate_model performance:")
    # print("Loss: ", loss,"Accuracy: ", accuracy)
    # print(predictions.shape)
    # ,,,,,
    output = pd.DataFrame({'id': test_id.flatten(),
                           'toxic': predictions[:, 0],
                           'severe_toxic': predictions[:, 1],
                           'obscene': predictions[:, 2],
                           'threat': predictions[:, 3],
                           'insult': predictions[:, 4],
                           'identity_hate': predictions[:, 5]
                           })
    output.to_csv('submission.csv', index=False)


# tpl_test = load_test('test.csv', 'test_labels.csv', True, True)
run2(data_dir='data', models_dir='saved_models',
     train_record_count=1000, test_record_count=-1, shuffle_data=False, remove_not_toxic=True)

# array(['a', 'b', 'a'], dtype=object)

# x = np.array([[1], [2], [3], [4], [5], [6]])
# y = np.array([[1], [2], [3], [4], [5], [6]])
#
# # print(x.shape)
# print("before shuffle_data \n X: \n", x)
# print("Y: \n", y)
# shuffle_data((x, y))
# # #
# #
# # np.random.seed(123)
# # np.random.shuffle_data(x)
# # np.random.seed(123)
# # np.random.shuffle_data(y)
# # # random.shuffle_data(x)
#
# print("after shuffle_data \nX: \n", x)
# print("Y: \n", y)

#
# train_id:  ['86e395fd449fac16']
# test_id:  ['b44d785af014a0b7']
