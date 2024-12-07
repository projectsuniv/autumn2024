import os;

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Embedding, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
# from google.colab import drive
import csv
import spacy

en = spacy.load("en_core_web_sm")

import seaborn

import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

print(tf.version.VERSION)

# drive.mount('/content/drive', force_remount=True)
if not os.path.exists('dataset_merged.csv'):
    df1 = pd.read_csv('WELFake_Dataset.csv')
    df2 = pd.read_csv('news_articles.csv')[['title', 'text', 'label']]
    df2['label'] = df2['label'] == 'Real'
    df = pd.concat([df1, df2])
    df.dropna(inplace=True)
    print("merge files")
    df.to_csv('dataset_merged.csv')
else:
    print("load merged")
    df = pd.read_csv('dataset_merged.csv')

plt.figure(figsize=(20, 6))
seaborn.histplot(df['text'].apply(lambda x: len(x.split())), bins=range(1, 3000, 50), alpha=0.8)
plt.ylabel("кількість текстів")
plt.xlabel("довжина тексту")
plt.show()

# all_tokens = []
# for text in tqdm(df['text']):
#     tokens = word_tokenize(text)
#     all_tokens.extend(tokens)
# stopwords = en.Defaults.stop_words
# fake_tokens = [token for token, label in zip(all_tokens, df['label']) if
#                label == 0]
# real_tokens = [token for token, label in zip(all_tokens, df['label']) if
#                label == 1]
# fdist_fake = FreqDist(fake_tokens)
# fdist_real = FreqDist(real_tokens)
# top_fake_terms = fdist_fake.most_common(100)
#
# print("Top 10 terms for fake news:")
# for term, frequency in top_fake_terms:
#     if term.lower() not in stopwords and any(c.isalpha() for c in term):
#         print(term, ":", frequency)
#
# top_real_terms = fdist_real.most_common(100)
# print("Top 10 terms for real news:")
# for term, frequency in top_real_terms:
#     if term.lower() not in stopwords and any(c.isalpha() for c in term):
#         print(term, ":", frequency)

nlp = spacy.load("en_core_web_sm")


def remove_proper_names(description):
    text = description["text"]
    if text and type(text) == str:
        doc = nlp(text)
        for ent in doc.ents:
            text = text.replace(ent.text, ent.label_)
        description["text"] = text

    title = description["title"]
    if title and type(title) == str:
        doc = nlp(title)
        for ent in doc.ents:
            title = title.replace(ent.text, ent.label_)

            description["title"] = title
    return description


if not os.path.exists('merged_dataset_proper_nouns_scrubbed.csv'):
    print("creating merged_dataset_proper_nouns_scrubbed.csv")
    tqdm.pandas()
    df_no_proper = df.parallel_apply(remove_proper_names, axis=1, result_type='expand')
    df_no_proper.to_csv('merged_dataset_proper_nouns_scrubbed.csv')
    print("saved merged_dataset_proper_nouns_scrubbed.csv")

TRAIN_SHARE, TEST_SHARE, VALIDATION_SHARE = 0.9, 0.05, 0.05
train_len = int(df.shape[0] * TRAIN_SHARE)
test_len = int(df.shape[0] * TEST_SHARE)
val_len = int(df.shape[0] * VALIDATION_SHARE)

x, x_test, y, y_test = train_test_split(df[['title', 'text']], df['label'], test_size=0.1, train_size=0.9)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5, train_size=0.5)

N_WORDS = 500
MAX_LEN = 512
tokenizer = Tokenizer(num_words=N_WORDS, oov_token="<OutOfVocabulary>")
tokenizer.fit_on_texts(x_train['text'])
word_index = tokenizer.word_index

# Naive Bayesian bag-of-words classifier as a baseline
vectorizer = CountVectorizer(max_features=N_WORDS)
X = vectorizer.fit_transform(df['text']).toarray()
X_train, X_test, Y_train, Y_test = train_test_split(X, df['label'], test_size=0.1, random_state=42)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, Y_train)

y_pred = naive_bayes.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)


def get_sequences(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded


def plot_history(history, metric):
    plt.plot([.5] + history.history[metric])
    plt.plot([.5] + history.history['val_' + metric], '')
    plt.xlabel("Епохи")
    plt.ylabel(metric)
    plt.ylim([0, 1])
    plt.legend([('Точність' if metric == 'accuracy' else 'Штраф'),
                ('Точність' if metric == 'accuracy' else 'Штраф') + ' (датасет валідації)'])


train_seqs = get_sequences(x_train['text'])
test_seqs = get_sequences(x_test['text'])
val_seqs = get_sequences(x_val['text'])

if not os.path.exists('dense-16-6-1-larger-dataset-proper-excluded.keras'):
    model = tf.keras.Sequential([
        Embedding(N_WORDS, 16, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(6, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_seqs,
                        y_train.astype('float32'),
                        epochs=10,
                        validation_data=(val_seqs.astype('float32'),
                                         y_val.astype('float32')))

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_history(history, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_history(history, 'loss')

    model.evaluate(test_seqs.astype('float32'), y_test.astype('float32'))

    model.save('dense-16-6-1-larger-dataset-proper-excluded.keras')

if not os.path.exists('dense-64-4-1-larger-dataset-proper-excluded.keras'):
    model_dense_larger = tf.keras.Sequential([
        Embedding(N_WORDS, 64, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_dense_larger.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_dense_larger.summary()

    history_dense = model_dense_larger.fit(train_seqs,
                                           y_train.astype('float32'),
                                           epochs=10,
                                           validation_data=(val_seqs.astype('float32'),
                                                            y_val.astype('float32')))

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_history(history_dense, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_history(history_dense, 'loss')

    model_dense_larger.evaluate(test_seqs.astype('float32'), y_test.astype('float32'))
    # model_dense_larger.save('dense-64-4-1-larger-dataset-proper-excluded.h5')
    model_dense_larger.save('dense-64-4-1-larger-dataset-proper-excluded.keras')

if not os.path.exists('lstm-larger-dataset-proper-excluded.keras'):
    model_lstm = tf.keras.Sequential([
        Input(name='input', shape=[MAX_LEN]),
        Embedding(N_WORDS, 16),
        Bidirectional(tf.keras.layers.LSTM(4, return_sequences=True)),
        Bidirectional(tf.keras.layers.LSTM(2)),
        Dense(2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                       optimizer=tf.keras.optimizers.Adam(1e-4),
                       metrics=['accuracy'])
    model_lstm.summary()

    history_lstm = model_lstm.fit(train_seqs,
                                  y_train.astype('float32'),
                                  epochs=10,
                                  validation_data=(val_seqs, y_val.astype('float32')),
                                  batch_size=64,

                                  callbacks=[EarlyStopping(monitor='val_accuracy', mode='max', patience=3,
                                                           verbose=False, restore_best_weights=True)]
                                  )

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_history(history_lstm, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_history(history_lstm, 'loss')

    model_lstm.evaluate(test_seqs, y_test.astype('float32'))

    model_lstm.save('lstm-larger-dataset-proper-excluded.keras')

if not os.path.exists('lstm-larger-larger-dataset-proper-excluded.keras'):
    model_lstm_largel = tf.keras.Sequential([
        Input(name='input', shape=[MAX_LEN]),
        Embedding(N_WORDS, 32),
        Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        Bidirectional(tf.keras.layers.LSTM(4)),
        Dense(8, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model_lstm_largel.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                              optimizer=tf.keras.optimizers.Adam(1e-4),
                              metrics=['accuracy'])
    model_lstm_largel.summary()

    history_lstm_larger = model_lstm_largel.fit(train_seqs,
                                                y_train.astype('float32'),
                                                epochs=10,
                                                validation_data=(val_seqs, y_val.astype('float32')),
                                                batch_size=64,

                                                callbacks=[EarlyStopping(monitor='val_accuracy', mode='max', patience=3,
                                                                         verbose=False, restore_best_weights=True)]
                                                )

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_history(history_lstm_larger, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_history(history_lstm_larger, 'loss')

    model_lstm_largel.evaluate(test_seqs, y_test.astype('float32'))
    model_lstm_largel.save('lstm-larger-larger-dataset-proper-excluded.keras')
