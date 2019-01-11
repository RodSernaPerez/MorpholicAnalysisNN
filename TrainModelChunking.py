"""Train CRF and BiLSTM-CRF on CONLL2000 chunking data, similar to https://arxiv.org/pdf/1508.01991v1.pdf.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy
from collections import Counter

from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Conv1D, Activation, Input, Concatenate, Merge, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.datasets import conll2000
from keras_contrib.utils import save_load_utils
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
import sys

import os
dir_path = os.path.dirname(os.path.realpath(__file__))


def classification_report(y_true, y_pred, labels):
    '''Similar to the one in sklearn.metrics, reports per classs recall, precision and F1 score'''
    y_true = numpy.asarray(y_true).ravel()
    y_pred = numpy.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s)
              for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format(
        '', 'recall', 'precision', 'f1-score', 'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report])
    N = len(y_true)
    print(formatter('avg / total',
                    sum(report2[0]) / N,
                    sum(report2[1]) / N,
                    sum(report2[2]) / N,
                    N) + '\n')


def plotConfusionMatrix(y_true, y_pred, labels, D):
    cnf_matrix = confusion_matrix(y_true, y_pred)

    y_true = y_true.tolist()
    y_pred = y_pred.tolist()

    for i in range(len(y_true)):
        y_true[i] = y_true[i][0]

    x = sorted(set(y_true + y_pred))
    labels = getFromDictionary(D, x)

    print_cm(cnf_matrix, labels)


def print_cm(cm, labels, hide_zeroes=False,
             hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end='')
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end='')
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end='')
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end='')
        print()


# Prepare Glove File
def readGloveFile(gloveFile):
    with open(gloveFile, 'r') as f:
        wordToGlove = {}  # map from a token (word) to a Glove embedding vector
        wordToIndex = {}  # map from a token to an index
        indexToWord = {}  # map from an index to a token

        for line in f:
            record = line.strip().split()
            token = record[0]  # take the token (word) from the text line
            # associate the Glove embedding vector to a that token (word)
            wordToGlove[token] = numpy.array(record[1:], dtype=numpy.float64)

        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            # 0 is reserved for masking in Keras (see above)
            kerasIdx = idx + 1
            wordToIndex[tok] = kerasIdx  # associate an index to a token (word)
            # associate a word to a token (word). Note: inverse of dictionary
            # above
            indexToWord[kerasIdx] = tok

    return wordToIndex, indexToWord, wordToGlove

# Create Pretrained Keras Embedding Layer


def createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, isTrainable):
    vocabLen = len(wordToIndex) + 1  # adding 1 to account for masking
    # works with any glove dimensions (e.g. 50)
    embDim = next(iter(wordToGlove.values())).shape[0]

    embeddingMatrix = numpy.zeros((vocabLen, embDim))  # initialize with zeros
    for word, index in wordToIndex.items():
        # create embedding: word index to Glove word embedding
        embeddingMatrix[index, :] = wordToGlove[word]

    embeddingLayer = Embedding(
        vocabLen,
        embDim,
        weights=[embeddingMatrix],
        trainable=isTrainable)
    return embeddingLayer


def loadDataSet(file_name):
    firstLine = True
    N = 0
    Matrix = []
    with open(file_name, "rt") as infile:
        for line in infile.readlines():
            pedazos = line.split("\t")

            if firstLine:
                N = len(pedazos)
                firstLine = False

            if(len(pedazos) == N):  # Hay algunas lineas que no estan completas, las quitamos
                element = [0] * N
                for i in range(0, N):
                    element[i] = pedazos[i].replace("\n", "")

                Matrix.append(element)
    return Matrix


def extractColoumn(matrix, i):
    return [row[i] for row in matrix]


def getDictionary(set):
    D = dict(zip(set, range(1, len(set) + 1)))
    return D


def getFromDictionary(D, set):
    x = []
    for n in set:
        x.append(D.get(n))

    return x


def convertToSentences(X, Y, value_dot, MAX_WORDS):

    sentence_x = []
    sentence_y = []

    numerSentences = 0
    X_sentences = numpy.ndarray(shape=(0, MAX_WORDS), dtype=int)
    Y_sentences = numpy.ndarray(shape=(0, MAX_WORDS), dtype=int)
    for i in range(0, len(X)):
        x = X[i]
        y = Y[i]
        sentence_x.append(x)
        sentence_y.append(y)
        if x == value_dot:
            if len(sentence_x) < MAX_WORDS:
                x_padded = numpy.asarray(
                    [0] * (MAX_WORDS - len(sentence_x)) + sentence_x)
                X_sentences = numpy.append(X_sentences, [x_padded], axis=0)
                y_padded = numpy.asarray(
                    [0] * (MAX_WORDS - len(sentence_y)) + sentence_y)
                Y_sentences = numpy.append(Y_sentences, [y_padded], axis=0)
                numerSentences = numerSentences + 1

            sentence_x = []
            sentence_y = []

    Y_sentences = numpy.reshape(Y_sentences, (numerSentences, MAX_WORDS, 1))
    return [X_sentences, Y_sentences]


def reverseDictionary(D):
    return {v: k for k, v in D.items()}


def convertToVectorOfChars(list_sentences, D, MAX_CHARS):

    NUMBER_SENTENCES = len(list_sentences)
    MAX_WORDS_IN_SENTENCE = len(list_sentences[0])

    D_inv = reverseDictionary(D)

    x = numpy.zeros(shape=(NUMBER_SENTENCES, MAX_WORDS_IN_SENTENCE, MAX_CHARS))

    for i in range(0, NUMBER_SENTENCES - 1):
        for j in range(0, MAX_WORDS_IN_SENTENCE - 1):
            word_index = list_sentences[i][j]
            if(word_index > 0):
                word = D_inv[word_index]
                if(len(word) < MAX_CHARS):
                    x[i, j, range(0, len(word))] = [ord(c) for c in word]
                else:
                    x[i, j, range(0, MAX_CHARS)] = [ord(c)
                                                    for c in word[:MAX_CHARS]]

    return x


# ------
# Parameters
# -----
EPOCHS = 15
EMBED_DIM = 200
PRE_TRAINED_WE = 0
numerOfModel = 1
BiRNN_UNITS = 200
MAX_CHARS = 15
MAX_WORDS = 25

# ------
# Data
# -----

# conll200 has two different targets, here will only use IBO like chunking as an example
#(train_x, _, train_y), (test_x, _, test_y), (vocab, _, class_labels) = conll2000.load_data()


M = loadDataSet(dir_path + "/data/conll2003/en/chunking/train.txt")
train_x_words = extractColoumn(M, 0)
train_y_labels = extractColoumn(M, 1)

M = loadDataSet(dir_path + "/data/conll2003/en/chunking/test.txt")
test_x_words = extractColoumn(M, 0)
test_y_labels = extractColoumn(M, 1)

class_labels = set(test_y_labels)  # Extrae los valores unicos
vocab = set(train_x_words + test_x_words)


dictionary_words = getDictionary(vocab)

train_x = getFromDictionary(dictionary_words, train_x_words)
test_x = getFromDictionary(dictionary_words, test_x_words)

dictionary_labels = getDictionary(class_labels)

train_y = getFromDictionary(dictionary_labels, train_y_labels)
test_y = getFromDictionary(dictionary_labels, test_y_labels)

print(format(len(class_labels)) + " different labels")
print(format(len(vocab)) + " different words")


[train_x, train_y] = convertToSentences(
    train_x, train_y, dictionary_words.get('.'), MAX_WORDS)

[test_x, test_y] = convertToSentences(
    test_x, test_y, dictionary_words.get('.'), MAX_WORDS)


print(train_x.shape)

# ------
# Glove
# -----
wordToIndex, indexToWord, wordToGlove = readGloveFile(
    dir_path + "/glove.6B.50d.txt")
if PRE_TRAINED_WE == 1:
    embedding = createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, False)
else:
    embedding = Embedding(
        len(vocab) + 1,
        EMBED_DIM,
        mask_zero=True)  # Random embedding
    # El tamagno va a ser len(vocab)+1 porque al poner mask_zero a True va a
    # agnadir el 0 como padding al diccionario


if numerOfModel == 1:
    # --------------
    # 1. Regular CRF
    # --------------

    print('==== training CRF ====')

    model = Sequential()
    model.add(embedding)
    crf = CRF(len(class_labels), sparse_target=True)
    model.add(crf)
    model.summary()

    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    print(type(train_x))
    print(train_x.shape)
    print(type(train_y))
    print(train_y.shape)
    print(type(test_x))
    print(test_x.shape)
    print(type(test_y))
    print(test_y.shape)
    print(train_y)
    model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        validation_data=[
            test_x,
            test_y])

    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]

    print('\n---- Result of CRF ----\n')

if numerOfModel == 2:
    # -------------
    # 2. BiLSTM-CRF
    # -------------

    print('==== training BiLSTM-CRF ====')

    model = Sequential()
    model.add(embedding)
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(class_labels), sparse_target=True)
    model.add(crf)
    model.summary()

    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        validation_data=[
            test_x,
            test_y])

    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]

    print('\n---- Result of BiLSTM-CRF ----\n')


if numerOfModel == 3:
    # -------------
    # 3. 2xBiLSTM-CRF
    # -------------

    print('==== 2xBiLSTM-CRF ====')

    model = Sequential()
    x = model.add(embedding)
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    model.add(Activation('relu'))
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(class_labels), sparse_target=True)
    model.add(crf)
    model.summary()

    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        validation_data=[
            test_x,
            test_y])

    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]

    print('\n---- Result of 2xBiLSTM-CRF ----\n')


if numerOfModel == 4:
    # -------------
    # 4. Character Emebdding + BiLSTM-CRF
    # -------------

    print('==== Character Emebdding + BiLSTM-CRF ====')

    train_x = convertToVectorOfChars(train_x, dictionary_words, MAX_CHARS)
    test_x = convertToVectorOfChars(test_x, dictionary_words, MAX_CHARS)

    model = Sequential()
    embedding = Embedding(255 + 1, 10, mask_zero=True)  # Random embedding
    x = model.add(TimeDistributed(embedding, input_shape=(None, MAX_CHARS)))
    x = model.add(TimeDistributed(Convolution1D(50, 5, 5, border_mode='same')))

    model.add(TimeDistributed(LSTM(BiRNN_UNITS, return_sequences=False)))

    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(class_labels), sparse_target=True)
    model.add(crf)
    model.summary()

    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        validation_data=[
            test_x,
            test_y])

    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]

    print('\n---- Result of Character Emebdding + BiLSTM-CRF ----\n')


classification_report(test_y_true, test_y_pred, class_labels)
plotConfusionMatrix(test_y_true, test_y_pred, class_labels,
                    reverseDictionary(dictionary_labels))


model.save('modelo.h5')  # creates a HDF5 file 'my_model.h5'
save_load_utils.save_all_weights(model, 'pesos.h5')

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
