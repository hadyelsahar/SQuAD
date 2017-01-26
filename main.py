"""
Reference:
A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task (Chen et al., 2016)
https://cs.stanford.edu/people/danqi/papers/acl2016.pdf
https://github.com/danqi/rc-cnn-dailymail/blob/master/code/nn_layers.py#L102
"""

import json
from __future__ import print_function
from keras.preprocessing.text import text_to_word_sequence
from glove import Glove
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from attention import BilinearAttentionLayer


TRAIN_PATH = "./data/train-v1.1.json"
TEST_PATH = "./data/dev-v1.1.json"
TOP_WORDS = 50000
EMB_VEC_LENGTH = 100
HIDDEN_SIZE = 128


# loading datasets
def load_dataset(filename):
    """
    the SQuAD dataset is only 29MB
    no problem in replicating the documents
    :param filename: file name of the dataset
    :return:
    """
    P = []  # contexts
    Q = []  # questions words
    S = []  # STARTS
    A = []  # ANSWERS

    dataset = json.load(open(filename))["data"]
    for doc in dataset :
        for paragraph in doc["paragraphs"]:
            p = paragraph['context']
            for question in paragraph['qas']:
                answers = {i['text']: i['answer_start'] for i in question['answers']}  # Take only unique answers
                q = question['question']
                for a in answers.items():
                    P.append(p)
                    Q.append(q)
                    A.append(a[0])
                    S.append(a[1])
    return P, Q, S, A


P_train, Q_train, S_train, A_train = load_dataset(TRAIN_PATH)
P_test, Q_test, S_test, A_test = load_dataset(TEST_PATH)

P_train = text_to_word_sequence(P_train, lower=False)
P_test = text_to_word_sequence(P_test, lower=False)
Q_train = text_to_word_sequence(Q_train, lower=False)
Q_test = text_to_word_sequence(Q_test, lower=False)


P_model = Sequential()
P_model.add(Embedding(TOP_WORDS, EMB_VEC_LENGTH))
P_model.add(Bidirectional(LSTM(HIDDEN_SIZE)))


