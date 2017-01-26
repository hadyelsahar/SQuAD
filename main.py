"""
Reference:
A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task (Chen et al., 2016)
https://cs.stanford.edu/people/danqi/papers/acl2016.pdf
https://github.com/danqi/rc-cnn-dailymail/blob/master/code/nn_layers.py#L102
"""

import json
from keras.preprocessing.text import Tokenizer
from glove import Glove
from utils.datareader import  load_dataset
import numpy as np
np.random.seed(1337)  # for reproducibility
from nltk.tokenize import word_tokenize
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from modules.attention import BilinearAttentionLayer


TRAIN_PATH = "./data/train-v1.1.json"
TEST_PATH = "./data/dev-v1.1.json"
TOP_WORDS = 50000
EMB_VEC_LENGTH = 100
HIDDEN_SIZE = 128
N_EPOCHS = 3


P_train, Q_train, S_train, A_train = load_dataset(TRAIN_PATH)
P_test, Q_test, S_test, A_test = load_dataset(TEST_PATH)

tokenizer = Tokenizer(nb_words=TOP_WORDS, lower=False)
tokenizer.fit_on_sequences([i.split() for i in P_train] + [i.split() for i in P_test])
PP_train = sequence.pad_sequences([i.split() for i in P_train])
PP_test = sequence.pad_sequences([i.split() for i in P_test])

Q_train = [i.split() for i in Q_train]
Q_test = [i.split() for i in Q_test]

MAX_SEQ_LENGTH = len(PP_train[0])


P_model = Sequential()
P_model.add(Embedding(TOP_WORDS, EMB_VEC_LENGTH))
P_model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))


Q_model = Sequential()
Q_model.add(Embedding(TOP_WORDS, EMB_VEC_LENGTH))
Q_model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))


model = Sequential()
model.add(BilinearAttentionLayer([P_model, Q_model]))
model.add(Dense(MAX_SEQ_LENGTH, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, clipnorm=10), metrics=['accuracy'])

model.fit([PP_train, Q_train], S_train, nb_epoch=N_EPOCHS, batch_size=64)

#todo : complete the model with start and end
