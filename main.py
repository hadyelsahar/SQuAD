"""
Reference:
A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task (Chen et al., 2016)
https://cs.stanford.edu/people/danqi/papers/acl2016.pdf
https://github.com/danqi/rc-cnn-dailymail/blob/master/code/nn_layers.py#L102
"""

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

from modules.attention import BilinearAttentionLayer
from utils.datareader import SquadReader

TRAIN_PATH = "./data/train-v1.1.json"
TEST_PATH = "./data/dev-v1.1.json"
TOP_WORDS = 50000
EMB_VEC_LENGTH = 100
HIDDEN_SIZE = 128
N_EPOCHS = 3


reader = SquadReader(max_vocabulary=TOP_WORDS)
train = reader.load_dataset(TRAIN_PATH)
test = reader.load_dataset(TEST_PATH)
reader.fit(train[0] + train[1] + test[0] + test[1])


P_train, Q_train, Aindx_train, A_train = reader.transform(train)
P_test, Q_test, Aindx_test, A_test = reader.transform(test)

# preprocessing

P_train = sequence.pad_sequences(P_train)
P_test = sequence.pad_sequences(P_test)

MAX_SEQ_WORD_LENGTH = len(P_train[0])

# hack because answers index sometimes can't match the tokenizer
# todo : fix
P_train = [i for c, i in enumerate(P_train) if Aindx_train[c] is not None]
P_test = [i for c, i in enumerate(P_test) if Aindx_test[c] is not None]
Q_train = [i for c, i in enumerate(Q_train) if Aindx_train[c] is not None]
Q_test = [i for c, i in enumerate(Q_test) if Aindx_test[c] is not None]
A_train = [i for c, i in enumerate(A_train) if Aindx_train[c] is not None]
A_test = [i for c, i in enumerate(A_test) if Aindx_test[c] is not None]
Aindx_train = [i for c, i in enumerate(Aindx_train) if Aindx_train[c] is not None]
Aindx_test = [i for c, i in enumerate(Aindx_test) if Aindx_test[c] is not None]

# create y as one hot vector
y_train = []
for i in Aindx_train:
     o = [0] * MAX_SEQ_WORD_LENGTH
     o[i] = 1
     y_train.append(o)

y_test = []
for i in Aindx_test:
    o = [0] * MAX_SEQ_WORD_LENGTH
    o[i] = 1
    y_test.append(o)


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

model.fit([P_train, Q_train], y_train, nb_epoch=N_EPOCHS, batch_size=64)

#todo : complete the model with start and end
