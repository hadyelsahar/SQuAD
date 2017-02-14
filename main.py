"""
Reference:
A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task (Chen et al., 2016)
https://cs.stanford.edu/people/danqi/papers/acl2016.pdf
https://github.com/danqi/rc-cnn-dailymail/blob/master/code/nn_layers.py#L102
"""

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout

from nn_modules.attention import BilinearAttentionLayer, DotproductAttentionLayer
from utils.datareader import SquadReader
from utils.glove import Glove

TRAIN_PATH = "./data/train-v1.1.json"
TEST_PATH = "./data/dev-v1.1.json"
GLOVE_PATH = "./data/wordvec/glove.6B.100d.txt"

TOP_WORDS = 50000
EMB_VEC_LENGTH = 100
HIDDEN_SIZE = 256
N_EPOCHS = 5

reader = SquadReader(TRAIN_PATH, TEST_PATH, TOP_WORDS)

[[P_train, Q_train, A_train, Aindx_train, A_onehot_word_train],
 [P_test, Q_test, A_test, Aindx_test, A_onehot_word_test]] = reader.prepare_train_dev()

glove = Glove(GLOVE_PATH, reader.trimmed_word_index)


P_model = Sequential()
P_model.add(Embedding(TOP_WORDS, EMB_VEC_LENGTH, input_length=P_train.shape[1], mask_zero=True, weights=[glove.embedding_matrix], trainable=False))
P_model.add(Dropout(0.1))
P_model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))

Q_model = Sequential()
Q_model.add(Embedding(TOP_WORDS, EMB_VEC_LENGTH, mask_zero=True, weights=[glove.embedding_matrix], trainable=False))
Q_model.add(Dropout(0.1))
Q_model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=False)))

model = Sequential()
# model.add(DotproductAttentionLayer([P_model, Q_model]))
model.add(BilinearAttentionLayer([P_model, Q_model]))

model.add(Dense(P_train.shape[1], activation='softmax'))

# sgd = SGD(lr=0.1, clipnorm=10)
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit([P_train, Q_train], A_onehot_word_train, nb_epoch=N_EPOCHS, batch_size=32)

scores = model.evaluate([P_test, Q_test], A_onehot_word_test)
print("\n Model Accuracy: %.2f%%" % (scores[1]*100))
