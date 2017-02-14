"""
Data Reader and preprocessor for SQuAD DATASET

reference:
https://rajpurkar.github.io/SQuAD-explorer/
"""

import json
import operator
from collections import defaultdict
from nltk.tokenize import WordPunctTokenizer
from keras.preprocessing import sequence
import numpy as np

class SquadReader:

    def __init__(self, train_path, test_path, max_vocabulary=50000, base_word_id=2):
        """
        :param train_filename:
        :param test_filename:
        :param max_vocabulary: select the top K words in the vocabulary and the rest replace by id 0
        """

        self.max_vocabulary = max_vocabulary
        self.tokenize = WordPunctTokenizer().span_tokenize    # return start and end of each word in an iterator

        self.word_index = None                              # { word : word_index_sorted_by_occurence } # starting from 2  1 reserved for self.oov_id and zero for padding.
        self.pad_id = 0
        self.oov_id = 1
        self.base_word_id = base_word_id
        self.inverse_word_index = None                      # initialized with fit it's another version of the dictionary that is the size of max vocab size
        self.word_counts = None                             # { Word_index: count }
        self.train_path = train_path
        self.test_path = test_path
        self.trimmed_word_index = None                      # initialized with fit it's another version of the dictionary that is the size of max vocab size




    def load_dataset(self, filename):
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
        for doc in dataset:
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

    def fit(self, docs):
        """
        :param docs:

        following keras.preprocessing.text.Tokenizer method of indexing and tokenizing
        we follow the same fit function but keeping in regard the preprocessing not
        to harm the "s" attribute the start of the answer position in the dataset.
        Better handling unicode.
        - load datasets train-dev
        - preprocessing

        :return:
        """

        word_counts = defaultdict(lambda: 1)
        # build word index out of all questions and answers of training and test sets

        for doc in docs:
            for s, e in self.tokenize(doc):
                w = doc[s:e]
                word_counts[w] += 1

        word_counts = dict(word_counts)
        word_counts = sorted(word_counts.items(), reverse=True, key=operator.itemgetter(1))

        self.word_index = dict()
        self.word_index["__PADDING__"] = self.pad_id
        self.word_index["__OOV_WORD__"] = self.oov_id
        self.word_counts = dict()

        for i, (w, c) in enumerate(word_counts):
            self.word_index[w] = i+ 1 + self.base_word_id
            self.word_counts[i+1] = c

        self.inverse_word_index = {v: k for k, v in self.word_index.iteritems()}

        tmp = sorted(self.word_index.items(), key=operator.itemgetter(1))
        tmp = tmp[:self.max_vocabulary]
        self.trimmed_word_index = dict(tmp)

    def preprocess_item(self, (p, q, s, a), max_vocabulary=None):
        """
        :param p: paragraph string
        :param q: question string
        :param s: char start index of the answer
        :param a: answer text
        :return:
         pp:  preprocessed context into word_index list   ( 0 if out of vocab )
         qpp: preprocessed question into word_index list   ( 0 if out of vocab )
         awid: answer start word id of the answer in the paragraph after tokenizing
         a:  answer text
        """
        # max vocab not set use the whole vocabulary
        if max_vocabulary is None:
            max_vocabulary = len(self.word_index)

        pp = []
        qpp = []
        awid = None
        for wi, (cs, ce) in enumerate(self.tokenize(p)):  # wid , char start, char end
            if s == cs:
                awid = wi

            w = p[cs:ce]
            if w in self.word_index and self.word_index[w] < max_vocabulary:
                pp.append(self.word_index[w])
            else:
                pp.append(self.oov_id)

        for wi, (cs, ce) in enumerate(self.tokenize(q)):
            w = q[cs:ce]
            if w in self.word_index and self.word_index[w] < max_vocabulary:
                qpp.append(self.word_index[w])
            else:
                qpp.append(self.oov_id)

        return pp, qpp, awid, a

    def transform(self, x):
        """
        :param x: (P, Q, S, A) as got from self.load_dataset()
        :return:

            tokenize P and Q
            change S to be word index instead of
            add them to self.train_pp and self.train_pp
            set self.train_pp  & self.test.pp
            self.train_pp  = (P, Q, Wid, A)    : P: Paragraph tokens list
                                                 Q: Question tokens list
                                                 Wid: list of answer starts
                                                 A: answer string (non-tokenized)
        """

        return zip(*[self.preprocess_item(i, self.max_vocabulary) for i in zip(*x)])

    def prepare_train_dev(self):

        train = self.load_dataset(self.train_path)
        test = self.load_dataset(self.test_path)
        self.fit(train[0] + train[1] + test[0] + test[1])

        P_train, Q_train, Aindx_train, A_train = self.transform(train)
        P_test, Q_test, Aindx_test, A_test = self.transform(test)

        # preprocessing
        P_train = sequence.pad_sequences(P_train, padding='post', value=self.pad_id)
        Q_train = sequence.pad_sequences(Q_train, padding='post', value=self.pad_id)
        P_test = sequence.pad_sequences(P_test, padding='post', value=self.pad_id)
        Q_test = sequence.pad_sequences(Q_test, padding='post', value=self.pad_id)

        # because answers index sometimes can't match the tokenizer
        P_train = np.array([i for c, i in enumerate(P_train) if Aindx_train[c] is not None])
        P_test = np.array([i for c, i in enumerate(P_test) if Aindx_test[c] is not None])
        Q_train = np.array([i for c, i in enumerate(Q_train) if Aindx_train[c] is not None])
        Q_test = np.array([i for c, i in enumerate(Q_test) if Aindx_test[c] is not None])
        A_train = np.array([i for c, i in enumerate(A_train) if Aindx_train[c] is not None])
        A_test = [i for c, i in enumerate(A_test) if Aindx_test[c] is not None]
        Aindx_train = np.array([i for c, i in enumerate(Aindx_train) if Aindx_train[c] is not None])
        Aindx_test = np.array([i for c, i in enumerate(Aindx_test) if Aindx_test[c] is not None])

        MAX_SEQ_WORD_LENGTH = len(P_train[0])

        # create y as one hot vector
        A_onehot_word_train = np.zeros((Aindx_train.shape[0], MAX_SEQ_WORD_LENGTH))
        for c, i in enumerate(Aindx_train):
            A_onehot_word_train[c, i] = 1

        A_onehot_word_test = np.zeros((Aindx_test.shape[0], MAX_SEQ_WORD_LENGTH))
        for c, i in enumerate(Aindx_test):
            A_onehot_word_test[c, i] = 1

        return [[P_train, Q_train, A_train, Aindx_train, A_onehot_word_train],[P_test, Q_test, A_test, Aindx_test,A_onehot_word_test]]