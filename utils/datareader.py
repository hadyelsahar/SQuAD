"""
Data Reader and preprocessor for SQuAD DATASET

reference:
https://rajpurkar.github.io/SQuAD-explorer/
"""

import json
import operator
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer

class SquadReader:

    def __init__(self, max_vocabulary=50000):
        """

        :param train_filename:
        :param test_filename:
        :param max_vocabulary: select the top K words in the vocabulary and the rest replace by id 0
        """

        self.max_vocabulary = max_vocabulary
        self.tokenize = WordPunctTokenizer().span_tokenize    # return start and end of each word in an iterator

        self.word_index = None                              # { word : word_index_sorted_by_occurence } # starting from 1  0 reserved for UOV_WORD
        self.inverse_word_index = None
        self.word_counts = None                             # { Word_index: count }

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
        word_counts = sorted(word_counts.items(), key=operator.itemgetter(1))

        self.word_index = dict()
        self.word_index["__UOV_WORD__"] = 0
        self.word_counts = dict()

        for i, (w, c) in enumerate(word_counts):
            self.word_index[w] = i+1
            self.word_counts[i+1] = c

        self.inverse_word_index = {v: k for k, v in self.word_index.iteritems()}

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
            if w in self.word_index and self.word_counts[self.word_index[w]] < max_vocabulary:
                pp.append(self.word_index[w])
            else:
                pp.append(0)

        for cs, ce in self.tokenize(q):
            w = p[cs:ce]
            if w in self.word_index and self.word_counts[self.word_index[w]] < max_vocabulary:
                qpp.append(self.word_index[w])
            else:
                qpp.append(0)

        return pp, qpp, awid, a

    def transform(self, x):
        """
        :param X: (P, Q, S, A) as got from self.load_dataset()
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

        return zip(*[self.preprocess_item(i) for i in zip(*x)])

