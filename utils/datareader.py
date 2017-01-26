

# loading datasets
def load_dataset(filename, max_vocabulary, ):
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


