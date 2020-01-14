import nltk
from corpus_mails import mails
import random
import math

def get_TP_FP_TN_FN(gold, predicted, class_label='spam'):
    tp = sum([
        1 if (gold[i] == class_label) and (predicted[i] == class_label)
        else 0
        for i in range(len(gold))
    ])

    fp = predicted.count(class_label) - tp

    fn = gold.count(class_label) - tp

    tn = len(gold) - (tp + fp + fn)

    return tp, fp, tn, fn

def get_precision(tp, fp):

    return tp/(tp+fp)

def get_recall(tp, fn):

    return tp/(tp+fn)

def get_f1(recall, precision):

    return 2*((recall*precision)/(recall+precision))


def compute_PRF(gold, predicted, class_label='spam'):

    # get tp, fp, tn, fn
    tp, fp, _, fn = get_TP_FP_TN_FN(gold, predicted,class_label)

    precision = get_precision(tp,fp)

    recall = get_recall(tp, fn)

    f1 = get_f1(recall, precision)

    return precision, recall, f1

# Function which returns a frequency distribution of words which can
# be found in the list 'include'. All words are transformed to
# lowercase
def get_fdist(words, include=[]):
    if len(include) > 0:
        return nltk.probability.FreqDist(
            [word.lower() for word in words if word.lower() in include]
        )
    else:
        return nltk.probability.FreqDist(
            [word.lower() for word in words]
        )

if __name__ == '__main__':
    # Define test values for testing precision, recall and f1 calculation
    test_gold = [1,1,1,1,0]
    test_prediction = [1,1,0,0,0]

    # test tp, fp, tn, fn calculation
    print(get_TP_FP_TN_FN(test_gold, test_prediction, class_label=1))

    # test precision, recall, f1 calculation
    print(compute_PRF(test_gold, test_prediction, class_label=1))

    # test correct import of mails corpus
    print(mails.categories())
    print(len(mails.fileids('spam')))
    print(len(mails.fileids('nospam')))

    # Generate list of tuples of mails and category
    mail_data = [
        (mails.words(fileid), category)
        for category in mails.categories()
        for fileid in mails.fileids(category)
    ]

    # Shuffle mail_data
    random.seed(1234)
    random.shuffle(mail_data)

    print(len(mail_data))

    # Define train/dev/test splits
    train_split = 0.8
    dev_split = 0.1

    # Split the dataset
    train_data = mail_data[:math.floor(train_split*len(mail_data))]
    dev_data = mail_data[
        math.ceil(train_split*len(mail_data))
        : math.floor(((train_split+dev_split) * len(mail_data)))
    ]
    test_data = mail_data[
        math.ceil(((train_split + dev_split) * len(mail_data)))
        :
    ]

    train_data_fd = [
        (get_fdist(words), category)
        for words, category in train_data
    ]

    dev_data_fd = [
        (get_fdist(words), category)
        for words, category in dev_data
    ]

    # Train naive Bayes classifier
    classifier = nltk.NaiveBayesClassifier.train(train_data_fd)

    # Evaluate classifier
    dev_prediction = classifier.classify_many(
        [dev_data_fd[i][0] for i in range(len(dev_data_fd))]
    )

    dev_gold = [dev_data_fd[i][1] for i in range(len(dev_data_fd))]

    print(compute_PRF(dev_gold, dev_prediction))
