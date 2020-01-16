import nltk
from corpus_mails import mails
import random
import math

'''
Homework 10
We use a simple Naive Bayes Classifier that is trained with simple unigram
features. Each feature indicates the presence of a specific token (no counts).
It achieves a precision of 0.90, recall of 1 and F1 of 0.95
We had two ideas to alter the spam mails.
    1.  Modify spam-specific words by inserting spaces in these words.
        This will "mask" the words for the classifier, but keeps their 
        readability.
    2.  Insert words which are specific for no-spam mails at the end of the
        spam mails. This leads to the presence of additional, no-spam-specific
        features while not altering the content of the mail itself.
        
Results
Evaluation of the unaltered test data (PRF): 
(0.8717948717948718, 1.0, 0.9315068493150686)

Evaluation of test data with space insertions and appended nospam-specific words(PRF): 
(0.6153846153846154, 0.23529411764705882, 0.3404255319148936)

Experiments on the development set showed that the combination of the two
techniques gives the strongest reduction in classification scores. The 
use of technique 1 alone produces no change in classification scores.
The reduction in classification scores is even stronger on the test set.

Only the spam mails are altered, which means that the classifier 
recognizes fewer emails as spam, but does not change its classification
for the no-spam mails. Hence, the recall value drops farther than the
precision value. (Recall only considers mails that are classified as 
spam in the gold data (tp+fn), while precision also considers nospam
mails wrongly classified as spam.
'''

# Function to calculate true positives etc from a list of gold labels and
# a list of predicted labels
def get_TP_FP_TN_FN(gold, predicted, class_label='spam'):
    # Get the true positives as the number times class_label is present in
    # both the predicted labels and the gold labels
    tp = sum([
        1 if (gold[i] == class_label) and (predicted[i] == class_label)
        else 0
        for i in range(len(gold))
    ])

    # Get the false positives as the total number of class_label instances
    # in the predicted labels minus the number of true positives
    fp = predicted.count(class_label) - tp

    # Get the false negatives as the total number of class_label instances
    # in the gold labels minus the number of true positives
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

# Function which replaces certain tokens in a list with a slightly altered
# version of the token. In this altered version, a space is inserted so that
# it will be counted as a different token in the frequency distribution of the
# list.
# Inputs: list of tokens to be altered and a second list which
# provides the patterns that are altered in the first list.
# Returns an altered version of the first list
def insert_spaces(words, patterns):
    new_words = []
    # Iterate over all tokens in words list
    for i, token in enumerate(words):
        # Check if any of the patterns match
        if any([(pattern in token) for pattern in patterns]):
            # If yes, insert spaces into the token and append to new list of tokens
            new_words.append(' '.join(token))
        else:
            new_words.append(token)

    return new_words

# Function which adds n non-spam specific tokens at the end of a spam text
def add_words(words, words_to_add, n):
    # Iterate over all words to be added
    for word_to_add in words_to_add:
        # Add each word to add n times
        for i in range(n):
            words.append(word_to_add)
    return words

# Function which gets the unigram features from a list of tokens
# (Indicates the presence of a token)
def get_unigram_features(words):
    return {
        word : 1
        for word in set(words)
    }

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
    print()

    # Generate list of tuples of mails and category
    mail_data = [
        (mails.words(fileid), category)
        for category in mails.categories()
        for fileid in mails.fileids(category)
    ]

    # Shuffle mail_data
    random.seed(1234)
    random.shuffle(mail_data)

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

    # Generate the unigram features for the train and
    # development set
    train_data_uf = [
        (get_unigram_features(words), category)
        for words, category in train_data
    ]

    dev_data_uf = [
        (get_unigram_features(words), category)
        for words, category in dev_data
    ]

    # Train naive Bayes classifier
    classifier = nltk.NaiveBayesClassifier.train(train_data_uf)

    # Evaluate classifier
    dev_prediction = classifier.classify_many(
        [dev_data_uf[i][0] for i in range(len(dev_data_uf))]
    )

    # patterns =  ['$', 'dollar', 'money', 'credit', 'free', 'sex', 'viagra']

    # Words that are typical for non-spam mails (from most informative features
    # of classifier
    no_spam_words = [
        'german', 'scientific', 'cambridge', 'ling', 'differences',
        'approach', 'areas', 'department', 'authors', 'papers',
        'germany', 'talks', 'topics', 'notification', 'studies',
        'invited', 'william', 'appear', 'submit', 'academic',
        'sum', 'academic', 'february', 'speech', 'ii', 'aims',
        'university', 'appropriate', 'posted', 'aspects', 'speaker',
        'particularly', 'evidence*', 'issues', 'language',
        'languages', 'literature', 'professor', 'formal',
        'discussion', 'university', 'french', 'researchers',
        'summary', 'affiliation', 'linguistics', 'linguists',
        'references'
    ]

    # Alter the spam mails in the development set so that spaces are inserted
    # into words that appear in the most informative features of the classifier
    # Most of these appear more often in spam than in nospam mails
    dev_data_insert = [
        (insert_spaces(
            mail,
            [word for word, _ in classifier.most_informative_features(1000)]
        ), category)
        if category == 'spam'
        else (mail, category)
        for mail, category in dev_data
    ]

    # Generate the unigram features
    dev_data_insert_uf = [
        (get_unigram_features(words), category)
        for words, category in dev_data_insert
    ]

    # Predict the labels
    dev_insert_prediction = classifier.classify_many(
        [dev_data_insert_uf[i][0] for i in range(len(dev_data_insert_uf))]
    )

    # Get the gold labels
    dev_gold = [dev_data_uf[i][1] for i in range(len(dev_data_uf))]

    print(
        'Evaluation of unaltered dev data(PRF):',
        compute_PRF(dev_gold, dev_prediction)
    )

    print(
        'Evaluation of dev data with space insertions(PRF):',
        compute_PRF(dev_gold, dev_insert_prediction)
    )

    # Alter the space-inserted mails further by appending nospam-specific
    # words to their ends
    dev_data_insert_append = [
        (add_words(mail, no_spam_words, 1), category)
        if category == 'spam'
        else (mail, category)
        for mail, category in dev_data_insert
    ]

    # Generate unigram features
    dev_data_insert_append_uf = [
        (get_unigram_features(words), category)
        for words, category in dev_data_insert_append
    ]

    # Predict the labels
    dev_insert_append_prediction = classifier.classify_many(
        [dev_data_insert_append_uf[i][0] for i in range(len(dev_data_insert_append_uf))]
    )

    # Alter the original spam mails in the development set by appending nospam-
    # specific words to their ends
    dev_data_append = [
        (add_words(list(mail), no_spam_words, 1), category)
        if category == 'spam'
        else (mail, category)
        for mail, category in dev_data
    ]

    # Generate the unigram features
    dev_data_append_uf = [
        (get_unigram_features(words), category)
        for words, category in dev_data_append
    ]

    # Predict the labels
    dev_append_prediction = classifier.classify_many(
        [dev_data_append_uf[i][0] for i in range(len(dev_data_append_uf))]
    )

    print(
        'Evaluation of dev data with space insertions and appended nospam-specific words(PRF):',
        compute_PRF(dev_gold, dev_insert_append_prediction)
    )

    print(
        'Evaluation of dev data with appended nospam-specific words(PRF):',
        compute_PRF(dev_gold, dev_append_prediction)
    )

    #--------------------------------------------------------------------------
    # Evaluation of the test set
    # Generate unigram features from the test set
    test_data_uf = [
        (get_unigram_features(list(words)), category)
        for words, category in test_data
    ]

    # Predict the labels
    test_prediction = classifier.classify_many(
        [test_data_uf[i][0] for i in range(len(test_data_uf))]
    )

    # Alter the test set by inserting spaces
    test_data_insert = [
        (insert_spaces(
            mail,
            [word for word, _ in classifier.most_informative_features(1000)]
        ), category)
        if category == 'spam'
        else (mail, category)
        for mail, category in test_data
    ]

    # Alter the space-inserted mails further by appending nospam-specific
    # words to their ends
    test_data_insert_append = [
        (add_words(mail, no_spam_words, 1), category)
        if category == 'spam'
        else (mail, category)
        for mail, category in test_data_insert
    ]

    # Generate unigram features
    test_data_insert_append_uf = [
        (get_unigram_features(words), category)
        for words, category in test_data_insert_append
    ]

    # Predict the labels
    test_insert_append_prediction = classifier.classify_many(
        [test_data_insert_append_uf[i][0] for i in range(len(test_data_insert_append_uf))]
    )

    # Get the test data gold labels
    test_gold = [
        category for _, category in test_data
    ]

    print(
        'Evaluation of the unaltered test data (PRF):',
        compute_PRF(test_gold, test_prediction)
    )

    print(
        'Evaluation of test data with space insertions and appended nospam-specific words(PRF):',
        compute_PRF(test_gold, test_insert_append_prediction)
    )