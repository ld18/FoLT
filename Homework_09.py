import nltk
import random
import math
from nltk.corpus import movie_reviews

# Homework 09
# Task 9.1
# (a)

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


# Function which takes a list of tokens, pos-tags it with the universal
# tagset and returns only the lowercased tokens tagged with the given
# pos-tags
def filter_words(words, pos_tags):
    return [
        token.lower()
        for token, tag in nltk.pos_tag(words, tagset='universal')
        if tag in pos_tags
    ]

# Function to get a dict of counts from a list of items to count and a list to
# count in
def get_counts(count_here, count_this):
    return {
        counted_item : count_here.count(counted_item)
        for counted_item in count_this
    }

# Function which counts
def get_tag_proportion_feature(text, counted_tag):
    return {
        'proportion_{counted_tag}'.format(counted_tag=counted_tag) : [
            tag for _, tag in nltk.pos_tag(text, tagset='universal')
        ].count(counted_tag)/len(text)
    }

# Function which return average sentence length feature
def get_avg_sent_length_feature(text):
    sent_count = 0
    # count all occurrences of '?', '!', '.'
    for sym in ['!', '?', '.']:
        sent_count += text.count(sym)

    return {
        'avg_sent_length' : len(text)/sent_count
    }

# function which returns average word length feature
def get_avg_word_length_feature(text):
    return {
        'avg_word_length': sum(len(token) for token in text) / len(text)
    }

def classify_get_wrong(feature_data, raw_data, classifier):
    # count wrongly classified examples
    wrong_count = 0

    # Make empty frequency distribution to count the number of wrongly
    # cllassified examples for each class individually
    wrong_classes = nltk.probability.FreqDist()

    # Iterate over all tuples of feature and class label
    for i, tup in enumerate(feature_data):
        # Predict the class using the trained classifier passed as
        # argument
        prediction = classifier.classify(tup[0])

        # Check if the prediction is correct
        if not (prediction == tup[1]):
            # If not, add 1 to the number of wrongly predicted examples
            wrong_count += 1

            # Add 1 to the count of wrongly predicted examples from the
            # class
            wrong_classes[tup[1]] += 1

            # Print for analysis
            print(
                'Wrong classification {}'.format(wrong_count)
            )
            print('Text:', raw_data[i][0])
            print('Features:', tup[0])

    # print and calculate the overall accuracy
    print('Accuracy:', ((len(feature_data)-wrong_count)/len(feature_data)))

    # Print the number of wrongly assigned examples for each class
    # individually
    for category in wrong_classes:
        print(
            'Number of wrong predictions for class {}:'.format(category),
            wrong_classes[category]
        )

if __name__ == '__main__':

print('''
We started the search for the best features by training a classifier
on the frequency distribution of all words as the featureset, where an 
accuracy of 0.73 was achieved. We then found the most informative words
and re-trained the classifier with a frequency distribution of the 100 / 
1000 / 10000 most informative words, leading to accuracies of 0.73, 0.745
and 0.75 on the development set, respectively.
As an alternative, we trained classifiers with the frequency distributions
of all adjectives, all adverbs and all nouns, leading to accuracies of
0.735, 0.68 and 0.71, respectively.
Furthermore, we tried the average sentence length (0.51), average word 
length (0.495) and the proportion of adjectives (0.475), adverbs 
and nouns in the respective
texts. 
''')
    # import data
    review_data = [
        (movie_reviews.words(fileid), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]

    # shuffle data
    random.seed(1234)
    random.shuffle(review_data)

    # Get train, test and development data
    train_data = review_data[:math.floor(0.8 * len(review_data))]
    dev_data = review_data[math.ceil(0.8*len(review_data)) : math.floor(0.9*len(review_data))]
    test_data = review_data[math.ceil(0.9*len(review_data)):]

    # Get frequency distribution of all words in the train data
    fd_train_words = nltk.FreqDist(
        w.lower() for words, _ in train_data for w in words
    )

    # Evaluation of classifier using simple unigram frequencies as
    # features
    print(
        'Feature Set: All words'
    )

    # Use all words from the training data as features
    train_data_fd = [
        (
            get_fdist(words),
            category
        ) for words, category in train_data
    ]

    dev_data_fd = [
        (
            get_fdist(words, fd_train_words.keys()),
            category
        ) for words, category in dev_data
    ]

    # Train naive Bayes classifier
    classifier = nltk.NaiveBayesClassifier.train(train_data_fd)

    print(
        'Accuracy for using all words as features:',
        nltk.classify.accuracy(classifier, dev_data_fd)
    )

    # Get a sorted list of the most informative features (i.e. words)
    most_informative = classifier.most_informative_features(10000)

    # Use a combination of the 10000 most informative words and the
    # average sentence length
    print('''Combination of 10000 most informative words and average
sentence length''')
    classifier = nltk.NaiveBayesClassifier.train(
        [
            (
                get_counts(words, most_informative).update(
                    get_avg_sent_length_feature(words)
                ),
                category
            ) for words, category in train_data
        ]
    )

    print('Accuracy:', nltk.classify.accuracy(
        classifier, [
            (
                get_counts(words, most_informative).update(
                    get_avg_sent_length_feature(words)
                ),
                category
            ) for words, category in dev_data
        ]
    ))

    '''
    threshold_list = [100, 1000, 10000]

    # iterate over possible threshold values
    # Use slices of the list of most informative features of different
    # sizes as features and re-train the classifier
    for threshold in threshold_list:
        # Evaluation of classifier using simple unigram frequencies as
        # features
        print(
            'Feature Set: {threshold} most informative words'.format(
                threshold = threshold
            )
        )

        # Use all words from the training data as features
        train_data_fd = [
            (
                get_fdist(
                    words,
                    [word for word,_ in most_informative[:threshold]]
                ),
                category
            ) for words, category in train_data
        ]

        dev_data_fd = [
            (
                get_fdist(
                    words,
                    [word for word, _ in most_informative[:threshold]]
                ),
                category
            ) for words, category in dev_data
        ]

        # Train naive Bayes classifier
        classifier = nltk.NaiveBayesClassifier.train(train_data_fd)

        print(
            'Accuracy for threshold {threshold}:'.format(
                threshold=threshold
            ),
            nltk.classify.accuracy(classifier, dev_data_fd)
        )

    # Test features individually
    # Sentence length
    print('Average sentence length')
    classifier = nltk.NaiveBayesClassifier.train(
        [
            (
                get_avg_sent_length_feature(words),
                category
            ) for words, category in train_data
        ]
    )

    print('Accuracy:', nltk.classify.accuracy(
        classifier,
        [
            (
                get_avg_sent_length_feature(words),
                category
            ) for words, category in dev_data
        ]
    ))

    # Test features individually
    # Word length
    print('Average word length')
    classifier = nltk.NaiveBayesClassifier.train(
        [
            (
                get_avg_word_length_feature(words),
                category
            ) for words, category in train_data
        ]
    )

    print('Accuracy:', nltk.classify.accuracy(
        classifier,
        [
            (
                get_avg_word_length_feature(words),
                category
            ) for words, category in dev_data
        ]
    ))

    # Proportion of adjectives
    print('Proportion of adverbs')
    classifier = nltk.NaiveBayesClassifier.train(
        [
            (
                get_tag_proportion_feature(words, 'ADV'),
                category
            ) for words, category in train_data
        ]
    )

    print('Accuracy:', nltk.classify.accuracy(
        classifier,
        [
            (
                get_tag_proportion_feature(words, 'ADV'),
                category
            ) for words, category in dev_data
        ]
    ))

    
    # Proportion of nouns
    print('Proportion of nouns')
    classifier = nltk.NaiveBayesClassifier.train(
        [
            (
                get_tag_proportion_feature(words, 'NOUN'),
                category
            ) for words, category in train_data
        ]
    )

    print('Accuracy:', nltk.classify.accuracy(
        classifier,
        [
            (
                get_tag_proportion_feature(words, 'NOUN'),
                category
            ) for words, category in dev_data
        ]
    ))
    

    # Use only words with a certain pos-tag as features
    print('Evaluation of classifier using only adjective counts as features')
    classifier = nltk.NaiveBayesClassifier.train(
        [
            (
                nltk.probability.FreqDist(filter_words(words, ['ADJ'])),
                category
            )
            for words, category in train_data
        ]
    )

    print('Accuracy:', nltk.classify.accuracy(
        classifier,
        [
            (
                nltk.probability.FreqDist(filter_words(words, ['ADJ'])),
                category
            )
            for words, category in dev_data
        ]
    ))

    # Use only words with a certain pos-tag as features
    print('Evaluation of classifier using only noun counts as features')
    classifier = nltk.NaiveBayesClassifier.train(
        [
            (
                nltk.probability.FreqDist(filter_words(words, ['NOUN'])),
                category
            )
            for words, category in train_data
        ]
    )

    classify_get_wrong(
        [
            (
                nltk.probability.FreqDist(filter_words(words, ['NOUN'])),
                category
            )
            for words, category in dev_data
        ],
        dev_data,
        classifier
    )

    # Use only words with a certain pos-tag as features
    print('Evaluation of classifier using only adverb counts as features')
    classifier = nltk.NaiveBayesClassifier.train(
        [
            (
                nltk.probability.FreqDist(filter_words(words, ['ADV'])),
                category
            )
            for words, category in train_data
        ]
    )

    print('Accuracy:', nltk.classify.accuracy(
        classifier,
        [
            (
                nltk.probability.FreqDist(filter_words(words, ['ADV'])),
                category
            )
            for words, category in dev_data
        ]
    ))
    '''





