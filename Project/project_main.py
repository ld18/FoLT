import sys
sys.path.append('../')

import logging
import Project.Features as Features
import Project.Data as Data
import Project.Classifier as Classifier
import Project.DataAugmentation as DataAugmentation
import nltk
import random

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format=('%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(message)s'),
        datefmt='%H:%M:%S'
    )
    logger.info("Started program.")

    # Read the labeled data
    path = "src/train.tsv"
    datapoints, header, numberOfPoints = Data.readDatapointsFromFile(path)
    logger.info(str(numberOfPoints) + f" Datapoints found inside {path}.")

    # Shuffle dataset
    random.Random(1234).shuffle(datapoints)

    # Split the labeled dataset
    trainingSet, developmentSet = Data.splitDataSet(datapoints)
    logger.info(f"Split data as following: {len(trainingSet)} for training, {len(developmentSet)} for development.")

    # Train a naive Bayes classifier on all unigram features
    unigram_classifier = nltk.classify.NaiveBayesClassifier.train(zip(
        [
            Features.getUnigramFeatures(
                datapoint.comment_text
            )
            for datapoint in datapoints
        ],
        [
            datapoint.toxicity for datapoint in datapoints
        ]
    ))

    print(unigram_classifier.most_informative_features(10))

    # Get the most informative unigram features
    most_informative_unigrams = [
        w for w, _ in unigram_classifier.most_informative_features(9000)
    ]

    # Define features
    feature_list = [
        Features.getMostCommonWordsCleaned,
        Features.getMostCommonWords,
        Features.getUnigramFeatures,
        Features.moreThanxWords,
    ]

    # Uninitialized Naive Bayes classifier
    NBC = nltk.classify.NaiveBayesClassifier

    # Initialize a classifier to be trained on the train split
    classifier_dev = Classifier.Classifier(
        NBC,
        Features.getFeatures,
        feature_list,
        include_unigrams=most_informative_unigrams,
        num_words_threshold=72,
        punctuation_threshold=0.03685
    )

    # Train the classifier
    classifier_dev.train(trainingSet)

    # Evaluate the classifier
    print(Classifier.calculateAccuracy(classifier_dev.predict(
        developmentSet
    )))
    FP = 0
    FN = 0

    for datapoint in developmentSet:
        if not (datapoint.toxicity_predicted == datapoint.toxicity):
            print(
                f'Actual Toxicity: {datapoint.toxicity},',
                f'Predicted toxicity: {datapoint.toxicity_predicted},',
                datapoint.comment_text
            )
            if datapoint.toxicity_predicted == 1:
                FP += 1
            elif datapoint.toxicity_predicted == 0:
                FN += 1

    print(f'Number of false positives: {FP}')
    print(f'number of false negatives: {FN}')

    # -------------------------------------------------------------------------
    # Augment the training data
    # Define Functions for augmentation
    augment_functions = [
        DataAugmentation.exchangeByDict,
        DataAugmentation.exchangeTagSensitive,
        DataAugmentation.exchangeNames
    ]

    # Get exchange_dict of male-female word pairs
    exchange_dict = Data.makeExchangeDict(Data.readWordPairData())

    # Get the names data as frequency distributions
    male_fd, female_fd = Data.read_names()

    # Get the dictionary for name replacement
    names_dict = Data.makeExchangeDict(
        Data.makePairsFromFDs(
            male_fd, female_fd
        )
    )
    # Augment the train set
    trainingSet_augmented = DataAugmentation.augmentDataset(
        trainingSet,
        augment_functions,
        exchange_dict = exchange_dict,
        exchange_dict_ts = {},
        names_dict = names_dict
    )

    # Augment the development set
    developmentSet_augmented = DataAugmentation.augmentDataset(
        developmentSet,
        augment_functions,
        exchange_dict = exchange_dict,
        exchange_dict_ts = {},
        names_dict = names_dict
    )
    logger.debug(f'len(trainingSet_augmented[0].comment_text: {trainingSet_augmented[0].comment_text}')
    logger.debug(f'len(trainingSet_augmented[1200].comment_text: {trainingSet_augmented[1200].comment_text}')

    # Shuffle
    random.Random(123).shuffle(developmentSet_augmented)
    random.Random(123).shuffle(trainingSet_augmented)

    # Initialize a classifier for the augmented data
    classifier_augmented = Classifier.Classifier(
        NBC,
        Features.getFeatures,
        feature_list,
        include_unigrams=most_informative_unigrams,
        num_words_threshold=72,
        punctuation_threshold=0.03685
    )

    # Train the classifier
    classifier_augmented.train(trainingSet_augmented)

    # Evaluate the classifier on the augmented development set
    print(
        'Accuracy on development set with augmented data: ',
        Classifier.calculateAccuracy(classifier_augmented.predict(developmentSet_augmented))
    )

    # -------------------------------------------------------------------------
    # Test data evaluation
    # Read the test data
    testSet = Data.readDatapointsFromFile('src/test.tsv')[0]

    # Initialize a classifier to be trained on the complete train data (train + dev)
    classifier_full = Classifier.Classifier(
        NBC,
        Features.getFeatures,
        feature_list,
        include_unigrams=most_informative_unigrams,
        num_words_threshold=72,
        punctuation_threshold=0.03685
    )

    # Train the classifier
    classifier_full.train(trainingSet + developmentSet)

    # Predict the labels of the test set
    classifier_full.predict(testSet)

    # Write the results file
    Data.outputResults(testSet, './test_data_evaluation')


    # Augment the test data
    testSet_augmented = DataAugmentation.augmentDataset(
        testSet,
        augment_functions,
        exchange_dict = exchange_dict,
        exchange_dict_ts = {},
        names_dict = names_dict
    )

    # Initialize a classifier to be trained on the complete augmented train dataset
    classifier_full_augmented = Classifier.Classifier(
        NBC,
        Features.getFeatures,
        feature_list,
        include_unigrams=most_informative_unigrams,
        num_words_threshold=72,
        punctuation_threshold=0.03685
    )

    # Train the classifier
    classifier_full_augmented.train(trainingSet_augmented + developmentSet_augmented)

    # Predict the labels of the augmented test set
    classifier_full_augmented.predict(testSet_augmented[700:])

    # Write the classification result file
    Data.outputResults(testSet_augmented[700:], './test_data_augmented_evaluation')
