
import logging
import Features
import Data
import Classifier
import DataAugmentation
import nltk
import random

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format=('%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(message)s'),
        datefmt='%H:%M:%S'
    )
    logger.info("Started program.")

    path = "src/train.tsv"
    datapoints, header, numberOfPoints = Data.readDatapointsFromFile(path)
    logger.info(str(numberOfPoints) + f" Datapoints found inside {path}.")

    # Shuffle dataset
    random.Random(1234).shuffle(datapoints)

    trainingSet, developmentSet = Data.splitDataSet(datapoints)
    logger.info(f"Split data as following: {len(trainingSet)} for training, {len(developmentSet)} for development.")

    # Get most informative unigram features
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

    most_informative_unigrams = [
        w for w, _ in unigram_classifier.most_informative_features(9000)
    ]

    # Define features
    feature_list = [
        Features.getMostCommonWordsCleaned
    ]

    # Uninitialized classifier
    NBC = nltk.classify.NaiveBayesClassifier

    # Try all combinations of features to get the best
    best_result, results = Classifier.testFeatureCombinations(
        trainingSet,
        developmentSet,
        NBC,
        feature_list,
        include_unigrams = most_informative_unigrams,
        num_words_threshold = 71,
        punctuation_threshold = 0.03685
    )
    # Currently Best combination: getMostCommonWordsCleaned, getUnigramFeatures,
    # moreThanxWords

    # Print the results
    for accuracy, feature_combination in results:
        print(
            'Accuracy: {:.2f}'.format(accuracy),
            'Features: ', [str(feature).split()[1] for feature in feature_combination]
        )

    # Augment the training data
    # Define Functions for augmentation
    augment_functions = [
        DataAugmentation.exchangeByDict,
        DataAugmentation.exchangeTagSensitive
    ]

    # Get exchange_dict
    exchange_dict = Data.makeExchangeDict(Data.readWordPairData())

    trainingSet_augmented = DataAugmentation.augmentDataset(
        trainingSet,
        augment_functions,
        exchange_dict = exchange_dict,
        exchange_dict_ts = {}
    )

    developmentSet_augmented = DataAugmentation.augmentDataset(
        developmentSet,
        augment_functions,
        exchange_dict = exchange_dict,
        exchange_dict_ts = {}
    )
    logger.debug(f'len(trainingSet_augmented[0].comment_text: {trainingSet_augmented[0].comment_text}')
    logger.debug(f'len(trainingSet_augmented[1200].comment_text: {trainingSet_augmented[1200].comment_text}')

    # Shuffle
    random.Random(123).shuffle(developmentSet_augmented)
    random.Random(123).shuffle(trainingSet_augmented)

    # Get the best feature functions
    best_feature_funcs = [
        getattr(Features, str(func).split()[1])
        for func in best_result[1]
    ]

    # Train the classifier on the augmented data using the best-performing
    # features
    classifier_augmented = Classifier.Classifier(
        NBC,
        Features.getFeatures,
        best_feature_funcs,
        include_unigrams=most_informative_unigrams,
        num_words_threshold=71,
        punctuation_threshold=0.03685
    )

    classifier_augmented.train(trainingSet_augmented)

    print(
        'Accuracy on development set with augmented data: ',
        Classifier.calculateAccuracy(classifier_augmented.predict(developmentSet_augmented))
    )

    Data.outputResults(developmentSet_augmented, './output_test')

    # if True:
    #     feature_list = [
    #         Features.getMostCommonWords,
    #         Features.getMostCommonWordsCleaned,
    #         Features.getPortionOfCapitalWords,
    #         Features.getPortionOfPunctuations
    #     ]
    #     for point in trainingSet:
    #         print(point)
    #         print("\t", Features.getFeatures(point.comment_text, feature_list))
    #         # print("\t", Features.getMostCommonWords(point.comment_text))
    #         # print("\t", Features.getMostCommonWordsCleaned(point.comment_text))
    #         # print("\t", Features.getPortionOfCapitalWords(point.comment_text))
    #         # print("\t", Features.getPortionOfPunctuations(point.comment_text))
