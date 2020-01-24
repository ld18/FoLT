
import logging
import Features
import Data
import Classifier
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

    path = "src/train.tsv"
    datapoints, header, numberOfPoints = Data.readDatapointsFromFile(path)
    logger.info(str(numberOfPoints) + f" Datapoints found inside {path}.")

    # Shuffle dataset
    random.Random(1234).shuffle(datapoints)

    trainingSet, developmentSet = Data.splitDataSet(datapoints)
    logger.info(f"Split data as following: {len(trainingSet)} for training, {len(developmentSet)} for development.")

    feature_list = [
        Features.getMostCommonWords,
        Features.getMostCommonWordsCleaned,
        Features.getPortionOfCapitalWords,
        Features.getPortionOfPunctuations,
        Features.getUnigramFeatures,
    ]

    NBC = nltk.classify.NaiveBayesClassifier

    best_result, results = Classifier.testFeatureCombinations(
        trainingSet, developmentSet, NBC, feature_list
    )

    for accuracy, feature_combination in results:
        print(
            'Accuracy: {:.2f}'.format(accuracy),
            'Features: ', [str(feature).split()[1] for feature in feature_combination]
        )

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
