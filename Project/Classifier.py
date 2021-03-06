import sys
sys.path.append('..')

import logging
import Project.Features as Features

logger = logging.getLogger(__name__)

class Classifier():
    #Save all needed functions and parameters for a costum classifier class
    def __init__(self, classifier, featureExtractor, feature_list, **feature_args):
        self.classifier = classifier
        self.featureExtractor = featureExtractor
        self.feature_list = feature_list
        self.feature_args = feature_args

    #train classifier by first extracting the features from the dataset
    def train(self, train_data):
        train_features = [
            self.featureExtractor(
                train_data[i].comment_text,
                self.feature_list,
                **self.feature_args
            )
            for i in range(len(train_data))
        ]
        self.classifier = self.classifier.train(
            zip(
                train_features,
                [train_data[i].toxicity for i in range(len(train_data))]
            )
        )
        logger.info('10 most informative features: {}'.format(
            self.classifier.most_informative_features(10)
        ))

    #predict a list of datapoints again by first extracting the features
    def predict(self, test_data):
        for datapoint in test_data:
            features = self.featureExtractor(
                datapoint.comment_text,
                self.feature_list,
                **self.feature_args
            )
            logger.debug('features: {}'.format(features))
            datapoint.toxicity_predicted = self.classifier.classify(features)

        return test_data

#calculate the accuracy of the classifier by comapring hte predicted labels of a dataset to the real label
def calculateAccuracy(dataset):
    wrongPredictions = 0
    for datapoint in dataset:
        if datapoint.toxicity_predicted == None:
            raise Exception("Datapoint is not predicted!")
        if datapoint.toxicity != datapoint.toxicity_predicted:
            wrongPredictions +=1
    return (len(dataset) - wrongPredictions) / len(dataset)

# Function to test all combinations of a list of feature extraction functions
# Inputs:
# train_data: The data for training a classifier as a list of data points in
# the Data.Datapoint format
# test_data: The data to test the classifier in the same format as train_data
# classifier_base: An uninitialized nltk classifier
# features: A list of feature functions to be tested
#
# Returns:
# 1. A tuple of the best accuracy and the best feature combination
# 2. A list of tuples of accuracy and corresponding feature combination
def testFeatureCombinations(train_data, test_data, classifier_base, features, **feature_args):
    # Get all combinations of the given features
    combinations = list(Features.getAllCombinations(set(features), set()))

    # Empty list to store the accuracies
    accuracy_results = []

    # variable to store the best accuracy value
    best_accuracy = 0

    # Variable to store the best feature combination
    best_feature_combination = tuple()

    # Iterate over all combinations of features
    for feature_tuple in combinations:
        # Initialize the classifier
        classifier = Classifier(
            classifier_base,
            Features.getFeatures,
            feature_tuple,
            **feature_args
        )
        # Train
        classifier.train(train_data)

        # Predict and get the accuracy
        accuracy = calculateAccuracy(classifier.predict(test_data))

        accuracy_results.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_feature_combination = feature_tuple

    print('Best Accuracy: {:.2f},'.format(best_accuracy), 'Best features: ', best_feature_combination)

    return (best_accuracy, best_feature_combination), zip(accuracy_results, combinations)
