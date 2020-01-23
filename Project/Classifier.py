import logging

logger = logging.getLogger(__name__)

class Classifier():
    def __init__(self, classifier, featureExtractor, feature_list, **feature_args):
        # Here, classifier is  still uninitialized
        self.classifier = classifier
        self.featureExtractor = featureExtractor
        self.feature_list = feature_list
        self.feature_args = feature_args

    def train(self, train_data):
        # Get all features from the training dataset
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
        #Dataset must be returned or used as ref, idk yet


def calculateAccuracy(dataset):
    wrongPredictions = 0
    for datapoint in dataset:
        if datapoint.toxicity_predicted == None:
            raise Exception("Datapoint is not predicted!")
        if datapoint.toxicity != datapoint.toxicity_predicted:
            wrongPredictions +=1
    return (len(dataset) - wrongPredictions) / len(dataset)
