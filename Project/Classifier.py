import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format=('%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(message)s'),
    datefmt='%H:%M:%S'
)

class Classifier():
    def __init__(self, classifier, featureExtractor, feature_list, **feature_args):
        # Here, classifier is  still uninitialized
        self.classifier = classifier
        self.featureExtractor = featureExtractor
        self.feature_list = feature_list
        self.feature_args = feature_args

    def train(self, train_data):
        # Initialize classifier with first datapoint
        self.classifier = self.classifier.train([(
            self.featureExtractor(
                train_data[0].comment_text,
                self.feature_list,
                **self.feature_args
            ),
            train_data[0].toxicity
        )])

        # Continue training with the rest of the datapoints
        for i in range(1, len(train_data)):
            features = self.featureExtractor(
                train_data[i].comment_text,
                self.feature_list,
                **self.feature_args
            )
            self.classifier.train([(features, train_data[i].toxicity)])

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
