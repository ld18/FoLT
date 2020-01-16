
class Classifier():
    def __init__(self, classifier, featureExtractor):
        self.classifier = classifier
        self.featureMethode = featureExtractor
    def train(self, dataset):
        for datapoint in dataset:
            features = self.featureExtractor(datapoint)
            self.classifier.train(features, datapoint.toxicity)
    def predict(self, dataset):
        for datapoint in dataset:
            features = self.featureExtractor(datapoint)
            datapoint.toxicity_predicted = self.classifier.predict(features)
        #Dataset must be returned or used as ref, idk yet


def calculateAccuracy(dataset):
    wrongPredictions = 0
    for datapoint in dataset:
        if datapoint.toxicity_predicted == None:
            raise Exception("Datapoint is not predicted!")
        if dataset.toxicity != datapoint.toxicity_predicted:
            wrongPredictions +=1
    return (len(dataset) - wrongPredictions) / len(dataset)
