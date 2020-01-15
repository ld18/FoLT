
class Classifier():
    def __init__(self, methode):
        pass
    def train(self, goldenDataset):
        pass
    def predict(self, dataset):
        pass


def calculateAccuracy(dataset):
    wrongPredictions = 0
    for datapoint in dataset:
        if datapoint.toxicity_predicted == None:
            raise Exception("Datapoint is not predicted!")
        if dataset.toxicity != datapoint.toxicity_predicted:
            wrongPredictions +=1
    return (len(dataset) - wrongPredictions) / len(dataset)
