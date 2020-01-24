import logging
import Features
import Data
import Classifier
import nltk
import random

def get_n_punctuation(text):
    words = nltk.tokenize.word_tokenize(text)


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

    # Get the number of toxic texts
    n_toxic = len([1 for datapoint in datapoints if datapoint.toxicity == 1])
    n_non_toxic = len([1 for datapoint in datapoints if datapoint.toxicity == 0])

    # Get the average number of words in toxic and non-toxic texts
    n_words_toxic = [
        len(nltk.tokenize.word_tokenize(datapoint.comment_text))
        for datapoint in datapoints
        if datapoint.toxicity == 1
    ]

    avg_n_words_toxic = sum(n_words_toxic) / len(n_words_toxic) # 61.12

    n_words_non_toxic = [
        len(nltk.tokenize.word_tokenize(datapoint.comment_text))
        for datapoint in datapoints if
        datapoint.toxicity == 0
    ]

    avg_n_words_non_toxic = sum(n_words_non_toxic) / len(n_words_non_toxic) # 83.13

    print(f'Average number of words in toxic texts: {avg_n_words_toxic:.2f}')

    print(f'Average number of words in non-toxic texts: {avg_n_words_non_toxic:.2f}')

    # Get average portion of punctuation symbols (0.0391)
    avg_portion_punctuation_toxic = sum([
        Features.getPortionOfPunctuations(datapoint.comment_text)['PortionOfPunctuation']
        for datapoint in datapoints
        if datapoint.toxicity == 1
    ])/n_toxic # 0.0391

    avg_portion_punctuation_non_toxic = sum([
        Features.getPortionOfPunctuations(datapoint.comment_text)['PortionOfPunctuation']
        for datapoint in datapoints
        if datapoint.toxicity == 0
    ])/n_non_toxic # 0.0346