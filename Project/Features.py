
import nltk
import string

# Function to extract all features given in feature_list from text
# If the feature functions require additional parameters, they can be passed
# in kwargs
def getFeatures(text, feature_list, **kwargs):
    features = {}

    for feature_function in feature_list:
        features.update(feature_function(text, **kwargs))

    return features

# Functions to extract individual types of features
# Use of kwargs argument ensures interoperability

def getMostCommonWords(text, **kwargs):
    words = nltk.tokenize.word_tokenize(text)
    mostCommonWords = nltk.FreqDist(words).most_common(30)
    return mostCommonWords


def getMostCommonWordsCleaned(text, **kwargs):
    words = nltk.tokenize.word_tokenize(text)
    wordsWithoutPunctuation = [token for token in words if not (token[0] in string.punctuation)]
    mostCommonWords = nltk.FreqDist(wordsWithoutPunctuation).most_common(30)
    return mostCommonWords


def getPortionOfCapitalWords(text, **kwargs):
    numberOfCapitalWords = sum(map(str.isupper, text.split()))
    return {'PortionOfCapital' : numberOfCapitalWords /len(text)}


def getPortionOfPunctuations(text, **kwargs):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    textWithoutPunctuation = text.translate(table)
    return {'PortionOfPunctuation' : (len(text) -len(textWithoutPunctuation)) /len(text)}

