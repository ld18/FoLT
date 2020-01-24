
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

# Function to extract unigram features
# if a list of words is passed as kwarg 'include_unigrams', each word in the
# list is checked. If it appears in the set of words from text, the feature
# value is set to 1
# If 'unigram_features' is empty or not passed, each word from the set of words
# in text is returned as a single feature with value 1
def getUnigramFeatures(text, **kwargs):

    # Get the set of words in the text
    words = set([word.lower() for word in nltk.tokenize.word_tokenize(text)])

    if 'include_unigrams' in kwargs.keys():
        if len(kwargs['include_unigrams']) > 0:
            return {
                word : (1 if word in words else 0)
                for word in kwargs['include_unigrams']
            }

    else:
        return {
            word : 1
            for word in words
        }

# Function to extract a feature which is 1 if a text contains more than
# x words and 0 if a text contains less than x words
def moreThanxWords(text, **kwargs):
    x = kwargs['num_words_threshold']
    if len(nltk.tokenize.word_tokenize(text)) > x:
        return {f'more_than_{x}_words' : 1}

    else:
        return {f'more_than_{x}_words' : 0}

def punctuationHigherThanx(text, **kwargs):
    x = kwargs['punctuation_threshold']
    portion_of_punctuation = getPortionOfPunctuations(text)['PortionOfPunctuation']
    if portion_of_punctuation > x:
        return {f'punctuation_higher_than_{x:.4f}' : 1}

    else:
        return {f'punctuation_higher_than_{x:.4f}' : 0}

# Function which takes a set s and returns a set of tuples of all combinations
# of the elements in s
def getAllCombinations(s, combinations):

    if len(s) > 1:
        for elem in s:
            combinations.update(getAllCombinations(s-{elem}, combinations))

    combinations.add(tuple(s))
    return combinations
