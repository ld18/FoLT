
import nltk
import string

def getMostCommonWords(text):
    words = nltk.tokenize.word_tokenize(text)
    mostCommonWords = nltk.FreqDist(words).most_common(30)
    return mostCommonWords


def getMostCommonWordsCleaned(text):
    words = nltk.tokenize.word_tokenize(text)
    wordsWithoutPunctuation = [token for token in words if not (token[0] in string.punctuation)]
    mostCommonWords = nltk.FreqDist(wordsWithoutPunctuation).most_common(30)
    return mostCommonWords


def getPortionOfCapitalWords(text):
    numberOfCapitalWords = sum(map(str.isupper, text.split()))
    return numberOfCapitalWords /len(text)


def getPortionOfPunctuations(text):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    textWithoutPunctuation = text.translate(table)
    return (len(text) -len(textWithoutPunctuation)) /len(text)
