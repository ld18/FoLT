
import nltk
import sys
from math import log

def compute_LL(phrase, fdist_fg, fdist_bg):
    try:
        A = fdist_fg[phrase]
        B = fdist_bg[phrase]
        C = len(fdist_fg)
        D = len(fdist_bg)
        N = C + D
        E1 = (C * (A + B)) / N
        E2 = (D * (A + B)) / N
        log2_AE1 = log(A / E1, 2)
        log2_BE2 = log(B / E2, 2)
        ll = 2 * (A * log2_AE1 + B * log2_BE2)
        return ll
    except ValueError:
        return - sys.maxsize
    except ZeroDivisionError:
        return - sys.maxsize


def print_10mostImprobableBigrams(fdist_fg, fdist_bg):
    SIPs = []
    for bigram in fdist_fg.keys():
        spiTuple = (bigram, compute_LL(bigram, fdist_fg, fdist_bg))
        SIPs.append(spiTuple)
    SIPs.sort(reverse = True, key = lambda spi: spi[1])
    print("10 most improbable bigrams:")
    for index, spi in enumerate(SIPs):
        if index >= 10:
            break
        print("( "+ spi[0][0] +" "+ spi[0][1] +" )\t\t LL: "+ str(spi[1]))


if __name__ == "__main__":
    text_fg = nltk.corpus.gutenberg.words("carroll-alice.txt")
    bigrams_fg = list(nltk.bigrams(text_fg))
    fdist_bigrams_fg = nltk.FreqDist(bigrams_fg)

    text_bg = nltk.corpus.brown.words("ca01")
    bigrams_bg = list(nltk.bigrams(text_bg))
    fdist_bigrams_bg = nltk.FreqDist(bigrams_bg)

    print_10mostImprobableBigrams(fdist_bigrams_fg, fdist_bigrams_bg)

