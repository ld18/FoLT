
import nltk
import sys
import logging
from math import log

def compute_LL(phrase, fdist_fg, fdist_bg):
    try:
        A = fdist_fg[phrase]
        B = fdist_bg[phrase]
        C = fdist_fg.N()
        D = fdist_bg.N()
        N = C + D
        E1 = (C * (A + B)) / N
        E2 = (D * (A + B)) / N
        log2_AE1 = log(A / E1, 2)
        log2_BE2 = log(B / E2, 2)
        ll = 2 * (A * log2_AE1 + B * log2_BE2)
        logging.info("compute_LL: noError (A:"+ str(A) +", B:"+ str(B) +", C:"+ str(C) +", D:"+ str(D) +", N:"+ str(N) +")")
        return ll
    except ValueError:
        logging.error("compute_LL: ValueError (A:"+ str(A) +", B:"+ str(B) +", C:"+ str(C) +", D:"+ str(D) +", N:"+ str(N) +")")
        return - sys.maxsize
    except Exception:
        logging.critical("compute_LL: unknownError (A:"+ str(A) +", B:"+ str(B) +", C:"+ str(C) +", D:"+ str(D) +", N:"+ str(N) +")")
        exit()


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
    logging.basicConfig(level = logging.ERROR)

    text_fg = nltk.corpus.gutenberg.words("carroll-alice.txt")
    bigrams_fg = list(nltk.bigrams(text_fg))
    fdist_bigrams_fg = nltk.FreqDist(bigrams_fg)

    text_bg = nltk.corpus.brown.words()
    bigrams_bg = list(nltk.bigrams(text_bg))
    fdist_bigrams_bg = nltk.FreqDist(bigrams_bg)

    print_10mostImprobableBigrams(fdist_bigrams_fg, fdist_bigrams_bg)


