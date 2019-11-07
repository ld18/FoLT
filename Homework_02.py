
import nltk
import sys
import logging
import string 
from math import log

def compute_LL(phrase, fdist_fg, fdist_bg):
    # Use exception handling for cases with frequency of 0
    try:
        # Define variables according to the given formula
        A = fdist_fg[phrase]
        B = fdist_bg[phrase]
        C = fdist_fg.N()
        D = fdist_bg.N()
        N = C + D
        # Compute the Log-Likelihood stepwise
        E1 = (C * (A + B)) / N
        E2 = (D * (A + B)) / N
        # log2_AE1 = log(A / E1, 2)
        # log2_BE2 = log(B / E2, 2)
        ll = 2 * (
            (
                ((A * (log(A / E1, 2))) if (A > 0) else 0) 
                + ((B * log(B / E2, 2)) if (B > 0) else 0)
            )
        )
        logging.info(
            "compute_LL: noError (A:" + str(A) 
            +", B:"+ str(B)
            +", C:"+ str(C)
            +", D:"+ str(D)
            +", N:"+ str(N)
            +")"
        )
        return ll
    # In cases where the computation of the LL score is not possible, 
    # return the negative of the largest representable number as the LL score
    # This guarantees  that these phrases will be at the end of the sorted list
    except ValueError:
        logging.error(
            "compute_LL: ValueError (A:" + str(A)
            +", B:"+ str(B)
            +", C:"+ str(C)
            +", D:"+ str(D)
            +", N:"+ str(N)
            +")"
        )
        return - sys.maxsize
    except Exception:
        logging.critical(
            "compute_LL: unknownError (A:" + str(A) 
            +", B:"+ str(B) 
            +", C:"+ str(C)
            +", D:"+ str(D) 
            +", N:"+ str(N) 
            +")"
        )
        exit()

# Function which takes two a foreground and a background frequency 
# distribution and prints the 10 words from the foreground frequency
# distribution with the highest LL scores
def print_10mostImprobableBigrams(fdist_fg, fdist_bg):
    # Create empty list to store the LL score for every 
    # bigram
    SIPs = []
    # Iterate over all bigrams in the frequency distribution 
    for bigram in fdist_fg.keys():
        # Calculate the LL score for the current bigram and store 
        # as a tuple together with the bigram itself
        spiTuple = (bigram, compute_LL(bigram, fdist_fg, fdist_bg))
        SIPs.append(spiTuple)
    
    # Sort the list of bigrams with LL scores according to LL score
    SIPs.sort(reverse = True, key = lambda spi: spi[1])
    print("10 most improbable bigrams:")
    # Iterate over the 10 bigrams with the largest LL scores,
    # print the bigram and the LL score
    for index, spi in enumerate(SIPs):
        if index >= 10:
            break
        print(
            "( "+ spi[0][0] +" "+ spi[0][1]
            +" )\t\t LL: "+ str(spi[1])
        )


if __name__ == "__main__":
    logging.basicConfig(level = logging.ERROR)

    text_fg = [
        t.lower() for t in nltk.corpus.gutenberg.words("carroll-alice.txt")
        if not (
            t in nltk.corpus.stopwords.words('english') 
            or t[0] in string.punctuation
        )
    ]
    bigrams_fg = list(nltk.bigrams(text_fg))
    fdist_bigrams_fg = nltk.FreqDist(bigrams_fg)

    text_bg = [
        t.lower() for t in nltk.corpus.brown.words()
        if not (
            t in nltk.corpus.stopwords.words('english') 
            or t[0] in string.punctuation
        )
    ]
    bigrams_bg = list(nltk.bigrams(text_bg))
    fdist_bigrams_bg = nltk.FreqDist(bigrams_bg)

    print_10mostImprobableBigrams(fdist_bigrams_fg, fdist_bigrams_bg)
