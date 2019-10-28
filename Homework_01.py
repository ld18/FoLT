
import nltk

def top_suffixes(words, numberOfMostCommon = 10):
    suffixes = []
    for word in words:
        if len(word) > 4:
            suffixes.append(word[-2:])
    freqDist = nltk.FreqDist(suffixes)
    mostCommonfSuffixes = [suffix[0] for suffix in freqDist.most_common(numberOfMostCommon)]
    return mostCommonfSuffixes

if __name__ == "__main__":
    words = ['caraa', 'bataa', 'dadbb', 'dadbbc', 'dadbcb', 'dadba', 'dadaa']
    print(top_suffixes(words))

    emma_words = nltk.corpus.gutenberg.words('austen-emma.txt')
    print(top_suffixes(emma_words))