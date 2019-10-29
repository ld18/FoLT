
import nltk

# Function which returns the most common suffixes in a list of words
# Inputs:
#   words: A list of words for which the most common suffixes are to be
#   found
#   numberOfMostCommon: The number of most common most common suffixes to return
# Returns:
#   mostCommonfSuffixes: A list of numberOfMostCommon most common suffixes
def top_suffixes(words, numberOfMostCommon = 10):
    # Empty list to store all suffixes found
    suffixes = []
    # Iterate over all words
    for word in words:
        # Check if analyzed word has 5 or more characters
        if len(word) > 4:
            # If yes, append the last 2 characters of the word to the list of
            # suffixes
            suffixes.append(word[-2:])
    # Calculate the frequency distribution in the list suffixes
    freqDist = nltk.FreqDist(suffixes)
    # Make a list of the 10 most common suffixes
    mostCommonfSuffixes = [suffix[0] for suffix in freqDist.most_common(numberOfMostCommon)]
    return mostCommonfSuffixes

if __name__ == "__main__":
    words = ['caraa', 'bataa', 'dadbb', 'dadbbc', 'dadbcb', 'dadba', 'dadaa']
    print(top_suffixes(words))

    emma_words = nltk.corpus.gutenberg.words('austen-emma.txt')
    print(top_suffixes(emma_words))