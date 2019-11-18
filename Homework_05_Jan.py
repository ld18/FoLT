import nltk
import scipy.stats as stats

# function to get the maximum similarity of two words
# from the synset similarities
def get_max_similarity(a, b):
    # Create an empty list to store the similarity values
    similarities = []
    
    # Iterate over all synsets of the first word
    for s1 in nltk.corpus.wordnet.synsets(a):
        # Iterate over all synsets of the second word
        for s2 in nltk.corpus.wordnet.synsets(b):
            # Compute the similarity
            sim = s1.path_similarity(s2)
            # Check whether the similarity is not None (will otherwise)
            # cause problems with the max() function
            if sim is not None:
                similarities.append(sim)
    
    # Check if the list of similarities is not empty (will otherwise)
    # cause problems with the max function
    if len(similarities) > 0:
        return max(similarities)
    
    # If list is empty, return minimum possible similarity of 0
    else:
        return 0

if __name__ == '__main__':
    # Homework 5.1
    # (a)
    print('Homework 5.1')
    print('(a)')
    text = (
        'car-automobile, gem-jewel, journey-voyage, boy-lad, coast-shore, asylum-madhouse, magician-wizard, '
        'midday-noon, furnace-stove, food-fruit, bird-cock, bird-crane, tool-implement, brother-monk, lad-brother, '
        'crane-implement, journey-car, monk-oracle, cemetery-woodland, food-rooster, coast-hill, forest-graveyard, '
        'shore-woodland, monk-slave, coast-forest, lad-wizard, chord-smile, glass-magician, rooster-voyage, noon-string'
    )
    
    # Split the raw text into tuples of word pairs.
    word_pairs = [
        tuple(pair.split('-')) for pair in text.split(', ')
    ]
    
    # Compute the list of word pairs together with similarities.
    pair_similarities = {
        pair : get_max_similarity(pair[0], pair[1])
        for pair in word_pairs
    }
    
    # Print list of word pairs sorted according to similarity score.
    print('List of word pairs sorted according to wordnet path similarity')
    for tup in sorted(
        pair_similarities.items(), key = lambda tup : tup[1], reverse=True
    ):
        print(tup)
    
    # (b)
    print('\n (b)')
    
    # There are multiple duplicate values in the generated list of 
    # similarity scores.
    # Ranking between the duplicate values is not possible, therefore
    # create a set of all unique values and sort it.
    unique_sim = sorted(
        list(set([tup[1] for tup in pair_similarities.items()])),
        reverse=True
    )
        
    # Get the spearman correlation coefficient, use the given list by 
    # Miller & Charles as reference and the ranks of the sorted
    # list of unique similarity values in the same order.
    print(
        'Spearman Correlation between Miller & Charles ranking of ' 
        'word similarity and ranking from wordnet:'
    )
    print(stats.spearmanr(
        [i for i in range(len(word_pairs))], # perfect ranking
        [
            unique_sim.index(pair_similarities[pair])
            for pair in word_pairs
        ]
    ))
    print(
        'The correlation might improve when directly using the '
        'ranks from the similarity scores without giving elements '
        'with the same score the same rank. However, the ranking of '
        'elements with the same score is only based on the input '
        'order, which is the correct order. Hence, it is biased. \n'
    )
    
    # Homework 5.2
    # Get all definitions of the synsets of the word "witch"
    print('Homework 5.1')
    print('Definitions of the synsets of "witch"')
    for i, s in  enumerate(nltk.corpus.wordnet.synsets('witch')):
        print(i+1, ':', s.definition())
        
    print('\n')
    print('They are not completely interchangeable')
    print(
        'Example 1: Definitions 1 and 4. Different meanings of the same noun'
    )
    print(
        'Example 2: Definitions 5 describes a verb, whereas all other synsets '
        'refer to nouns.'
    )

    print(
        'To avoid inappropriate substitution, one possibility is to '
        'analyze the POS-tag of the word to be substituted. This way, '
        'one avoids replacing verbs with nouns etc. Regarding words of the '
        'same part of speech, one can analyze the textual context. '
        'In the example of "witch", one can check whether the containing '
        'text is a text about magic or not to know whether replacement '
        'with words from synset 1 or 4 is more appropriate. '
    )