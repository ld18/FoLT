import nltk

# Basic function to augment a given text using the functions given in
# augment_functions
def augmentDuplicate(text, augment_functions, **kwargs):
    # tokenize
    words = nltk.tokenize.word_tokenize(text)

    # pos-tag
    pos_tagged_words = nltk.pos_tag(words, tagset='universal')

    # iterate over augmentation functions
    # All augmentation functions should take a list of (word, pos-tag) tuples
    # as input and return a similar list
    for func in augment_functions:
        pos_tagged_words = func(pos_tagged_words, **kwargs)

    # Join together words with exchanges
    return ' '.join([t for t, _ in pos_tagged_words])

# Function which replaces all words which are keys in exchange_dict
# with the respective value
def exchangeByDict(pos_tagged_words, **kwargs):

    new_pos_tagged_words = []

    for word, pos_tag in pos_tagged_words:
        if word.lower() in kwargs['exchange_dict']:
            new_pos_tagged_words.append(tuple((kwargs['exchange_dict'][word], pos_tag)))

        else:
            new_pos_tagged_words.append(tuple((word, pos_tag)))

    return new_pos_tagged_words