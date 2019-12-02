import nltk

def findAndSortAllByTag(taged_tokens, tag):
    tokens = []
    for taged_token in taged_tokens:
        if taged_token[1] == tag:
            tokens.append(taged_token[0])
    return sorted(list(set(tokens)))

def findAndAllByTagSeq(taged_tokens, tag_sepqence):
    tokens = []
    for i in range(len(taged_tokens) -1 -len(tag_sepqence)):
        for count, searchTag in enumerate(tag_sepqence):
            tokenTag = taged_tokens[i +count][1]
            if not tokenTag == searchTag:
                break
        else:
            tokens.append([taged_token[0] for taged_token in taged_tokens[i:i +len(tag_sepqence)]])
    return tokens

# Function to get all words which are tagged with all of the tags given in
# list tags
def get_tagged_with(tagged_words, tag_list):
    # Create set of word types for every tag
    type_sets = [
        {word for word, tag2 in tagged_words if tag2 == tag}
        for tag in tag_list
    ]

    # Get the intersection of all sets
    if len(tag_list) == 1:
        intersection = type_sets[0]

    else:
        intersection = type_sets[0].intersection(
            *type_sets[1:]
        )


    # return the sorted intersection of all sets of word types of the given
    # tags
    return sorted(list(intersection))

# Function to get the ratio of words from two categories and a certain tag
def get_ratio(tagged_words, wordlist_1, wordlist_2, tag):
    # Create two counters
    count_1 = 0
    count_2 = 0

    # iterate over all words
    for word, tag_1 in tagged_words:
        # Check if the tag of the current word is the needed tag
        if tag_1 == tag:
            # If yes, check if the word belongs to either of the categories
            # of interest
            if word.lower() in wordlist_1:
                count_1 += 1
            elif word.lower() in wordlist_2:
                count_2 += 1

    return count_1/count_2

# Function which takes a list of sentences and a list of word-tag combinations
# and finds for each combination the first sentence in which it appears
# Could be improved by stopping the search search aim is completed
def get_single_sent_per_word_tag(sents, word_list, num_tags):
    # Create a dictionary with a list of tags found for each word to keep
    found_tags = {
        word : []
        for word in word_list
    }
    # Create dictionary to store found sentences
    found_sents = {}
    # Iterate over all sentences as long as not for every word-tag-combination
    # a sentence has been found
    # Create counter
    i = 0
    while all([len(l) < num_tags for l in found_tags.values()]):
        sent = sents[i]
        # Iterate over all words in the sentence
        for word, tag in sent:
            # Iterate over all words in list of words to be found
            for word_to_find in word_list:
                # Check if current word is one of the words to be found
                if (
                        (word_to_find == word.lower())
                        and (tag not in found_tags[word_to_find])
                ):
                    found_sents[tuple((word_to_find, tag))] = sent
                    found_tags[word_to_find].append(tag)

        # Update counter
        i += 1

    return found_sents


if __name__ == '__main__':
    words = nltk.corpus.brown.words()
    taged_tokens = nltk.pos_tag(words)

    print("\n(a) Produce an alphabetically sorted list of the distinct words tagged as MD.")
    MDtaged_tokens = findAndSortAllByTag(taged_tokens, "MD")
    print(MDtaged_tokens)

    # (b)
    print('(b)')
    tagged_words = nltk.corpus.brown.tagged_words()
    print(get_tagged_with(tagged_words, ['NNS', 'VBZ']))

    print("\n(c) Identify three-word prepositional phrases of the form ADP + DET + NOUN (eg. 'at the end').")
    sepTaged_tokens = findAndAllByTagSeq(taged_tokens, ["RB", "DT", "NN"])
    print(sepTaged_tokens)

    # (d)
    print('(d)')

    # Get the set of all words tagged with 'PRON' (universal tagset)
    print(set([
        word.lower() for word, tag
        in nltk.corpus.brown.tagged_words(tagset='universal')
        if tag == 'PRON'
    ]))

    # Manually create lists of feminine and masculine pronouns
    masc_prons = [
        'he', 'himself', 'him', 'his', 'hisself'
    ]

    fem_prons = [
        'herself', 'her', 'she', 'hers'
    ]

    print(
        'The ratio of masculine to feminine pronouns is',
        get_ratio(
            nltk.corpus.brown.tagged_words(tagset='universal'),
            masc_prons,
            fem_prons,
            'PRON'
        )
    )

    # 7.2
    # (a)
    print('Homework 7.2')
    print('(a)')
    # Generate the set of all tuples of word and tag to get all possible
    # combinations of word and tag
    lowercase_tagged_words = set([
        (word.lower(), tag)
        for word, tag in nltk.corpus.brown.tagged_words(tagset='universal')
    ])

    # Calculate a frequency distribution to get the number of different tags
    # from each word by counting how often each word appears in the
    # set of all combinations of word and tag
    num_tags_fd = nltk.probability.FreqDist(
        word for word, tag in lowercase_tagged_words
    )

    # From the frequency distributions of the number of different tags per
    # word, calculate a second frequency distribution of the number of
    # different words per number of tags
    num_words_num_tags_fd = nltk.probability.FreqDist(
        num_tags for word, num_tags
        in num_tags_fd.items()
    )
    print('Distribution of words which can have x different tags')
    print('x  : num_words')
    for num_tags, num_words in num_words_num_tags_fd.items():
        print('{num_tags:2} : {num_words:5}'.format(
            num_tags=num_tags,
            num_words=num_words
        ))

    # (b)
    print('(b)')
    # Get the maximum number of different tags
    max_tags = max(num_words_num_tags_fd.keys())
    print(
        'Maximum number of different tags per word:',
        max_tags
    )

    # Get a list of words with the maximum number of different tags
    words_max_tags = [
        word for word, tag_count in num_tags_fd.items()
        if tag_count == max_tags
    ]

    print(words_max_tags)

    # get one sentence per word-tag combination of the words in word_max_tags
    sents = get_single_sent_per_word_tag(
        nltk.corpus.brown.tagged_sents(tagset='universal'),
        words_max_tags,
        max_tags
    )

    # Print
    for word in words_max_tags:
        for word_tag, sent in sents.items():
            if word == word_tag[0]:
                print(word_tag, [w for w, t in sent])


