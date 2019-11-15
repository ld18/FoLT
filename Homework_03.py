import nltk
import math
import string

# Homework 4.2
# (a)

# Function to calculate a conditional frequency distribution
# for the character frequencies in different languages
# Inputs:
# languages: list of languages which to include in the cfd
# words: dictionary of words for each language as input
# tokenized: bool, tells whether the corpora for building the cfd
# are given as strings or as lists of tokens
# Returns 
# A conditional frequency distribution where the languages are the
# conditions and the values are the lower case character frequencies
def build_language_models(languages, words, tokenized=True):
    
    # If the input corpora are not tokenized, tokenize them
    if tokenized:
        tokenized_words = words
    else: 
        tokenized_words = {
            lang : nltk.tokenize.word_tokenize(words[lang])
            for lang in words
        }
        
    # Initialize empty cfd
    cfd = nltk.probability.ConditionalFreqDist()
    
    # Iterate over all languages
    for lang in languages:
        # Iterate over all tokens in the corpus for the language
        for token in tokenized_words[lang]:
            # Iterate over all characters in lowercased token:
                for char in token.lower():
                    # check if char in alphabetic
                    if char.isalpha():
                        # If yes, update cfd
                        cfd[lang][char] += 1
    
    return cfd
    
# (b)

# Function to return list of lexical units
def get_lex_units(text, lex_unit_type, tokenized=True):
    # If the input text is not tokenized, tokenize it
    # Remove punctuation in any case
    if tokenized:
        tokenized_text = [
            token for token in text if not (token[0] in string.punctuation)
        ]
    else: 
        tokenized_text = [
            token for token in nltk.tokenize.word_tokenize(text)
            if not (token[0] in string.punctuation)
        ]
        
    # Check which type of lexical unit is to be returned, and return appropriate
    # list
    if lex_unit_type == 'char':
        return get_char_list(tokenized_text)
    
    if lex_unit_type == 'token':
        return tokenized_text
    
    if lex_unit_type == 'token_bigram':
        return get_bigram_list(tokenized_text)
    
    if lex_unit_type == 'char_bigram':
        return get_char_bigram_list(tokenized_text)
    
# Function to get a list of characters from an input list of tokens
def get_char_list(tokens):
    return [
        char for token in tokens for char in token
    ]

# Function to get a list of token bigrams from a list of tokens
def get_bigram_list(tokens):
    return [
        (tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)
    ]

# Function to get a list of character bigrams from a list of tokens
def get_char_bigram_list(tokens):
    # Generate empty list to store bigrams
    char_bigram_list = []
    
    # Iterate over all tokens
    for token in tokens:
        # Check if token is longer than 1 letter
        if len(token) > 1:
            for i in range(len(token) - 1):
                char_bigram_list.append((token[i], token[i+1]))
        else:
            char_bigram_list.append((token[0], ''))
    
    return char_bigram_list
    
# Function which sums the lexical unit (characters, bigrams, ...) probabilities 
# of a given language for a list of lexical units from a text
def get_prob_sum(lex_units, lang, cfd):
    # Create variable to store the sum of probabilities
    score = 0
    
    # Iterate over all lexical units
    for lu in lex_units:
        # Add the frequency of the lexical unit to the score
        score += cfd[lang].freq(lu)
            
    return score
    
# (c)
# Function to guess the language for a sample text
# Calculates the probability scores for all languages in the cfd and returns
# the language with the highest score
def guess_language(text, cfd, lex_unit_type, tokenized=True):
    # Get the list of lexical units
    lex_units = get_lex_units(text, lex_unit_type, tokenized=tokenized)
    
    scores = [
        (
            lang,
            get_prob_sum(
                lex_units,
                lang,
                cfd
            )
        )
        for lang in cfd
    ]
    
    return sorted(scores, reverse=True, key = lambda entry: entry[1])[0][0]

# Homework 4.2
# (a-c)
# Function to calculate a conditional frequency distribution
# of tokens, token bigrams or character bigrams
def get_cfd(languages, words, lex_unit_type, tokenized=True):
    
    # If the input corpora are not tokenized, tokenize them
    # In any case, remove punctuation and make lowercase
    if tokenized:
        tokenized_words = {
            lang : [
                t.lower() for t in words[lang] 
                if not (t in string.punctuation)
            ]
            for lang in words
        }
    else: 
        tokenized_words = {
            lang : [
                t.lower() for t in nltk.tokenize.word_tokenize(words[lang])
                if not (t in string.punctuation)
            ]
            for lang in words
        }
        
    # Initialize empty cfd
    cfd = nltk.probability.ConditionalFreqDist()
    
    # Check which lexical unit type is to be analyzed
    if lex_unit_type == 'token':
    
        # Iterate over all languages
        for lang in languages:
            # Iterate over all tokens in the corpus for the language
            for token in tokenized_words[lang]:
                # Update cfd
                cfd[lang][token] += 1
                    
    elif lex_unit_type == 'char_bigram':
        # Iterate over all languages
        for lang in languages:
            # Iterate over all tokens in the corpus for the language
            for token in tokenized_words[lang]:
                # Iterate over characters in token
                for i in range(len(token) - 1):
                    # Update cfd
                    cfd[lang][tuple(token[i+j] for j in range(2))] += 1
    
    elif lex_unit_type == 'token_bigram':
        # Iterate over all languages
        for lang in languages:
            # Iterate over all tokens in the corpus for the language
            for i in range(len(tokenized_words[lang])-1):
                # Update cfd
                cfd[lang][tuple(tokenized_words[lang][i+j] for j in range(2))] += 1
    
    return cfd

if __name__ == '__main__':

    # Exercise 4.1 
    # (a)
    
    languages = [
        'English', 'German_Deutsch', 'French_Francais'
    ]

    language_base = dict(
        (language, nltk.corpus.udhr.words(language + '-Latin1'))
        for language in languages
    )

    language_model_cfd = build_language_models(languages, language_base)

    print('Homework 4.1')
    print('(a)')
    for language in languages:
        for key in list(language_model_cfd[language].keys())[:10]:
            print(language, key, '->', language_model_cfd[language].freq(key))

    print('\n')
    
    # (d)
     
    text1 = "Peter had been to the office before they arrived."
    text2 = "Si tu finis tes devoirs, je te donnerai des bonbons."
    text3 = "Das ist ein schon recht langes deutsches Beispiel."

    # guess the language by comparing the frequency distributions
    print('(d)')
    print(
        'guess for english text is',
        guess_language(text1, language_model_cfd, 'char', tokenized=False)
    )
    print(
        'guess for french text is',
        guess_language(text2, language_model_cfd, 'char', tokenized=False)
    )
    print(
        'guess for german text is',
        guess_language(text3, language_model_cfd, 'char', tokenized=False)
    )
    print('\n')
    # (e)
    # Compare character probabilities between German, English and French
    alphabet = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'r', 's', 't', 'u', 'v', 'w', 'x',
        'y', 'z'
    ]
    print('(e)')
    for char in alphabet:
        print(
            'Frequency of {char} in English: {e_freq:.4f}. In German: {g_freq:.4f}. In French: {f_freq:.4f}.'.format(
                char = char,
                e_freq = language_model_cfd['English'].freq(char), 
                g_freq = language_model_cfd['German_Deutsch'].freq(char),
                f_freq = language_model_cfd['French_Francais'].freq(char)
            )
        )
    print('\n')
    
    # Calculate the sum of the squared differences of the character probabilities
    # between the languages
    print('Sum of squared differences between')
    print('English and German: {diff:.4f}'.format(
        diff = sum([
            (
                language_model_cfd['English'].freq(char) 
                - language_model_cfd['German_Deutsch'].freq(char)
            )**2
            for char in alphabet
        ])
    ))
    print('French and German: {diff:.4f}'.format(
        diff = sum([
            (
                language_model_cfd['French_Francais'].freq(char) 
                - language_model_cfd['German_Deutsch'].freq(char)
            )**2
            for char in alphabet
        ])
    ))
    print('French and English: {diff:.4f}'.format(
        diff = sum([
            (
                language_model_cfd['French_Francais'].freq(char) 
                - language_model_cfd['English'].freq(char)
            )**2
            for char in alphabet
        ])
    ))
    print('\n')
    print(
        'Because English and German are closely related languages, it is '
        'expected that the character frequencies in the two languages are '
        'mostly similar. Thus it might be difficult to decide between German '
        'and English for a given text, it achieves high similarity scores '
        'with both languages.\n'
        'However, analyzing the sum of squared differences between the '
        'letter frequencies from the different languages, it seems that '
        'they are even more similar between French and English. This '
        'suggests that English and French should be even harder to '
        'distinguish. \n'
        'That this is not reflected in the experimental results might '
        'be because the guessed texts are very short.'
    )
    print('\n')
    
    # Homework 4.2
    # (a-c)
    # Generate cfds for tokens, token bigrams and character bigrams
    token_cfd = get_cfd(languages, language_base, 'token')
    token_bigram_cfd = get_cfd(languages, language_base, 'token_bigram')
    char_bigram_cfd = get_cfd(languages, language_base,'char_bigram')

    # guess the language by comparing the frequency distributions of tokens
    print('Homework 4.2')
    print('Language guessing based on token frequencies:')
    print(
        'guess for english text is',
        guess_language(text1, token_cfd, 'token', tokenized=False)
    )
    print(
        'guess for french text is',
        guess_language(text2, token_cfd, 'token', tokenized=False)
    )
    print(
        'guess for german text is',
        guess_language(text3, token_cfd, 'token', tokenized=False)
    )
    print('\n')

    # guess the language by comparing the frequency distributions of token bigrams
    print('Language guessing based on token bigram frequencies:')
    print(
        'guess for english text is',
        guess_language(text1, token_bigram_cfd, 'token_bigram', tokenized=False)
    )
    print(
        'guess for french text is',
        guess_language(text2, token_bigram_cfd, 'token_bigram', tokenized=False)
    )
    print(
        'guess for german text is',
        guess_language(text3, token_bigram_cfd, 'token_bigram', tokenized=False)
    )
    print('\n')

    # guess the language by comparing the frequency distributions of character bigrams
    print('Language guessing based on token bigram frequencies:')
    print(
        'guess for english text is',
        guess_language(text1, char_bigram_cfd, 'char_bigram', tokenized=False)
    )
    print(
        'guess for french text is',
        guess_language(text2, char_bigram_cfd, 'char_bigram', tokenized=False)
    )
    print(
        'guess for german text is',
        guess_language(text3, char_bigram_cfd, 'char_bigram', tokenized=False)
    )
    print('\n')

    # (d)
    print(
        'The language guesser based on tokens should work best, '
        'because most words are unique for a given language. '
        'This means that the score for the correct language will be '
        'high in the scoring scheme used here, and low for all incorrect '
        'languages. As visible above, the language guesser based on token '
        'bigrams works less well, as most bigrams from the texts to be '
        'guessed to not appear in the text used to generate the cdfs. '
        'Hence, a larger "training corpus" to generate the cdfs would '
        'be needed. Character bigrams seem to work well, too. '
    )