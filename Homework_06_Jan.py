import nltk
import re

def get_t9_word(digits, freq_dist):
    digit_to_re = {
        '2' : '[abc]',
        '3' : '[def]',
        '4' : '[ghi]',
        '5' : '[jkl]',
        '6' : '[mno]',
        '7' : '[pqrs]',
        '8' : '[tuv]',
        '9' : '[wxyz]'
    }
    
    regex = "^"+ "".join([digit_to_re[digit] for digit in digits]) +"$"

    best_match = ''
    best_freq = 0
    for key, value in freq_dist.items():
        if re.search(regex, key):
            if freq_dist[key] > best_freq:
                best_match = key
                best_freq = freq_dist[key]
                
    return best_match
    
words = (
    [w.lower() for w in nltk.corpus.nps_chat.words()] 
     + [w.lower() for w in nltk.corpus.names.words()]
)       

fd = nltk.probability.FreqDist(words)

if __name__ == '__main__':
    sent = ['43556','73837','4','26','3463']
    print('Homework 6.2')
    print('(a)')
    print(
        '''We chose a combination of the nps chat corpus and the 
names corpus. The communication style in SMS is somewhat 
related to chat communication(informal, short, dialogous). 
The name "Peter" was not contained in the chat corpus, so we 
added the names corpus.'''
    )
    print('\n(b)')
    print('252473 -->', get_t9_word('252473', fd))
    print('\n(c)')
    print('Translation:')
    print([get_t9_word(digits, fd) for digits in sent])
    print(
        '''The output is definitely readable, but it contains one 
error in the last word (fine --> find). '''
    )
    print('Frequency of "fine":', fd['fine'])
    print('Frequency of "find":', fd['find'])
    print(
        '''Analysis of the word frequencies of "fine" and "find"
shows that they have almost the same value. This means that the 
error of mixing up the two words cannot really be removed with this
simple technique of finding the most frequent word in the corpus.
In a corpus with more counts for "fine" than "find" we would have 
correct guesses for the former, but incorrect guesses for the 
latter.'''
    )