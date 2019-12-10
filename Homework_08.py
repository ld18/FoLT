# Homework 8
# 8.1
# Many words can have multiple senses (e.g. "program" can be both a verb
# and a noun). POS-taggers are important for NLP because they can aid in
# disambiguating word senses by providing additional knowledge about the
# processed words (their tags).

# Information retrieval, especially web search, is improved by POS-tagging
# It allows a more specific search than would be possible with only the raw
# tokens of the query by reducing the search to specific senses of a query
# word.

# 1. Rule-based taggers (e.g. regex-based)
# Pro: fast, small bias from training data
# Con: Not very accurate, lots of specific rules need to be written
# for improved accuracy, problem when no rule is matched (backup /default
# tagger needed)

# 2. Unigram/bigram/trigram based taggers
# Pro: Accurate
# Con: Need a lot of training data, trained model needs to be stored,
# can suffer from bias of training data, problem with unseen words/bigrams
# /trigrams (backup / default tagger needed)

# 8.2
import nltk
import re
import logging

# Class to implement chat tagger
# "Training" is done with initialization
# General class setup inspired by
# https://www.cs.bgu.ac.il/~elhadad/nlp18/NLTKPOSTagging.html
class Chat_Tagger():
    # Training of tagger happens in initialization
    # The tagger is a simple nltk trigram tagger
    def __init__(self, train_posts):
        self.t1 = nltk.UnigramTagger(train_posts)
        self.t2 = nltk.BigramTagger(train_posts, backoff=self.t1)
        self.t3 = nltk.TrigramTagger(train_posts, backoff=self.t2)

        patterns = [
            (r'^:-*[()P*oD]$', 'X'),
            (r'^l+o+l+$', 'X'),
            (r'^r+o+f+l+$', 'X')
        ]
        self.rt = nltk.RegexpTagger(patterns, backoff=self.t3)

    # Function to tag a single sentence / post
    def tag(self, post):
        return self.t3.tag(post)

    # Function to tag a list of posts
    def batch_tag(self, posts):
        return [self.tag(post) for post in posts]

    # Function to remove the tags from a post
    def untag(self, post):
        return [token[0] for token in post]

    # Function to get the tagging accuracy in percent of a single tagged
    # sentence and its gold standard
    def get_accuracy(self, gold_post, tagged_post):
        return sum([
            1 if gold_post[i][1] == tagged_post[i][1] else 0
            for i in range(len(gold_post))
        ])/(len(gold_post)*0.01)

    # Function to get the tagging accuracy from a list of tagged
    # sentences
    def get_batch_accuracy(self, gold_posts, tagged_posts):
        return sum([
            self.get_accuracy(gold_post, tagged_post)
            for gold_post, tagged_post in zip(gold_posts, tagged_posts)
        ])/(len(gold_posts))

    # Function to evaluate the tagging of sentences tagged with gold standard
    # tags
    def evaluate(self, gold_posts):
        tagged_posts = self.batch_tag(self.untag(post) for post in gold_posts)
        return self.get_batch_accuracy(gold_posts, tagged_posts)

# Function to replace a word with 'UNK' if it is not in keep_list and cannot
# be matched to keep_regexp
def replace_word(word, keep_list, keep_regexp):
    return word if (
            (word in keep_list)
            or any(re.match(pattern, word) for pattern in keep_regexp)
    ) else 'UNK'

if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format=('%(asctime)s : %(levelname)s : '
                '%(module)s : %(funcName)s : %(message)s'),
        datefmt='%H:%M:%S'
    )

    logger.info('Starting program')

    # Get complete set of tagged chat posts
    tagged_posts = nltk.corpus.nps_chat.tagged_posts(tagset='universal')

    # Get tagged sentences from the Brown corpus
    tagged_sents = nltk.corpus.brown.tagged_sents(tagset='universal')

    # Preprocess the training and test data
    # Get frequency distributions of words
    chat_fd = nltk.probability.FreqDist(
        [token[0] for post in tagged_posts for token in post]
    )

    # Get the most frequent words
    most_freq_X = 1000
    chat_most_freq = [
        t[0] for t in chat_fd.most_common(most_freq_X)
    ]
    # brown_most_freq = brown_fd.most_common(most_freq_X)

    # Define regex patterns to keep
    keep_regexp = [
        r'^:-*[()P*oD]$',
        r'^l+o+l+$',
        r'^r+o+f+l+$'
    ]

    # Replace every word that is not in most frequent words with 'UNK',
    # except it is an emoticon
    logger.info('Starting word replacement')
    tagged_posts = [
        [
            (replace_word(word, chat_most_freq, keep_regexp), tag)
            for word, tag in post
        ]
        for post in tagged_posts
    ]

    logger.info('Completed word replacement')

    # Get the total number of tagged posts
    nr_posts = len(tagged_posts)

    # Split the posts into training, development and test set
    train_posts = tagged_posts[:(nr_posts * 8) // 10]
    development_posts = tagged_posts[
        (nr_posts * 8) // 10 : (nr_posts * 9) // 10
    ]
    test_posts = tagged_posts[(nr_posts * 9) // 10:]

    # Initialize tagger
    tagger = Chat_Tagger(train_posts)

    # Evaluate tagger
    print(tagger.evaluate(development_posts))