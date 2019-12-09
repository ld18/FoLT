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

# Class to implement chat tagger
# "Training" is done with initialization
# Inspired by https://www.cs.bgu.ac.il/~elhadad/nlp18/NLTKPOSTagging.html
class Chat_Tagger():
    # Training of tagger happens in initialization
    # The tagger is a simple nltk trigram tagger
    def __init__(self, train_posts):
        self.trigram_tagger = nltk.TrigramTagger(train_posts)

    # Function to tag a single sentence / post
    def tag(self, post):
        return self.trigram_tagger.tag(post)

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


if __name__ == '__main__':
    tagged_posts = nltk.corpus.nps_chat.tagged_posts(tagset='universal')
    nr_posts = len(tagged_posts)

    train_posts = tagged_posts[:(nr_posts * 8) // 10]
    development_posts = tagged_posts[
        (nr_posts * 8) // 10 : (nr_posts * 9) // 10
    ]
    test_posts = tagged_posts[(nr_posts * 9) // 10:]

    tagger = Chat_Tagger(train_posts)

    print(tagger.evaluate(development_posts))





