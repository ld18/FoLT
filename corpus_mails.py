import nltk

mails = nltk.corpus.LazyCorpusLoader(
    'mails',
    nltk.corpus.CategorizedPlaintextCorpusReader,
    r'(?!\.).*\.txt',
    cat_pattern=r'(spam|nospam)/.*'
)