import gensim
from gensim import models
import nltk
from nltk.corpus import stopwords
import logging
from random import shuffle

logging.basicConfig(level = logging.INFO)

reuters_sents = list(nltk.corpus.reuters.sents()) 
#shuffle(reuters_sents)
stop = set(stopwords.words('english'))

class LabeledLineSentence(object):
    def __init__(self, corpus):
        self.sents = corpus
    def __iter__(self):
        for uid, line in enumerate(self.sents):
            words = [w for w in line if w not in stop]
            yield models.doc2vec.LabeledSentence(words=words, tags=['SENT_%s' % uid])

logging.info('loading sentences')           
sentences = LabeledLineSentence(reuters_sents)  

logging.info('initiating model')
model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1, iter = 1)
model.build_vocab(sentences)

logging.info('training model')
for epoch in range(10):
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate`
    model.min_alpha = model.alpha  # fix the learning rate, no decay

logging.info('saving model')
model.save("reuters_model.doc2vec")
