from nltk.corpus import brown
from nltk import ngrams
from nltk import FreqDist
import numpy as np
from collections import Counter


class BrownDataset(object):
    def __init__(self, include_start=True):
        self.words = brown.words()
        self.words = map(lambda x: x.lower(), self.words)
        self.total_word_cnt = len(self.words) + 2 * len(brown.sents())  # include START and END
        if include_start:
            self.words.append(u'START')
        self.words.append(u'END')
        self.vocab = set(self.words)


        self.vocab_len = len(self.vocab)
        self.word_to_idx = dict(zip(list(self.vocab), range(self.vocab_len)))

        self.sentences = []
        self.bigrams = []
        self.unigrams = []
        for sent in brown.sents():
            sentence = map(lambda x: x.lower(), sent)
            if include_start:
                sentence.insert(0, u'START')
            sentence.append(u'END')
            self.sentences.append(sentence)
            self.bigrams.extend(list(ngrams(sentence, 2)))
            self.unigrams.extend(sentence)

        self.unigram_freq = dict(Counter(self.unigrams))

        self.num_sentences = len(self.sentences)
        self.bigram_cnt = FreqDist(self.bigrams)
        self.bigram_len = len(self.bigram_cnt)
        self.bigram_idx = dict(zip(self.bigram_cnt.keys(), range(self.bigram_len)))
        self.bigram_freq = np.asarray(self.bigram_cnt.values())
        self.num_bigrams = len(self.bigram_cnt)


if __name__ == '__main__':
    dataset = BrownDataset()
    print dataset.num_bigrams   # 434952
    print dataset.V             # 49815
    print dataset.num_sentences # 57340







