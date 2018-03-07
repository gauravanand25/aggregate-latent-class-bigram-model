from __future__ import division
from data_utils import BrownDataset
import numpy as np
import math


class UnigramMLEModel:
    def __init__(self, dataset, debug=False):
        self.debug = debug
        self.dataset = dataset

    def prob_word(self, word):
        ret = self.dataset.unigram_freq[word] / self.dataset.total_word_cnt
        assert ret != 0.0, word
        return  ret

    def log_prob_sentence(self, sentence):
        return sum([math.log(self.prob_word(word)) for word in sentence])


class BigramMLEModel:
    def __init__(self, dataset, debug=False):
        self.debug = debug
        self.dataset = dataset

    def prob_word_given_ctxt(self, nextw, prevw):
        ret = self.dataset.bigram_cnt[(prevw, nextw)] / self.dataset.unigram_freq[prevw]
        assert ret != 0, (prevw, nextw)
        return ret

    def log_prob_sentence(self, sentence):
        # Assuming START and END are included
        log_p = 0
        for i in range(len(sentence) - 1):
            log_p += math.log(self.prob_word_given_ctxt(sentence[i + 1], sentence[i]))
        return log_p


class LatentBigramModel:
    def __init__(self, dataset, num_classes=3, num_iterations=20, debug=False):
        self.dataset = dataset
        self.K = num_classes
        self.posterior = np.random.random(size=(dataset.num_bigrams, self.K))  # N x K
        self.w_to_hs_transition_prob = np.random.random(size=(dataset.vocab_len, self.K))  # V x K
        self.hs_to_w_emission_prob = np.random.random(size=(self.K, self.dataset.vocab_len))  # K x V
        self.iterations = num_iterations
        self.debug = debug


    def prob_word_given_ctxt(self, wprev, wnext):
        save = (wprev, wnext)
        wprev = self.dataset.word_to_idx[wprev]
        wnext = self.dataset.word_to_idx[wnext]
        ret = np.sum(self.w_to_hs_transition_prob[wprev] * self.hs_to_w_emission_prob[:, wnext])
        assert ret != 0, (save, self.w_to_hs_transition_prob[wprev], self.hs_to_w_emission_prob[:, wnext])
        return ret

    def log_prob_sentence(self, sentence):
        # Assuming START and END are included
        log_p = 0
        for i in range(len(sentence) - 1):
            log_p += math.log(self.prob_word_given_ctxt(sentence[i], sentence[i+1]))
        return log_p

    def prob_sentence(self, sentence):
        # Assuming START and END are included
        pr = 1
        for i in range(len(sentence) - 1):
            pr *= self.prob_word_given_ctxt(sentence[i], sentence[i + 1])
        return pr

    def marginal_log_likelihood(self):
        ll = 0
        for sentence in self.dataset.sentences:
            ll += self.log_prob_sentence(sentence)
        return ll/self.dataset.total_word_cnt

    def e_step(self):
        for bigram, freq in self.dataset.bigram_cnt.iteritems():
            wprev = self.dataset.word_to_idx[bigram[0]]
            wnext = self.dataset.word_to_idx[bigram[1]]
            unnorm_posterior_prob = self.w_to_hs_transition_prob[wprev] * self.hs_to_w_emission_prob[:, wnext]
            self.posterior[dataset.bigram_idx[bigram]] = unnorm_posterior_prob / np.sum(unnorm_posterior_prob)

    def m_step(self):
        cnt_category_bigram = self.posterior * self.dataset.bigram_freq[:, None]  # N x K
        idx = 0
        self.w_to_hs_transition_prob.fill(0)  # causes error for 'END' as it doesn't appear as first word in any bigram.
        self.w_to_hs_transition_prob[dataset.word_to_idx[u'END']].fill(1)
        self.hs_to_w_emission_prob.fill(0)  # causes error for 'START' as it doesn't apprear as second word in any bigram
        self.hs_to_w_emission_prob[:, self.dataset.word_to_idx[u'START']].fill(1)
        for bigram, freq in self.dataset.bigram_cnt.iteritems():
            wprev = self.dataset.word_to_idx[bigram[0]]
            self.w_to_hs_transition_prob[wprev] += cnt_category_bigram[idx]
            wnext = self.dataset.word_to_idx[bigram[1]]
            self.hs_to_w_emission_prob[:, wnext] += cnt_category_bigram[idx]
            idx += 1

        self.w_to_hs_transition_prob /= np.sum(self.w_to_hs_transition_prob, axis=1).reshape(-1, 1)
        self.hs_to_w_emission_prob /= np.sum(self.hs_to_w_emission_prob, axis=1).reshape(-1, 1)

        if self.debug:
            assert np.sum(np.abs(np.sum(cnt_category_bigram, axis=1) - self.dataset.bigram_freq)) < 1e-8, "Error"
            assert np.sum(np.abs(np.sum(self.w_to_hs_transition_prob, axis=1) - np.ones(dataset.vocab_len))) < 1e-8, "Error"
            assert np.sum(np.abs(np.sum(self.hs_to_w_emission_prob, axis=1) - np.ones(self.K))) < 1e-8, "Error"

    def em(self):
        for iter in range(self.iterations):
            # E-step
            self.e_step()
            # M-step
            self.m_step()
            print "iteration = ", iter, self.marginal_log_likelihood()


if __name__ == '__main__':
    dataset = BrownDataset()

    # # Q1.1
    # uni_model = UnigramMLEModel(dataset)
    # print "PROB UNIGRAM colorless green ideas sleep furiously END", uni_model.log_prob_sentence(
    #     ['colorless', 'green', 'ideas', 'sleep', 'furiously', 'END'])
    #
    # # Q1.2
    # bi_model = BigramMLEModel(dataset)
    # print "PROB BIGRAM colorless green ideas sleep furiously END", bi_model.log_prob_sentence(
    #     ['START', 'colorless', 'green', 'ideas', 'sleep', 'furiously', 'END'])

    latent_model = LatentBigramModel(dataset, num_classes=3, num_iterations=20, debug=True)
    latent_model.em()
    a = latent_model.prob_sentence(['START', 'colorless', 'green', 'ideas', 'sleep', 'furiously', 'END'])
    b = latent_model.prob_sentence(['START', 'furiously', 'sleep', 'ideas', 'green', 'colorless', 'END'])
    print a, b, a/b
