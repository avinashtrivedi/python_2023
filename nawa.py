import sys
from collections import defaultdict
import math
import random
import os
import os.path

import numpy as np
"""
COMS W4705 - Natural Language Processing - Spring 2018
Homework 1 - Programming Component: Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    pad_start = ['START'] * (n-1) if n > 2 else ['START']
    pad_stop = ['STOP']
    sequence = pad_start + sequence + pad_stop
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngrams.append(tuple(sequence[i:i+n]))
    return ngrams

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)

            for unigram in unigrams:
                self.unigramcounts[unigram] += 1

            for bigram in bigrams:
                self.bigramcounts[bigram] += 1

            for trigram in trigrams:
                self.trigramcounts[trigram] += 1

                if trigram[:2] == ('START', 'START'):
                    # to compute trigram probs of type ('START', 'START', 'xxx')
                    self.bigramcounts[('START', 'START')] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if self.bigramcounts[trigram[:2]] != 0 :
            # warning: do not test as ```trigram[:2] in self.bigramcounts```
            # because evaluating d[x] on a defaultdict d that doesn't contain x
            # adds x to d (sets x: 0) and that might happen elswhere in the code
            return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]
        else:
            # case trigram with first two words not seen as bigram during training
            return 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.unigramcounts[bigram[:1]] != 0:
            # same as for raw_trigram_probability
            return self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]]
        else:
            # case bigram with 1st word not seen during training
            return 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        if not hasattr(self, 'n_words'):
            # if we haven't computed the denominator yet, do it now
            self.n_words = sum(self.unigramcounts.values())
            self.n_words -= - self.unigramcounts[('START',)] + self.unigramcounts[('STOP',)]
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        
        return self.unigramcounts[unigram]/self.n_words

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        current_trigram = (None, 'START', 'START')
        sentence = list()
        i = 0

        while current_trigram[2] != 'STOP' and i < t:
            word_1 = current_trigram[1]
            word_2 = current_trigram[2]
            # find all trigrams that start with word_1, word_2
            candidates = [trigram for trigram in self.trigramcounts.keys() if trigram[:2] == (word_1, word_2)]
            probabilities = [self.raw_trigram_probability(trigram) for trigram in candidates]

            generated_word = np.random.choice([candidate[2] for candidate in candidates], 1, p=probabilities)[0]
            current_trigram = (word_1, word_2, generated_word)
            sentence.append(generated_word)
            i += 1

        return sentence            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 3/5.0
        lambda2 = 1/5.0
        lambda3 = 1/5.0
        
        smoothed = 0.0
        smoothed += lambda1 * self.raw_trigram_probability(trigram)
        smoothed += lambda2 * self.raw_bigram_probability(trigram[1:])
        smoothed += lambda3 * self.raw_unigram_probability(trigram[2:])

        return smoothed
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        
        trigram_probs = [self.smoothed_trigram_probability(trigram) for trigram in trigrams]
        return sum(math.log2(prob) for prob in trigram_probs)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l = 0
        M = 0
        for sentence in corpus:
            l += self.sentence_logprob(sentence)
            M += len(sentence)
        l /= M
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            correct += (pp1 < pp2)

        for f in os.listdir(testdir2):
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total += 1
            correct += (pp2 < pp1)
        
        return correct / total




if __name__ == "__main__":
    #generator = corpus_reader("hw1_data/brown_train.txt")
    #for sentence in generator:
    #    print(sentence)

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)

    # Training data perplexity:
    dev_corpus = corpus_reader(sys.argv[1], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 'hw1_data/ets_toefl_data/train_low.txt',
                                   "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
    print(acc)


# if __name__ == "__main__":
#     train_file_left = './data/left/left-all.txt'
#     train_file_right = './data/right/right-all.txt'
#     left_model = TrigramModel(train_file_left)
#     right_model = TrigramModel(train_file_right)
#     left_model.save(filename="left-model.txt")
#     right_model.save(filename="right-model.txt")
