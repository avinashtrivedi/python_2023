import sys
from collections import defaultdict
import math
import random
from tqdm.notebook import tqdm
import os
import os.path


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
    This should work for arbitrary values of n >= 1 
    """
    sequence = ['START'] + sequence + ['STOP']
    if n > 2:
        sequence = ['START']*(n-2) + sequence
    res = []
    for i in range(len(sequence)-n+1):
        res.append(tuple(sequence[i: i+n]))
    return res


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
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here
        self.sentencetotal = 0

        for seq in corpus:
            self.sentencetotal += 1
            for gram in get_ngrams(seq, 1):
                self.unigramcounts[gram] += 1
            for gram in get_ngrams(seq, 2):
                self.bigramcounts[gram] += 1
            for gram in get_ngrams(seq, 3):
                self.trigramcounts[gram] += 1

        self.unigramtotal = sum(self.unigramcounts.values())
        self.bigramtotal = sum(self.bigramcounts.values())
        self.trigramtotal = sum(self.trigramcounts.values())
        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        #print(trigram[:2])
        if self.trigramcounts[trigram] == 0:
            return 0
        if trigram[:2] == ('START', 'START'):
            return self.trigramcounts[trigram]/self.sentencetotal
        return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        #print(bigram, tuple([bigram[0]]))
        if self.bigramcounts[bigram] == 0:
            return 0
        return self.bigramcounts[bigram]/self.unigramcounts[tuple([bigram[0]])]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts[unigram]/self.unigramtotal

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability(trigram[1:]) + \
               lambda3 * self.raw_unigram_probability(tuple([trigram[2]]))
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        res = 0.0
        for trigram in trigrams:
            res += math.log2(self.smoothed_trigram_probability(trigram))
        return res

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        M = 0
        l = 0.0
        for seq in tqdm(corpus):
            M += len(seq)
            l += self.sentence_logprob(seq)
        l /= M
        return pow(2, -l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1 <= pp2:
                correct += 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp2 <= pp1:
                correct += 1
            total += 1
        
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

