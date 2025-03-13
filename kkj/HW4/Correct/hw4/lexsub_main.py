#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 
from collections import defaultdict
# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 
import string
from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    res = []
    for lm in wn.lemmas(lemma, pos):
        for l in lm.synset().lemmas():
            if l.name()!=lemma:
                res.append(l.name().replace('_', ' '))
    return list(sorted(list(set(res))))

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # Part 2
    dic = defaultdict(int)
    for lm in wn.lemmas(context.lemma, context.pos):
        for l in lm.synset().lemmas():
            if l.name() != context.lemma:
                dic[l.name().replace('_', ' ')] += l.count()
    return max(dic, key=lambda x: dic[x])

def wn_simple_lesk_predictor(context : Context) -> str:
    # Part 3
    stop_words = stopwords.words('english')
    lr_context = []
    for w in context.left_context+context.right_context:
        if w not in stop_words:
            lr_context.append(w)

    res = None
    res_score = -1

    for lm in wn.lemmas(context.lemma, context.pos):
        synset = lm.synset()
        definition = synset.definition()
        for example in synset.examples():
            definition += " " + example
        for hypernym in synset.hypernyms():
            definition += " " + hypernym.definition()
            for example in hypernym.examples():
                definition += " " + example
        tokens = tokenize(definition)
        definition = []
        for w in tokens:
            if w not in stop_words:
                definition.append(w)
        overlap = len((set(lr_context) & set(definition)))
        q_count = 0
        for l in synset.lemmas():
            if l.name() == context.lemma:
                q_count = l.count()

        for l in synset.lemmas():
            score = 1000 * overlap + 100 * q_count
            word = l.name()
            if word!=context.lemma:
                score += l.count()
                if score > res_score:
                    res = word
                    res_score = score
    return res.replace('_', ' ')
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        # Part 4
        maxsim = 0
        res = None
        for candidate in get_candidates(context.lemma, context.pos):
            try:
                sim = self.model.similarity(context.lemma, candidate)
                if sim>maxsim:
                    maxsim = sim
                    res = candidate
            except:
                continue
        return res


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # Part 5
        candidates = get_candidates(context.lemma, context.pos)
        input_toks = ['[CLS]'] + context.left_context + ['[MASK]'] + context.right_context + ['[SEP]']
        idx = len(context.left_context) + 1
        input_ids = self.tokenizer.convert_tokens_to_ids(input_toks)
        input_mat = np.array(input_ids).reshape((1, -1))
        outputs = self.model.predict(input_mat, verbose=False)
        predictions = outputs[0]
        best_words_ids = np.argsort(predictions[0][idx])[::-1]
        best_words_tokens = self.tokenizer.convert_ids_to_tokens(best_words_ids)
        for w in best_words_tokens:
            if w.replace('_', ' ') in candidates:
                return w
        return ''

# Part 6
class MyPredictor(object):

    def __init__(self, filename):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict(self, context : Context) -> str:
        input_toks = ['[CLS]'] + context.left_context + ['[MASK]'] + context.right_context + ['[SEP]']
        idx = len(context.left_context) + 1
        input_ids = self.tokenizer.convert_tokens_to_ids(input_toks)
        input_mat = np.array(input_ids).reshape((1, -1))
        outputs = self.bert_model.predict(input_mat, verbose=False)
        predictions = outputs[0]
        best_words_ids = np.argsort(predictions[0][idx])[::-1]
        best_words_tokens = self.tokenizer.convert_ids_to_tokens(best_words_ids)[:100]
        maxsim = 0
        res = ''
        for candidate in best_words_tokens:
            candidate = candidate.replace('_', ' ')
            if candidate == context.lemma:
                continue
            try:
                sim = self.w2v_model.similarity(context.lemma, candidate)
                if sim > maxsim:
                    maxsim = sim
                    res = candidate
            except:
                continue
        return res.replace('\u2022', '')


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    #predictor = MyPredictor(W2VMODEL_FILENAME)
    predictor = BertPredictor()
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
