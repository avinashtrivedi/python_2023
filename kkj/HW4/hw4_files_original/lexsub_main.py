#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context
from nltk.tokenize import word_tokenize

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import string
import tensorflow

import gensim
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    res = {l.name().replace('_', ' ') for lm in wn.lemmas(lemma, pos) for l in lm.synset().lemmas() if l.name() != lemma}
    
    return sorted(res)


def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # Part 2
    lemma_freq = {}
    for lemma_obj in wn.lemmas(context.lemma, context.pos):
        for related_lemma_obj in lemma_obj.synset().lemmas():
            related_lemma = related_lemma_obj.name().replace('_', ' ')
            if related_lemma != context.lemma:
                lemma_freq[related_lemma] = lemma_freq.get(related_lemma, 0) + related_lemma_obj.count()

    most_frequent_lemma = max(lemma_freq, key=lemma_freq.get)
    return most_frequent_lemma

def wn_simple_lesk_predictor(context: Context) -> str:
    # Part 3
    stop_word_list = stopwords.words('english')
    max_val = 0
    max_key = max_lemma = ''
    max_count = -1

    for syn in wn.synsets(context.lemma, pos=context.pos):
        hyper_str = ''
        for item in syn.hypernyms():
            hyper_str = hyper_str + ' ' + item.definition() + ' ' + ' '.join(item.examples()) + ' '

        res_str = syn.definition().strip() + ' ' + ' '.join(syn.examples()).strip() + ' ' + hyper_str
        context_str = ' '.join(context.left_context) + ' ' + ' '.join(context.right_context)

        res_list = {word for word in set(res_str.split()) if (word.isalpha() and word not in stop_word_list)}
        context_list = {word for word in set(context_str.split()) if (word.isalpha() and word not in stop_word_list)}
        
        overlap_count = len(res_list.intersection(context_list))

        if overlap_count > max_val:
            if any(lem.name() != context.lemma for lem in syn.lemmas()):
                max_val = overlap_count
                max_key = syn
                
    def update_max_lemma(lemma_list):
        nonlocal max_count, max_lemma
        for l in lemma_list:
            if l.name() == context.lemma:
                continue
            if l.count() > max_count:
                max_count = l.count()
                max_lemma = l.name()

    if max_val != 0:
        update_max_lemma(max_key.lemmas())
    else:
        for syn in wn.synsets(context.lemma, pos=context.pos):
            update_max_lemma(syn.lemmas())

    return max_lemma.replace('_', ' ')


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        # Part 4
        maxsim = 0
        res = None
        context_word = context.lemma.lower()
        candidates = [candidate for candidate in get_candidates(context.lemma, context.pos) if candidate.lower() in self.model]

        for candidate in candidates:
            sim = self.model.similarity(context_word, candidate.lower())
            if sim > maxsim:
                maxsim = sim
                res = candidate

        return res


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # Part 5
        list_candidates = get_candidates(context.lemma, context.pos)
        input_tokens = ['[CLS]'] + context.left_context + ['[MASK]'] + context.right_context + ['[SEP]']
        mask_index  = len(context.left_context) + 1
        input_matrix = np.array(self.tokenizer.convert_tokens_to_ids(input_tokens)).reshape((1, -1))
        predictions = self.model.predict(input_matrix, verbose=False)[0]
        best_words_tokens = self.tokenizer.convert_ids_to_tokens(np.argsort(predictions[0][mask_index])[::-1])
        for w in best_words_tokens:
            if w.replace('_', ' ') in list_candidates:
                return w
        return ''


# Part 6
class custom_predictor(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True) 
        self.stop_words = stopwords.words('english')
    def predict(self,context):  
    
        lemma = context.lemma
        pos = context.pos
        right = [word for word in context.right_context if word.isalpha() and word.lower() not in self.stop_words and word not in string.punctuation]
        left = [word for word in context.left_context if word.isalpha() and word.lower() not in self.stop_words and word not in string.punctuation]

        res_list = left[-2:] + right[0:2] + [lemma]

        res_vec = np.zeros(300)
        for word in res_list:
            try:
                res_vec += self.model[word]
            except:
                continue

        def get_similarity_score(vector1, vector2):
            return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

        synonyms = get_candidates(lemma, pos)
        max_score = -1
        max_key = ''
        for syn in synonyms:
            try:
                syn_vec = self.model[syn]
                if syn_vec is not None:
                    sim_score = get_similarity_score(syn_vec, res_vec)
                    if max_score == -1 or sim_score > max_score:
                        max_score = sim_score
                        max_key = syn
            except:
                continue
        return max_key


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    
    bert_object = BertPredictor()
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = bert_object.predict(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
