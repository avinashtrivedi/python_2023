import sys
import os
import re
import math
from nltk.stem import PorterStemmer

class VectorData:
    def __init__(self):
        self.norm = 0
        self.termid = []
        self.weight = []

# VectorSpaceModel class as provided
class VectorSpaceModel:
    def __init__(self, index, stop_words, number_of_docs):
        self.index = index
        self.vdocs = {}
        self.term_i = 0
        self.stop_words = stop_words
        self.number_of_docs = number_of_docs
        self.stemmer = PorterStemmer()

    def preprocess_query(self, query):
        tokens = re.findall(r'\w+', query.lower())
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return tokens

    def search(self, query):
        query_terms = self.preprocess_query(query)
        print(f"Processed Query Terms: {query_terms}")

        for term in query_terms:
            data = self.index.get(term, {})
            for doc_id, doc_data in data.items():
                if 'tf' in doc_data:
                    tf = doc_data['tf']
                    idf = 0
                    if data and len(data) < self.number_of_docs:
                        idf = math.log(self.number_of_docs / len(data))
                    w = (1 + math.log(tf)) * idf if tf > 0 else 0

                    if doc_id not in self.vdocs:
                        self.vdocs[doc_id] = VectorData()

                    self.vdocs[doc_id].norm += w * w
                    self.vdocs[doc_id].termid.append(self.term_i)
                    self.vdocs[doc_id].weight.append(w)
                self.term_i += 1

        for v in self.vdocs.values():
            v.norm = math.sqrt(v.norm)

        vquery = VectorData()
        term_i = 0

        for term, data in self.index.items():
            idf = 0
            if term in query_terms:
                if len(data) and len(data) < self.number_of_docs:
                    idf = math.log(self.number_of_docs / len(data))
                w = (1 + math.log(query_terms.count(term))) * idf
                vquery.norm += w * w
                vquery.termid.append(term_i)
                vquery.weight.append(w)
            term_i += 1

        vquery.norm = math.sqrt(vquery.norm)

        scores = []
        for doc_id, vdoc in self.vdocs.items():
            dot_product = sum([a * b for i, a in enumerate(vquery.weight) if i in vdoc.termid for j, b in enumerate(vdoc.weight) if j == i])
            score = dot_product / (vquery.norm * vdoc.norm) if vquery.norm > 0 and vdoc.norm > 0 else 0
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_stopword_list():
        list = []
        try:
            with open("C:\\Users\\mccut\\OneDrive\\Desktop\\cps842f23_a1_LukeMcCutcheon\\stopwords.txt", "r") as file:
                list = [line.strip() for line in file]
        except FileNotFoundError:
            print("File 'common_words' not found.")
            print("You are missing a file crucial to perform stopword removal.\n")
            sys.exit(1)
        return list

def perform_stopword_removal(terms, stopwords):
    return [t for t in terms if t not in stopwords]

def perform_stemming(terms, stemmer):
    return [stemmer.stem(t) for t in terms]

# Placeholder for command-line arguments processing
message_usage = "Usage: python eval.py [-stem] [-stopwords]\n"

# Placeholder for argument parsing
flag_stem = '-stem' in sys.argv
flag_stop = '-stopwords' in sys.argv

if flag_stop:
    stopwords = 'C:\\Users\\mccut\\OneDrive\\Desktop\\cps842f23_a1_LukeMcCutcheon\\stopwords.txt'
else:
    stopwords = []

if '-help' in sys.argv:
    print(message_usage)
    sys.exit(0)

vsm = VectorSpaceModel(index, stopwords, number_of_docs)

# Open and process query.text
query_file_path = "C:\\Users\\mccut\\OneDrive\\Desktop\\cps842f23_a1_LukeMcCutcheon\\query"
try:
    with open(query_file_path, "r") as file_query:
        info = {}
        current_op = ""
        qid = 0
        for line in file_query:
            if not line.strip():
                continue
            if line.startswith("."):
                current_op = line[1]
                if current_op == "I":
                    qid = int(line.split()[1])
                    info[qid] = {'query': '', 'relevant': []}
            else:
                if current_op == "W":
                    info[qid]['query'] += line.strip() + " "
except FileNotFoundError:
    print(f"File '{query_file_path}' not found.\n")
    sys.exit(1)

# Open and process qrels.text
qrels_file_path = "C:\\Users\\mccut\\OneDrive\\Desktop\\cps842f23_a1_LukeMcCutcheon\\qrels"
try:
    with open(qrels_file_path, "r") as file_qrels:
        for line in file_qrels:
            id_q, id_d = map(int, line.split())
            if id_q in info:
                info[id_q]['relevant'].append(id_d)
except FileNotFoundError:
    print(f"File '{qrels_file_path}' not found.\n")
    sys.exit(1)

# Process queries and calculate scores
for k, v in info.items():
    if k == 0 or not v['relevant']:
        continue

    query = v['query'].lower()
    query_terms = re.sub(r'[^a-z0-9\' ]', ' ', query).split()
    
    if flag_stem or flag_stop:
        if flag_stop:
            query_terms = perform_stopword_removal(query_terms, stopwords)
        if flag_stem:
            stemmer = PorterStemmer()
            query_terms = perform_stemming(query_terms, stemmer)
        
        query = " ".join(query_terms)
    
    results = vsm.search(query)
    
    retrieved = [doc_id for doc_id, _ in results]
    
    # Compute R-Precision and MAP
    map_sum = 0.0
    num_rel = 0
    for i, doc_id in enumerate(retrieved[:25]):
        if doc_id in v['relevant']:
            num_rel += 1
            map_sum += num_rel / (i + 1)

    r_precision = len(set(retrieved[:len(v['relevant'])]) & set(v['relevant'])) / len(v['relevant'])
    mean_avg_precision = map_sum / len(v['relevant'])
    
    print(f"{k}:")
    print(f"    R-Precision = {r_precision:.4f}")
    print(f"    MAP = {mean_avg_precision:.4f}")
