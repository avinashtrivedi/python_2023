import os
import re
import sys
import math
from collections import defaultdict
from nltk.stem import PorterStemmer

CACM_PATH = '..\cacm.all'
STOPWORDS_PATH = '..\stopwords.txt'

def load_stop_words():
    with open(STOPWORDS_PATH, 'r') as f:
        return set(f.read().splitlines())

def process_file(content, stop_words, apply_stemming=True, remove_stopwords=True):
    documents = []
    document = {'ID': '', 'T': '', 'W': '', 'B': '', 'A': ''}
    current_field = ''

    for line in content:
        if line.startswith('.I'):
            if document['ID']:
                documents.append(document)
                document = {'ID': '', 'T': '', 'W': '', 'B': '', 'A': ''}
            document['ID'] = line[2:].strip()
        elif line.startswith('.T'):
            current_field = 'T'
        elif line.startswith('.W'):
            current_field = 'W'
        elif line.startswith('.B'):
            current_field = 'B'
        elif line.startswith('.A'):
            current_field = 'A'
        elif line.startswith('.'):
            current_field = ''
        elif current_field:
            document[current_field] += ' ' + line.strip()

    if document['ID']:
        documents.append(document)

    inverted_index = defaultdict(lambda: defaultdict(dict))
    stemmer = PorterStemmer()

    for document in documents:
        for field in ['T', 'W']:
            tokens = re.findall(r'\w+', document[field].lower())
            term_positions = defaultdict(list)

            for i, token in enumerate(tokens):
                if remove_stopwords and token in stop_words:
                    continue
                if apply_stemming:
                    token = stemmer.stem(token)
                term_positions[token].append(i)

            for token, pos in term_positions.items():
                if pos:
                    if document['ID'] not in inverted_index[token]:
                        inverted_index[token][document['ID']] = {'tf': 0, 'positions': []}
                    inverted_index[token][document['ID']]['tf'] += 1
                    inverted_index[token][document['ID']]['positions'].extend(pos)

    return inverted_index, documents

def save_index(inverted_index):
    with open('dictionary.txt', 'w') as dictionary_file, open('postings.txt', 'w') as postings_file:
        for term, postings in sorted(inverted_index.items()):
            doc_freq = len(postings)
            dictionary_file.write(f'{term} {doc_freq}\n')
            
            postings_list = []
            for doc_id, data in sorted(postings.items(), key=lambda x: int(x[0])):
                postings_entry = f"{doc_id} {data['tf']} [{' '.join(map(str, data['positions']))}]"
                postings_list.append(postings_entry)
            
            postings_file.write(f'{term}:' + '\n\t' + '\n\t'.join(postings_list) + '\n\n')

class VectorData:
    def __init__(self):
        self.norm = 0
        self.termid = []
        self.weight = []

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

def main():
    try:
        with open(CACM_PATH, 'r', encoding='latin1') as f:
            content = f.read().splitlines()

        stop_words = load_stop_words()
        inverted_index, documents = process_file(content, stop_words)
        save_index(inverted_index)

        number_of_docs = len(documents)
        vsm = VectorSpaceModel(inverted_index, stop_words, number_of_docs)

        doc_id_to_metadata = {doc['ID']: {'title': doc['T'], 'author': doc.get('A', '')} for doc in documents}

        while True:
            query = input("\nEnter Query: ")
            if query == "ZZEND":
                break

            results = vsm.search(query)
            if not results or results[0][1] == 0:
                print("\nQuery did not match any documents.\n")
                continue

            num_results_to_show = min(10, len(results))
            for i, r in enumerate(results[:num_results_to_show]):
                doc_id = r[0]
                title = doc_id_to_metadata[doc_id]['title']
                author = doc_id_to_metadata[doc_id]['author']
                print(f"\n{str(i+1).zfill(2)}) Document ID: {doc_id}, Title: {title}, Author: {author}, Similarity Score: {r[1]:.4f}\n")

    except FileNotFoundError:
        print(f"Error: File not found at {CACM_PATH}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


