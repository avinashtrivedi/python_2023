import os
import re
import sys
from collections import defaultdict
from nltk.stem import PorterStemmer

# Paths for the main content and the stop words file
CACM_PATH = '../cacm.all'
STOPWORDS_PATH = '../stopwords.txt'

def load_stop_words():
    """Load stopwords from the provided path."""
    with open(STOPWORDS_PATH, 'r') as f:
        return set(f.read().splitlines())

def process_file(content, stop_words, apply_stemming=True, remove_stopwords=True):
    """Process file content to produce an inverted index."""
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

def main():
    stem_option = '-stem' in sys.argv
    stopword_option = '-stopwords' in sys.argv

    try:
        with open(CACM_PATH, 'r', encoding='latin1') as f:
            content = f.read().splitlines()

        stop_words = set()
        if stopword_option:
            stop_words = load_stop_words()

        inverted_index, documents = process_file(content, stop_words, apply_stemming=stem_option, remove_stopwords=stopword_option)
        save_index(inverted_index)
    except FileNotFoundError:
        print(f"Error: File not found at {CACM_PATH}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


