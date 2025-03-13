
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg


### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and \
                isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str):  # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(
                        bps))
                return False
            if len(bps) != 2:
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(
                        bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp) != 3:
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(
                            bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(
                            bp))
                    return False
    return True


def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True


class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        parse_table = defaultdict(lambda: defaultdict(list))
        tokens_len = len(tokens)
        for indx,token in enumerate(tokens):
            for lhs, rhs, prob in self.grammar.rhs_to_rules[(token,)]:
                parse_table[indx, indx + 1][lhs] = rhs[0]

        for tok_len in range(2, tokens_len + 1):
            for i in range(0, tokens_len - tok_len + 1):
                j = i + tok_len
                for k in range(i + 1, j):
                    for val_1 in parse_table[(i, k)]:
                        for val_2 in parse_table[(k, j)]:
                            for lhs, rhs, prob in self.grammar.rhs_to_rules[(val_1, val_2)]:
                                parse_table[(i, j)][lhs] = ((val_1, i, k), (val_2, k, j))
        if parse_table[(0, tokens_len)][self.grammar.startsymbol]:
            return True
        return False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        table = defaultdict(lambda: defaultdict(list))
        probs = defaultdict(lambda: defaultdict(list))
        tokens_len = len(tokens)
        for indx,token in enumerate(tokens):
            for lhs, rhs, prob in self.grammar.rhs_to_rules[(token,)]:
                table[(indx, indx+1)][lhs] = rhs[0]
                probs[(indx, indx+1)][lhs] = math.log2(prob)

        for tok_len in range(2, tokens_len + 1):
            for i in range(0, tokens_len - tok_len + 1):
                j = i + tok_len
                for k in range(i + 1, j):
                    for val_1 in table[(i, k)]:
                        for val_2 in table[(k, j)]:
                            for lhs, rhs, prob in self.grammar.rhs_to_rules[(val_1, val_2)]:
                                log_prob = math.log2(prob) + probs[(i, k)][val_1] + probs[(k, j)][val_2]
                                if isinstance(probs[(i, j)][lhs], list) or log_prob >= probs[(i, j)][lhs]:
                                    table[(i, j)][lhs] = ((val_1, i, k), (val_2, k, j))
                                    probs[(i, j)][lhs] = log_prob

        return table, probs


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in the non-terminal 'nt' and covering the span from 'i' to 'j'.
    """
    chart_entry = chart[(i, j)][nt]
    
    if isinstance(chart_entry, list):
        raise KeyError("Chart entry is a list.")
    elif isinstance(chart_entry, str):
        return (nt, chart_entry)
    else:
        x, y = chart_entry
        left_subtree = get_tree(chart, x[1], x[2], x[0])
        right_subtree = get_tree(chart, y[1], y[2], y[0])
        return (nt, left_subtree, right_subtree)



if __name__ == "__main__":
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
