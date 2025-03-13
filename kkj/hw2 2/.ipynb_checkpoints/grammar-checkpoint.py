import math
import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self, rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        NonTerm = list(self.lhs_to_rules)
        for lhs in self.lhs_to_rules.keys():
            sum_prob = 0
            for rule in self.lhs_to_rules[lhs]:
                lhs, rhs, prob = rule
                sum_prob = sum_prob + prob
                if len(rhs) == 1:
                    if rhs[0] in NonTerm:
                        return False
                elif len(rhs) == 2:
                    if rhs[0] not in NonTerm or rhs[1] not in NonTerm:
                        return False
                else:
                    return False
            if not math.isclose(sum_prob, 1):
                return False

        return True


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        if grammar.verify_grammar()==True:
            print("Confirmation message: The grammar is a valid PCFG in CNF.")
        else:
            print("ERROR message: The grammar is not a valid PCFG in CNF")

