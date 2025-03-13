import re, string
from generic_func import df_apply



punc_no_sq = '!“#$%&\()*+,./:;<=>?@[\\]^_`{|}~“”—’-"'
def remove_punctuation(text):
    for punctuation in punc_no_sq:
        text = text.replace(punctuation, '')
    return text


def lower_case(text):
    text = text.lower()
    return text

def apply_cleaning(df):

    df_apply(df, 'text_clean', 'text_clean', remove_punctuation)
    df_apply(df,'text', 'text_clean', lower_case)


    return df
