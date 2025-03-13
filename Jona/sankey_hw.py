"""
File: sankey.py
Description:  A simple library for building sankey diagrams from a dataframe
Author: John Rachlin
Date: etc.
"""


import plotly.graph_objects as go
import pandas as pd


def _code_mapping(df, src, targ):
    # get the distinct labels from src/targ columns
    labels = list(set(list(df[src]) + list(df[targ])))

    # generate n integers for n labels
    codes = list(range(len(labels)))

    # create a map from label to code
    lc_map = dict(zip(labels, codes))

    # substitute names for codes in the dataframe
    df = df.replace({src: lc_map, targ: lc_map})

    # Return modified dataframe and list of labels
    return df, labels


def stack_columns(df, col1, col2, cols):
    col_list = [col1, col2]
    for col in cols:
        col_list.append(col)

    pairs = []
    for i in range(len(col_list) - 1):
        stack_pair = df[[col_list[i], col_list[i + 1]]]
        stack_pair.columns = ['src', 'targ']
        pairs.append(stack_pair)

    stacked = pd.concat(pairs, axis=0)
    return stacked


def make_sankey(df, col1, col2, *cols, vals=None, save=None, **kwargs):
    """
    Create a sankey diagram from a dataframe and specified columns
    :param df:
    :param col1:
    :param col2:
    :param cols:
    :param vals:
    :param save:
    :param kwargs:
    :return:
    """

    # stack columns if there are more than 2
    col_list = []
    for col in cols:
        col_list.append(col)
    df_stacked = stack_columns(df, col1, col2, col_list)
#     print(df_stacked.dtypes)
    # convert df labels to integer values
    df_stacked, labels = _code_mapping(df_stacked, 'src', 'targ')
    
    
    values = [1] * len(df_stacked)
        
    link = {'source': df_stacked['src'], 'target': df_stacked['targ'], 'value': values,
            'line': {'color': 'black', 'width': 1}}

    node_thickness = kwargs.get("node_thickness", 50)

    node = {'label': labels, 'pad': 50, 'thickness': node_thickness,
            'line': {'color': 'black', 'width': 1}}

    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)

    # For dashboarding, you will want to return the fig
    # rather than show the fig.

    fig.show()

    # This requires installation of kaleido library
    # https://pypi.org/project/kaleido/
    # See: https://anaconda.org/conda-forge/python-kaleido
    # conda install -c conda-forge python-kaleido

    if save:
        fig.write_image(save)
