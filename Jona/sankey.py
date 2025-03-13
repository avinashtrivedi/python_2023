"""
File: sankey.py
Description:  A simple library for building sankey diagrams from a dataframe
Author: John Rachlin
Date: etc.
"""


import plotly.graph_objects as go


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


def make_sankey(df, src, targ, vals=None, save=None, **kwargs):
    """
    Create a sankey diagram from a dataframe and specified columns
    :param df:
    :param src:
    :param targ:
    :param vals:
    :param save:
    :param kwargs:
    :return:
    """
    if vals:
        values = df[vals]
    else:
        values = [1] * len(df)

    # convert df labels to integer values
    df, labels = _code_mapping(df, src, targ)

    # stack columns if there are more than 2
    df = stack_columns(df)


    x = 5

    link = {'source': df[src], 'target': df[targ], 'value': values,
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
