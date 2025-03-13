"""
file: df_stacking.py
description: How to convert a multi-column df into a 2-column dataframe
by stacking pairs of columns
"""

import pandas as pd
import sankey_hw as sk

art = pd.DataFrame({'nationality':['A', 'A', 'B', 'C', 'D'],
                    'gender': ['M', 'M', 'M', 'F', 'M'],
                    'decade':['1930', '1950', '1940', '1930', '1940'],
                    'num_artists':[3, 4, 5, 1, 2]})

print(art)
#sk.make_sankey(art, 'nationality', 'gender')
#sk.make_sankey(art, 'gender', 'decade')


art

ng = art[['nationality', 'gender']]
ng.columns = ['src', 'targ']
# print(ng)

gd = art[['gender', 'decade']]
gd.columns = ['src', 'targ']
# print(gd)

stacked = pd.concat([ng, gd], axis=0)
# print(stacked)

stacked

sk.make_sankey(stacked, "src", "targ")


def make_sankey(df, *cols, vals=None, **kwargs):
    pass


