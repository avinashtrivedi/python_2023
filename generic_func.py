import pandas as pd

def df_apply(df, column, result_column, fn):
  df[result_column]=df[column].apply(fn)
