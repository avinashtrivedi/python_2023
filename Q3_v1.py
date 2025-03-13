import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def run_logistic_regression(filename):
    df = pd.read_csv(filename)
    X = df.drop('class',axis=1)
    y = df['class']
    clf = LogisticRegression(random_state=0).fit(X, y)
    y_pred = clf.predict(X)
    f1_wt = f1_score(y,y_pred,average='weighted')
    f1_m = f1_score(y,y_pred,average='macro')
    return f1_wt,f1_m
    
weighted_F1, macro_F1 = run_logistic_regression(r"D:\OneDrive - NITT\Custom_Download\q3_data.csv")
print('weighted_F1:',weighted_F1)
print('macro_F1:',macro_F1)