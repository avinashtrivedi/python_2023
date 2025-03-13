import pandas as pd
from sklearn import svm
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

def run_kfold_CV_SVM(filename,n_folds):
    df = pd.read_csv(filename)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = df['content']
    y = df['label']
    X = vectorizer.fit_transform(X)

    kf = KFold(n_splits=n_folds)

    out = []
    for i, (train_index, test_index) in tqdm(enumerate(kf.split(X))):
        X_train, y_train = X[train_index],y[train_index]
        X_test, y_test = X[test_index],y[test_index]

        clf = svm.SVC(random_state=42).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        d = classification_report(y_test,y_pred,output_dict=True)
        result = pd.DataFrame(d)
        out.append(result)
    out = sum(out)/n_folds
    out.rename({'comedy': 'class-comedy',
            'crime':'class-crime','drama': 'class-drama',
           'horror':'class-horror','western':'class-western'}, 
           axis=1,inplace=True)  
    
    out = out.T
    out.reset_index(inplace=True)
    out['Model'] = 'SVM'
    out.rename({'index': 'Class'}, axis=1,inplace=True)  
    out = out[['Model','Class','precision', 'recall', 'f1-score', 'support']]
    return out.sort_values('Class').reset_index(drop=True)

df_report_svm = run_kfold_CV_SVM("q5_text.csv", n_folds=3)
print(df_report_svm)