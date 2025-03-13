# Import your libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    
    # Load and extract X and Y
    df = pd.read_csv('weather-data.csv')
    X = df.drop(['date','weather'],axis=1)
    y = df['weather']

    # Initialise kFold
    kf = KFold(n_splits=10,random_state=7,shuffle=True)
    
    score_lr = []
    score_dt = []
    score_rf = []
    scores = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test, y_train, y_test = X.iloc[train_index],X.iloc[test_index],y.iloc[train_index],y.iloc[test_index]

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Logistic Regression
        clf_lr = LogisticRegression(C=100,max_iter=2000, random_state = 7)
        clf_lr.fit(X_train, y_train)
        y_pred = clf_lr.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        score_lr.append(acc)

        # Decision Tree
        clf_dt = DecisionTreeClassifier(random_state = 7)
        clf_dt.fit(X_train, y_train)
        y_pred = clf_dt.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        score_dt.append(acc)

        # Random Forest
        clf_rf = RandomForestClassifier(n_estimators=100,criterion='gini', random_state = 7)
        clf_rf.fit(X_train, y_train)
        y_pred = clf_rf.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        score_rf.append(acc)

    # Return mean cross validation scores for each model
    scores.append(round(np.array(score_lr).mean(),2))
    scores.append(round(np.array(score_dt).mean(),2))
    scores.append(round(np.array(score_rf).mean(),2))
    return scores


# +

# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
