import pandas as pd
import numpy as np
import os
import yaml

import pickle

from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('./data/processed/train_processed.csv')

#X_train=train_data.iloc[:,0:-1].values
#y_train= train_data.iloc[:,-1].values

X_train = train_data.drop(columns=["Potability"],axis=1)
y_train = train_data['Potability']

n_estimators = yaml.safe_load(open("./src/params.yaml"))["model_building"]["n_estimators"]

clf = RandomForestClassifier(n_estimators=n_estimators)
clf.fit(X_train, y_train)

with open("model.pkl","wb") as file:
    pickle.dump(clf, file)