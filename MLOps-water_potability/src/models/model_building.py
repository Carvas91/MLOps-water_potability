import pandas as pd
import numpy as np
import os
import yaml

import pickle

from sklearn.ensemble import RandomForestClassifier


def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from path {filepath}: {e}")
#train_data = pd.read_csv('./data/processed/train_processed.csv')

#X_train=train_data.iloc[:,0:-1].values
#y_train= train_data.iloc[:,-1].values


def prepare_data(data: pd.DataFrame)->tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=["Potability"],axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"error preparing data {e}")

#X_train = train_data.drop(columns=["Potability"],axis=1)
#y_train = train_data['Potability']


def load_params(params_path: str)-> int:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        return params ["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"error when loading params from path {params_path}: {e}")
#n_estimators = yaml.safe_load(open("./params.yaml"))["model_building"]["n_estimators"]


def train_model(X: pd.DataFrame,y: pd.Series, n_estimators:int)->RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y)
        return clf
    except Exception as e:
        raise Exception(f'Error in training model: {e}')



def save_model(model: RandomForestClassifier,filepath:str)->None:
    try:
        with open(filepath,'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f'Error saving model to fileppath {filepath}: {e}')

    
    
#with open("model.pkl","wb") as file:
 #   pickle.dump(clf, file)


def main():
    try:
        params_path = 'params.yaml'
        data_path = './data/processed/train_processed.csv'
        model_name = "model.pkl"

        n_estimators = load_params(params_path)
        train_data = load_data(data_path)
        X_train,y_train =prepare_data(train_data)
        
        model =train_model(X_train,y_train,n_estimators)
        save_model(model, model_name)
    except Exception as e:
        raise Exception(f'Error in main function:{e}')




if __name__ == "__main__":
    main()