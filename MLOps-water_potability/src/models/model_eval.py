import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path: str)-> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f'Error loading data file {file_path}: {e}')

#test_data = pd.read_csv('./data/processed/test_processed.csv')

#X_test = test_data.iloc[:,0:-1].values
#y_test = test_data.iloc[:,-1].values

def prepare_data(data: pd.DataFrame)->tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f'Error preparing data {e}')


#with open("model.pkl", "rb") as file:
#    model = pickle.load(file)

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f'Error loading model from path {file_path}: {e}')


def evaluation_model(model, X_test:pd.DataFrame,y_test:pd.Series )-> dict:
    try:
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)

        metrics_dict = {
            "acc": acc,
            "precision":pre,
            "recall": recall,
            "f1 score": f1score
        }

        return metrics_dict
    except Exception as e:
        raise Exception(f'Error avaluating the model: {e}')

def save_metrics(metrics_dict:dict, file_path: str)->None:
    try:
        with open("metrics.json", "w") as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        Exception(f'Error saving matrics to {file_path}: {e}')


def main():
    try:
        test_data_path = './data/processed/test_processed.csv'
        model_path = 'model.pkl'
        metrics_path = 'metrics.json'
        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f'Error in main function {e}')

if __name__ == '__main__':
    main()