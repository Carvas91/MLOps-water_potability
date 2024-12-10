import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml




def load_params(filepath: str)-> float:
    with open (filepath, 'r') as file:
        params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
#test_size = yaml.safe_load(open("./params.yaml"))["data_collection"]["test_size"]

def load_data(filepath: str)-> pd.DataFrame:
    return pd.read_csv(filepath)
#data = pd.read_csv('./water_potability.csv')


def split_data(data: pd.DataFrame, test_size: float)-> tuple[pd.DataFrame,pd.DataFrame]:
    return train_test_split(data, test_size=test_size, random_state=41)
#train_data, test_data = train_test_split(data, test_size=test_size, random_state=41)

def save_data(df: pd.DataFrame, filepath: str)-> None:
    df.to_csv(filepath, index=False)

def main():
    data_filepath = './water_potability.csv'
    params_filepath = './params.yaml'
    raw_data_path =  os.path.join("data","raw")

#data_path = os.path.join("data","raw")

os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path,"train.csv"),index = False)
test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)