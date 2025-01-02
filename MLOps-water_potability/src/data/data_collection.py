import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml




def load_params(filepath: str)-> float:
    try:
        with open (filepath, 'r') as file:
            params = yaml.safe_load(file)
            return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}:{e}")

#test_size = yaml.safe_load(open("./params.yaml"))["data_collection"]["test_size"]

def load_data(filepath: str)-> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}: {e}')
#data = pd.read_csv('./water_potability.csv')


def split_data(data: pd.DataFrame, test_size: float)-> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        return train_test_split(data, test_size=test_size, random_state=41)
    except ValueError as e:
        raise ValueError(f'Error splitting data: {e}')
#train_data, test_data = train_test_split(data, test_size=test_size, random_state=41)

def save_data(df: pd.DataFrame, filepath: str)-> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f'Erro saving data to {filepath}: {e}')

def main():
    try:
        data_filepath = './water_potability.csv'
        params_filepath = './params.yaml'
        raw_data_path =  os.path.join("data","raw")

        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size=test_size)
        os.makedirs(raw_data_path)
        save_data(train_data, os.path.join(raw_data_path, 'train.csv'))
        save_data(test_data, os.path.join(raw_data_path, 'test.csv'))
    except Exception as e:
        raise Exception(f"An error ocurred {e}")



if __name__ == "__main__":
    main()

#data_path = os.path.join("data","raw")

#os.makedirs(data_path)
#train_data.to_csv(os.path.join(data_path,"train.csv"),index = False)
#test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)