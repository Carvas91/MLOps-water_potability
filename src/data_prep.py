import pandas as pd
import numpy as np
import os


def load_data(filepath: str)-> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from filepath {filepath}: {e}")
#train_data = pd.read_csv("./data/raw/train.csv")
#test_data = pd.read_csv("./data/raw/test.csv")

def fill_missing_with_median(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values: {e}")


def save_data(df:pd.DataFrame, filepath:str)->None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f'Erro saving data to filepath:{filepath}: {e}')
#train_processed_data = fill_missing_with_median(train_data)
#test_processed_data = fill_missing_with_median(test_data)

def main():
    raw_data_path = 

data_path = os.path.join("data", "processed")

os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)


