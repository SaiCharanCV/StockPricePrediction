import pandas as pd
from pathlib import Path

class CSVLoader:
    def __init__(self,dataset_name):
        self.dataset = dataset_name

    def data_loader(self):
        if not Path(self.dataset).exists():
            raise FileNotFoundError(f"The dataset {self.dataset} does not exist.")
        try:
            df = pd.read_csv(self.dataset)
            print(f"Dataset {self.dataset} loaded successfully.")
            return df
        except ValueError as ve:
            raise ValueError(f"Invalid value encountered while loading the dataset. Details: {ve}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the dataset. Details: {e}")
    
    def get_data(self):
        self.df = self.data_loader()
        self.df.set_index(self.df.Date, inplace=True)
        self.df.index = pd.to_datetime(self.df.index)
        return self.df
    
    ## basic dataset operations

    def get_dataset_size(self):
        return self.df.shape
    
    def get_dataset_columns(self):
        return self.df.columns.tolist()
    
    def get_na_count(self):
        return self.df.isna().sum()
    
data = "Stock Price.csv"
dataset = CSVLoader(dataset_name=data)

try:
    dataset.get_data()
    print(f"Dataset loaded successfully with shape: {dataset.get_dataset_size()}")
except Exception as e:
    raise Exception(f"Failed to load dataset. Details: {e}")

### doing basic dataset validation

print("Columns in the dataset:", dataset.get_dataset_columns())
print("Missing values in the dataset:", dataset.get_na_count())
print(f"Dataset Size is: {dataset.get_dataset_size()}")