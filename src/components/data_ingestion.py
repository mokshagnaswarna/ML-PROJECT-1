
#artifact folder should be created and train.csv and test.csv should be there in artifact folder after running this file.
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import sys
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        print("STEP 1: Function started")

        try:
            file_path = os.path.join('src', 'pipeline', 'notebook', 'stud.csv')
            print("STEP 2:", file_path)

            df = pd.read_csv(file_path)
            print("STEP 3: CSV Loaded")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            print("STEP 4: Folder created")

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            from sklearn.model_selection import train_test_split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            print("STEP 5: Split done")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            print("STEP 6: Files saved ✅")

        except Exception as e:
            print("ERROR OCCURRED:", e)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()